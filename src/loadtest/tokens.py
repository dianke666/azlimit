"""Token 工具：封装 tiktoken ``o200k_base``（gpt-4o / gpt-4.1 系列使用的编码）。

- ``count(text)``：文本 token 数。
- ``count_messages(...)``：按 OpenAI chat 格式计入消息开销（每条消息 3 token、name 1 token、
  回复引导 3 token），用于让组装器对完整消息复算的 input_tokens 落在 ±20 内（Req 14.1）。
- ``trim_to_tokens(text, max_tokens)``：先在句子边界裁剪，再按 token 截断，结果 token 数
  永远不超过预算（Req 14.8）。支持中文（。！？；）与英文（. ! ?）句末标点。

编码文件首次使用时由 tiktoken 下载；未设置 ``TIKTOKEN_CACHE_DIR`` 时缓存到
``~/.cache/loadtest/tiktoken``，避免系统临时目录被清理后重复下载。
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import tiktoken

ENCODING_NAME = "o200k_base"

# OpenAI chat 格式开销（gpt-4o / gpt-4.1 系列沿用 gpt-4 的约定）
TOKENS_PER_MESSAGE = 3  # <|start|>{role}\n ... <|end|>
TOKENS_PER_NAME = 1
REPLY_PRIMING_TOKENS = 3  # 每次回复以 <|start|>assistant<|message|> 引导

# 句子边界：中文句末标点无需空白；英文句末标点需后接空白或文本结尾（避免切开 3.14、e.g.x）。
# 句末标点后的右引号/右括号归属于当前句。
_CLOSERS = "”’\"'）)」』】》]"
# 注意：_CLOSERS 含 "]"，嵌入字符类前必须转义，否则会提前闭合字符类
_CLOSERS_CLASS = f"[{re.escape(_CLOSERS)}]"
_SENTENCE_END_RE = re.compile(
    rf"(?:[。！？；!?]+|[.!?]+(?={_CLOSERS_CLASS}*(?:\s|$))){_CLOSERS_CLASS}*"
)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "loadtest" / "tiktoken"


def _ensure_cache_dir() -> None:
    if "TIKTOKEN_CACHE_DIR" not in os.environ and "DATA_GYM_CACHE_DIR" not in os.environ:
        _DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(_DEFAULT_CACHE_DIR)


@lru_cache(maxsize=1)
def get_encoding() -> tiktoken.Encoding:
    """返回进程内共享的 o200k_base 编码（首次调用可能触发下载）。"""
    _ensure_cache_dir()
    return tiktoken.get_encoding(ENCODING_NAME)


class TokenCounter:
    """o200k_base 计数器；组装器、过滤器等组件以 ``enc`` 形式注入。"""

    def __init__(self, encoding: tiktoken.Encoding | None = None) -> None:
        self.encoding = encoding or get_encoding()

    # ---- 基础 -------------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        # disallowed_special=()：语料中出现 "<|endoftext|>" 等字面量时按普通文本计数
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens: Sequence[int]) -> str:
        """解码 token 前缀；截断处若落在多字节字符中间，丢弃不完整的字节。"""
        data = self.encoding.decode_bytes(list(tokens))
        return data.decode("utf-8", errors="ignore")

    def count(self, text: str) -> int:
        return len(self.encode(text)) if text else 0

    # ---- 消息开销 ---------------------------------------------------------
    def count_messages(self, *args: Any) -> int:
        """按 OpenAI chat 格式计算完整请求的 input token 数（含消息开销与回复引导）。

        两种调用方式：
        - ``count_messages(messages)``：``messages`` 为 ``[{"role": ..., "content": ..., "name"?: ...}]``。
        - ``count_messages(system_prompt, user_prefix, user_suffix)``：计算指令部分的固定开销
          （system 消息 + 仅含前后缀的 user 消息 + 回复引导），用于组装器求段落预算；
          前缀/后缀按各自独立计数，空字符串计 0。
        """
        if len(args) == 1 and not isinstance(args[0], str):
            return self.count_messages_full(args[0])
        if not args or not all(isinstance(a, str) for a in args):
            raise TypeError("count_messages expects a message list or (system_prompt, user_prefix, user_suffix) strings")
        system_prompt, *user_parts = args
        total = REPLY_PRIMING_TOKENS
        if system_prompt:
            total += TOKENS_PER_MESSAGE + self.count("system") + self.count(system_prompt)
        total += TOKENS_PER_MESSAGE + self.count("user") + sum(self.count(p) for p in user_parts)
        return total

    def count_messages_full(self, messages: Iterable[Mapping[str, Any]]) -> int:
        """对完整消息列表复算 input token 数。"""
        total = REPLY_PRIMING_TOKENS
        for msg in messages:
            total += TOKENS_PER_MESSAGE
            for key, value in msg.items():
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise TypeError(f"message field {key!r} must be str, got {type(value).__name__}")
                total += self.count(value)
                if key == "name":
                    total += TOKENS_PER_NAME
        return total

    # ---- 裁剪 -------------------------------------------------------------
    def trim_to_tokens(self, text: str, max_tokens: int, *, slack: int = 8) -> str:
        """把 ``text`` 裁剪到不超过 ``max_tokens`` 个 token。

        步骤：
        1. 已在预算内则原样返回。
        2. 取前 ``max_tokens`` 个 token 的解码前缀；若其中最后一个句子边界处的长度
           ≥ ``max_tokens - slack``，在该句子边界裁剪（结尾完整、误差不超过 slack）。
        3. 否则按 token 截断（保证组装器的 ±20 拟合精度）。
        结果 token 数恒 ≤ ``max_tokens``；``max_tokens <= 0`` 返回空串。
        """
        if max_tokens <= 0 or not text:
            return ""
        tokens = self.encode(text)
        if len(tokens) <= max_tokens:
            return text

        hard = self.decode(tokens[:max_tokens])
        result = hard.rstrip()
        cut = _last_sentence_end(hard)
        if cut is not None:
            candidate = hard[:cut].rstrip()
            if candidate and self.count(candidate) >= max_tokens - max(0, slack):
                result = candidate
        return self._enforce_budget(result, max_tokens)

    def _enforce_budget(self, text: str, max_tokens: int) -> str:
        """解码前缀重新编码时 BPE 合并可能略有不同；循环截断直到严格满足预算。"""
        while text:
            toks = self.encode(text)
            if len(toks) <= max_tokens:
                return text
            # 每轮至少缩短 1 个 token，保证终止
            text = self.decode(toks[: min(max_tokens, len(toks) - 1)]).rstrip()
        return text


def _last_sentence_end(text: str) -> int | None:
    """返回最后一个句子边界（句末标点及其后的右引号/括号之后）的字符下标。"""
    end = None
    for m in _SENTENCE_END_RE.finditer(text):
        end = m.end()
    return end


@lru_cache(maxsize=1)
def default_counter() -> TokenCounter:
    return TokenCounter()


# ---- 模块级便捷函数 ------------------------------------------------------
def count(text: str) -> int:
    return default_counter().count(text)


def count_messages(*args: Any) -> int:
    return default_counter().count_messages(*args)


def count_messages_full(messages: Iterable[Mapping[str, Any]]) -> int:
    return default_counter().count_messages_full(messages)


def trim_to_tokens(text: str, max_tokens: int, enc: TokenCounter | None = None, *, slack: int = 8) -> str:
    """与设计伪代码 ``trim_to_tokens(text, n, enc)`` 一致的函数式入口。"""
    return (enc or default_counter()).trim_to_tokens(text, max_tokens, slack=slack)


__all__ = [
    "ENCODING_NAME",
    "TOKENS_PER_MESSAGE",
    "TOKENS_PER_NAME",
    "REPLY_PRIMING_TOKENS",
    "TokenCounter",
    "get_encoding",
    "default_counter",
    "count",
    "count_messages",
    "count_messages_full",
    "trim_to_tokens",
]
