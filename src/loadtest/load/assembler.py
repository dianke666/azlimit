"""RequestAssembler：请求组装与精确长度拟合（design.md 组件 7 / 算法 3）。

流程（Requirements 9.3, 9.5, 14.1–14.9, 15.1, 15.2）：

1. ``choose_lang`` 按 ``lang_ratio`` 选语言（zh 默认 0.60），只会是 zh / en。
2. 同语言任务指令均匀抽样（``TaskCatalog.sample``），指令位置随机 before / after。
3. 段落预算 = target − 指令与消息开销；按预算分档确定初始段落数
   （< 2000 → 1–2；[2000, 4000) → 2–3；否则 3–4），逐段 ``pick(near=剩余/剩余槽位)``。
4. 随机打乱段落顺序；段落之间以空行分隔。
5. 不足 > tolerance：段落 < 4 且有候选时追加补足片段；仍不足时用更长的同语言候选替换
   最短段落（结果仍满足 1–4 段、互不重复、复用上限）。
6. 拟合：最后一段始终由其来源段落全文重新裁剪（句子边界 → token 截断），对完整消息复算
   token 并按差值迭代调整，直到 ``|input_tokens − target| ≤ tolerance``。
7. 成功后对全部所用段落（含补足片段来源）``mark_used``；失败不修改 run_uses。

异常：
- ``CorpusExhausted``：请求语言下没有任何满足复用上限的候选段落。
- ``TargetUnreachable``（``CorpusExhausted`` 子类）：有候选，但 ≤ 4 个不重复段落的总长
  无法达到目标长度（例如 target 接近 6000 而剩余长段落不足）。调度器若只捕获
  ``CorpusExhausted`` 会安全地停止；也可单独捕获它并重新采样目标长度。
"""
from __future__ import annotations

import uuid
from collections.abc import Mapping
from random import Random
from typing import Any

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage
from loadtest.load.models import PlannedRequest
from loadtest.load.tasks import TaskCatalog, TaskSpec
from loadtest.tokens import TokenCounter, default_counter

SEP = "\n\n"
MAX_PASSAGES = 4
DEFAULT_TOLERANCE = 20
DEFAULT_MAX_TOKENS = 40
DEFAULT_LANG_RATIO: dict[str, float] = {"zh": 0.60, "en": 0.40}
_MIN_KEEP = 100  # 超出预算时优先裁剪裁后仍能保留 ≥ 该 token 数的段落
_UPGRADE_SAMPLES = 48  # 不足时为寻找更长候选而额外抽样的次数
_FIT_ITERS = 8


class CorpusExhausted(RuntimeError):
    """请求语言下没有满足复用上限的候选段落（Req 9.3）。"""

    def __init__(self, lang: str, message: str | None = None) -> None:
        self.lang = lang
        super().__init__(message or f"corpus exhausted for lang={lang!r}")


class TargetUnreachable(CorpusExhausted):
    """候选段落不足以在 ≤ 4 个不重复段落内达到目标长度。"""

    def __init__(self, lang: str, target: int, reachable: int) -> None:
        self.target = target
        self.reachable = reachable
        super().__init__(lang, f"target {target} unreachable for lang={lang!r} (max ≈ {reachable})")


def choose_lang(rng: Random, ratio: Mapping[str, float] | None = None) -> str:
    """以 ratio['zh'] 的概率返回 'zh'，否则 'en'（Req 15.1, 15.2）。"""
    r = DEFAULT_LANG_RATIO if ratio is None else ratio
    return "zh" if rng.random() < float(r["zh"]) else "en"


def choose_passage_count(budget: int, rng: Random) -> int:
    """按段落预算分档确定初始段落数（Req 14.9）。"""
    if budget < 2000:
        return rng.randint(1, 2)
    if budget < 4000:
        return rng.randint(2, 3)
    return rng.randint(3, 4)


def build_messages(task: TaskSpec, parts: list[str], position: str) -> list[dict]:
    body = SEP.join(parts)
    if position == "before":
        user = f"{task.instruction}{SEP}{body}"
    else:
        user = f"{body}{SEP}{task.instruction}"
    return [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": user},
    ]


class RequestAssembler:
    def __init__(
        self,
        index: PassageIndex,
        tasks: TaskCatalog,
        counter: TokenCounter | None = None,
        cfg: Any = None,
        *,
        lang_ratio: Mapping[str, float] | None = None,
        max_tokens: int | None = None,
        tolerance: int = DEFAULT_TOLERANCE,
    ) -> None:
        self.index = index
        self.tasks = tasks
        self.counter = counter or default_counter()
        if lang_ratio is None:
            lang_ratio = getattr(cfg, "lang_ratio", None) or DEFAULT_LANG_RATIO
        if set(lang_ratio) != {"zh", "en"}:
            raise ValueError("lang_ratio keys must be exactly {zh, en}")
        self.lang_ratio = dict(lang_ratio)
        if max_tokens is None:
            max_tokens = getattr(cfg, "max_tokens", None) or DEFAULT_MAX_TOKENS
        self.max_tokens = int(max_tokens)
        self.tolerance = int(tolerance)
        self._sep_tokens = self.counter.count(SEP)

    # ------------------------------------------------------------------ 辅助

    def _full_len(self, ps: list[Passage]) -> int:
        if not ps:
            return 0
        return sum(p.token_len for p in ps) + self._sep_tokens * (len(ps) - 1)

    def _overhead(self, task: TaskSpec, position: str) -> int:
        """指令 + system + 消息开销 + 指令与正文之间的分隔符（正文为空时）。"""
        return self.counter.count_messages_full(build_messages(task, [""], position))

    def _upgrade(self, lang: str, order: list[Passage], budget: int, rng: Random, now: float | None) -> None:
        """不足时用更长的候选替换最短段落（原地修改 order）。"""
        exclude = {p.id for p in order}
        cands: list[Passage] = []
        for _ in range(_UPGRADE_SAMPLES):
            c = self.index.pick(lang, near=10**9, exclude=exclude, rng=rng, now=now)
            if c is None:
                break
            exclude.add(c.id)
            cands.append(c)
        cands.sort(key=lambda p: p.token_len, reverse=True)
        for c in cands:
            if self._full_len(order) >= budget:
                return
            i = min(range(len(order)), key=lambda j: order[j].token_len)
            if c.token_len <= order[i].token_len:
                return  # 候选已按长度降序，后面不会更长
            order[i] = c

    # ------------------------------------------------------------------ 组装

    def assemble(self, target: int, rng: Random, now: float | None = None) -> PlannedRequest:
        tol = self.tolerance
        idx = self.index
        lang = choose_lang(rng, self.lang_ratio)
        task = self.tasks.sample(rng, lang)
        position = "before" if rng.random() < 0.5 else "after"
        overhead = self._overhead(task, position)
        budget = int(target) - overhead

        # 1) 初始段落（Req 14.9）
        k = choose_passage_count(budget, rng)
        chosen: list[Passage] = []
        used = 0
        for i in range(k):
            sep = self._sep_tokens if chosen else 0
            remaining = budget - used - sep
            want = max(1, remaining // (k - i))
            p = idx.pick(lang, near=want, exclude={c.id for c in chosen}, rng=rng, now=now)
            if p is None:
                break
            chosen.append(p)
            used += p.token_len + sep
            if used >= budget:
                break
        if not chosen:
            raise CorpusExhausted(lang)
        initial_count = len(chosen)

        order = list(chosen)
        rng.shuffle(order)

        # 2) 不足：追加补足片段（Req 14.8）
        while budget - self._full_len(order) > tol and len(order) < MAX_PASSAGES:
            deficit = budget - self._full_len(order) - self._sep_tokens
            f = idx.pick(lang, near=max(1, deficit), exclude={p.id for p in order}, rng=rng, now=now)
            if f is None:
                break
            order.append(f)

        # 3) 仍不足：用更长候选替换最短段落（目标：全文总长 ≥ 预算，给 BPE 边界留余量）
        if budget - self._full_len(order) > tol // 2:
            self._upgrade(lang, order, budget, rng, now)
            if budget - self._full_len(order) > tol:
                raise TargetUnreachable(lang, int(target), self._full_len(order) + overhead)

        # 4) 超出：把一个裁剪后仍保留足够内容的段落放到最后（被裁剪的总是最后一段）
        over = self._full_len(order) - budget
        if over > 0 and order[-1].token_len - over < _MIN_KEEP:
            eligible = [j for j, p in enumerate(order) if p.token_len - over >= _MIN_KEEP]
            j = rng.choice(eligible) if eligible else max(range(len(order)), key=lambda j: order[j].token_len)
            order.append(order.pop(j))

        # 5) 拟合：最后一段由来源全文重新裁剪，按完整消息复算迭代调整（Req 14.1, 14.8）
        messages, n = self._fit(task, position, order, int(target))
        if abs(n - int(target)) > tol:
            raise TargetUnreachable(lang, int(target), n)

        for p in order:
            idx.mark_used(p.id)  # Req 9.5（含补足片段来源）

        return PlannedRequest(
            request_id=str(uuid.uuid4()),
            lang=lang,  # type: ignore[arg-type]
            messages=messages,
            input_tokens=n,
            max_tokens=self.max_tokens,
            passage_ids=tuple(p.id for p in order),
            task_id=task.id,
            target_tokens=int(target),
            instruction_position=position,  # type: ignore[arg-type]
            initial_passage_count=initial_count,
            filler_count=len(order) - initial_count,
            budget_tokens=budget,
        )

    def _fit(self, task: TaskSpec, position: str, order: list[Passage], target: int) -> tuple[list[dict], int]:
        counter = self.counter
        fixed = [p.text for p in order[:-1]]
        src = order[-1].text
        src_len = counter.count(src)

        def render(m: int, slack: int) -> tuple[list[dict], int]:
            last = src if m >= src_len else counter.trim_to_tokens(src, m, slack=slack)
            msgs = build_messages(task, fixed + [last], position)
            return msgs, counter.count_messages_full(msgs)

        # 初值：完整消息复算一次，按差值确定最后一段的 token 数
        msgs, n = render(src_len, 0)
        best = (msgs, n)
        m = src_len - (n - target)
        for slack in (8, 0):
            for _ in range(_FIT_ITERS):
                m = max(1, min(src_len, m))
                msgs, n = render(m, slack)
                if abs(n - target) < abs(best[1] - target):
                    best = (msgs, n)
                diff = n - target
                if abs(diff) <= self.tolerance // 2 or (diff < 0 and m >= src_len):
                    break
                if diff == 0:
                    break
                m -= diff
            if abs(best[1] - target) <= self.tolerance:
                break
            # 句子边界裁剪仍不满足时，以 slack=0（纯 token 截断）从当前 m 继续
        return best


__all__ = [
    "RequestAssembler",
    "CorpusExhausted",
    "TargetUnreachable",
    "choose_lang",
    "choose_passage_count",
    "build_messages",
    "SEP",
    "MAX_PASSAGES",
]
