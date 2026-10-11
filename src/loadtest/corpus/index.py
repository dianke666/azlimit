"""PassageIndex：按 (lang, length_bucket) 分桶的内存段落索引。

只被调度协程访问（单线程，无锁）。职责：
- 加权抽样：权重 ``1 / (1 + use_count + run_uses)^α``（Req 9.2）
- 单次运行复用上限：``run_uses < per_run_max_uses``（Req 9.1）
- 过期：``pick`` 永不返回 ``expires_at <= now`` 的段落；``expire_due`` 移除过期段落（Req 4.3, 4.4）
- 隔离/过期移除后同一运行内不可再加入（Req 24.2）
- ``ingest_pending`` 只用 ``get_nowait``，从不 await（Req 10.3）

确定性：桶内段落按加入顺序存放（移除使用 swap-remove，仍是确定性的），抽样只消费
``rng.random()`` 一次，因此相同种子 + 相同初始索引得到相同的选取序列。
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable, Mapping
from random import Random
from typing import Any

from loadtest.corpus.store import Passage

DEFAULT_BUCKETS: dict[str, tuple[int, int]] = {"S": (500, 799), "M": (800, 1199), "L": (1200, 1500)}

BucketKey = tuple[str, str]  # (lang, bucket)


def _interval_distance(lo: int, hi: int, x: float) -> float:
    """x 到闭区间 [lo, hi] 的距离（包含时为 0）。"""
    if x < lo:
        return lo - x
    if x > hi:
        return x - hi
    return 0.0


class PassageIndex:
    def __init__(
        self,
        passages: Iterable[Passage] = (),
        per_run_max_uses: int = 4,
        alpha: float = 1.0,
        buckets: Mapping[str, tuple[int, int]] | None = None,
        clock: Any = None,
    ) -> None:
        if per_run_max_uses < 1:
            raise ValueError("per_run_max_uses must be >= 1")
        self.per_run_max_uses = int(per_run_max_uses)
        self.alpha = float(alpha)
        self.buckets: dict[str, tuple[int, int]] = dict(buckets or DEFAULT_BUCKETS)
        # 按下界排序的桶名，用于确定性的距离排序与 tie-break
        self._bucket_order = sorted(self.buckets, key=lambda b: (self.buckets[b][0], self.buckets[b][1], b))
        self._clock = clock

        self._passages: dict[int, Passage] = {}
        self._key_of: dict[int, BucketKey] = {}
        self._lists: dict[BucketKey, list[int]] = {}
        self._pos: dict[int, int] = {}  # pid -> 在其桶列表中的下标
        self._removed: set[int] = set()
        self._run_uses: dict[int, int] = {}

        for p in passages:
            self.ingest(p)

    # ------------------------------------------------------------------ 辅助

    def _now(self) -> float:
        if self._clock is not None:
            wall = getattr(self._clock, "wall", None)
            return float(wall() if callable(wall) else self._clock())
        return time.time()

    def bucket_of(self, p: Passage) -> str:
        """段落的长度桶：优先使用 ``length_bucket``，否则按 token_len 计算（区间外取最近桶）。"""
        if p.length_bucket in self.buckets:
            return p.length_bucket  # type: ignore[return-value]
        return self._buckets_by_distance(p.token_len)[0]

    def _buckets_by_distance(self, near: float) -> list[str]:
        """按与 near 的距离由近到远排序的桶名（包含 near 的桶距离为 0）。"""
        return sorted(
            self._bucket_order,
            key=lambda b: (_interval_distance(*self.buckets[b], near), self._bucket_order.index(b)),
        )

    def _weight(self, p: Passage) -> float:
        return 1.0 / (1.0 + p.use_count + self._run_uses.get(p.id, 0)) ** self.alpha

    # ------------------------------------------------------------------ 选取

    def pick(
        self,
        lang: str,
        near: int,
        exclude: set[int] | frozenset[int] = frozenset(),
        rng: Random | None = None,
        now: float | None = None,
    ) -> Passage | None:
        """在 lang 下按与 near 的距离逐桶加权抽样；全部无候选返回 None。不修改 run_uses。"""
        rng = rng if rng is not None else Random()
        t = self._now() if now is None else float(now)
        cap = self.per_run_max_uses
        run_uses = self._run_uses

        for b in self._buckets_by_distance(near):
            pids = self._lists.get((lang, b))
            if not pids:
                continue
            cands: list[Passage] = []
            weights: list[float] = []
            total = 0.0
            for pid in pids:
                if pid in exclude or run_uses.get(pid, 0) >= cap:
                    continue
                p = self._passages[pid]
                if p.expires_at is not None and p.expires_at <= t:
                    continue
                w = self._weight(p)
                cands.append(p)
                weights.append(w)
                total += w
            if not cands:
                continue
            r = rng.random() * total
            acc = 0.0
            for p, w in zip(cands, weights):
                acc += w
                if r < acc:
                    return p
            return cands[-1]  # 浮点累加误差兜底
        return None

    def mark_used(self, pid: int) -> None:
        """run_uses[pid] += 1（Req 9.5）。"""
        self._run_uses[pid] = self._run_uses.get(pid, 0) + 1

    # ------------------------------------------------------------------ 增删

    def ingest(self, p: Passage, now: float | None = None) -> bool:
        """加入一个段落。已存在、已移除（隔离/过期）或已过期时忽略并返回 False。"""
        if p.id in self._passages or p.id in self._removed:
            return False
        if p.expires_at is not None:
            t = self._now() if now is None else float(now)
            if p.expires_at <= t:
                return False
        key = (p.lang, self.bucket_of(p))
        lst = self._lists.setdefault(key, [])
        self._pos[p.id] = len(lst)
        lst.append(p.id)
        self._passages[p.id] = p
        self._key_of[p.id] = key
        return True

    def ingest_pending(self, q: asyncio.Queue, max_items: int = 64) -> int:
        """非阻塞读取最多 max_items 个新段落（只用 get_nowait）；返回实际加入索引的数量。"""
        added = 0
        for _ in range(max_items):
            try:
                p = q.get_nowait()
            except asyncio.QueueEmpty:
                break
            if self.ingest(p):
                added += 1
        return added

    def remove(self, pid: int) -> bool:
        """移除段落（隔离或过期）；之后同一索引不会再接收该 id。run_uses 保留以便运行结束累加。"""
        self._removed.add(pid)
        if pid not in self._passages:
            return False
        key = self._key_of.pop(pid)
        lst = self._lists[key]
        i = self._pos.pop(pid)
        last = lst.pop()
        if last != pid:
            lst[i] = last
            self._pos[last] = i
        del self._passages[pid]
        return True

    def expire_due(self, now: float | None = None) -> int:
        """移除全部 expires_at <= now 的段落，返回移除数量。"""
        t = self._now() if now is None else float(now)
        due = [pid for pid, p in self._passages.items() if p.expires_at is not None and p.expires_at <= t]
        for pid in due:
            self.remove(pid)
        return len(due)

    # ------------------------------------------------------------------ 查询

    def inventory(self) -> dict[BucketKey, int]:
        return {k: len(v) for k, v in self._lists.items() if v}

    @property
    def run_uses(self) -> dict[int, int]:
        """本次运行各段落的使用次数（副本，供 flush_use_counts）。"""
        return dict(self._run_uses)

    @property
    def size(self) -> int:
        return len(self._passages)

    def __len__(self) -> int:
        return len(self._passages)

    def contains(self, pid: int) -> bool:
        return pid in self._passages

    def __contains__(self, pid: object) -> bool:
        return pid in self._passages

    def get(self, pid: int) -> Passage | None:
        return self._passages.get(pid)


__all__ = ["PassageIndex", "DEFAULT_BUCKETS"]
