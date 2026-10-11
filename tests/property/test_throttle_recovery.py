"""Property 11: 探测成功后 60 秒阶梯回满（ThrottleController, mode=probe）。

∀ 事件序列：
    - 从 HOLD / PROBE 进入 RECOVER 只能经由 PROBE 中一次 OK 的探测；
    - level 升高只发生在 RECOVER 的 tick 上，每次只升相邻一级，
      且距上一次 节流 / 主动减速 / 升档 / 探测成功 ≥ recover_step_s（15s，容差 1e-6）；
    - 从探测成功到 level == 1.0 至少 recover_full_s（60s），tick 间隔 1s 且无新事件时 ≤ 61s；
    - RECOVER 中出现新的（非 stale）Throttle_Event → HOLD。

Validates: Requirements 20.1, 20.4, 20.5, 20.7
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.outcome import Outcome, RateLimitHeaders
from loadtest.settings import ThrottleSettings
from loadtest.throttle import DispatchTicket, ThrottleController, ThrottleState

S = ThrottleState
CFG = ThrottleSettings()
LEVELS = list(CFG.levels)
STEP_S = CFG.recover_step_s
FLOOR_IDX = LEVELS.index(CFG.soft_down_floor)
TOL = 1e-6

LOW = RateLimitHeaders(remaining_tokens=1, limit_tokens=100)
HIGH = RateLimitHeaders(remaining_tokens=90, limit_tokens=100, remaining_requests=50, limit_requests=100)


@dataclass
class Res:
    outcome: Outcome
    retry_after_s: float | None = None
    headers: RateLimitHeaders | None = None


RA = st.one_of(st.none(), st.floats(min_value=0.0, max_value=150.0, allow_nan=False))
OFF = st.floats(min_value=0.01, max_value=0.99, allow_nan=False)
# OK 加权，使探测成功 → RECOVER 的路径被充分覆盖
PROBE_OUTCOME = st.sampled_from([Outcome.OK, Outcome.OK, Outcome.OK, *Outcome])
EVENT = st.one_of(
    st.tuples(st.just("idle"), st.integers(1, 20)),
    st.tuples(st.just("idle"), st.integers(55, 70)),  # 足以完成一次 60s 阶梯
    # 宏：(必要时)新节流 → 等到可探测 → 派发探测 → 以给定结果返回
    st.tuples(st.just("to_probe"), RA, PROBE_OUTCOME, OFF),
    st.tuples(st.just("throttle"), OFF, RA),  # 新的 Throttle_Event（非 stale）
    st.tuples(st.just("stale"), OFF, RA),  # HOLD 前派发的在途请求返回 429
    st.tuples(st.just("dispatch"), OFF, st.integers(1, 3)),
    st.tuples(st.just("complete"), OFF, st.sampled_from(["ok_low", "ok_high", "ok_none", "throttled"])),
    st.tuples(st.just("probe"), OFF),
    st.tuples(st.just("probe_result"), OFF, PROBE_OUTCOME, RA),
)


class Sim:
    def __init__(self, start: float) -> None:
        self.clock = FakeClock(start)
        self.th = ThrottleController(CFG, self.clock)
        self.tickets: list[DispatchTicket] = []
        self.probe: DispatchTicket | None = None
        # 独立模型
        self.hold_start = -math.inf
        self.last_evt = -math.inf
        self.last_soft = -math.inf
        self.episode: float | None = None  # 最近一次 OK 探测进入 RECOVER 的时刻
        self.clean = False  # 该 episode 内是否没有任何其他事件
        self.prev_hold_until = self.th.hold_until

    # ---- 检查 ----
    def check(self, s0: ThrottleState, i0: int, op: str, tags: set[str]) -> None:
        th, now = self.th, self.clock()
        s1, i1 = th.state, LEVELS.index(th.level)

        # P-T1：hold_until 只增不减；HOLD ⇒ (0, 0)
        assert th.hold_until >= self.prev_hold_until
        self.prev_hold_until = th.hold_until
        if s1 is S.HOLD:
            assert th.allowed_dispatch(now) == (0, 0.0)

        # Req 20.1：HOLD / PROBE → RECOVER 只经 PROBE 中的 OK 探测
        if s1 is S.RECOVER and s0 in (S.HOLD, S.PROBE):
            assert s0 is S.PROBE and "probe_ok" in tags, (op, tags)
            assert i1 == 0
        # 状态转移只能来自合法路径
        if s1 is not s0:
            if op == "tick":
                assert (s0, s1) in {(S.HOLD, S.PROBE), (S.RECOVER, S.NORMAL)}
                if s0 is S.HOLD:
                    assert now >= th.hold_until
            elif s1 is S.HOLD:
                assert "fresh" in tags
            elif s1 is S.RECOVER:
                assert "probe_ok" in tags or (s0 is S.NORMAL and "soft" in tags)
            else:
                raise AssertionError((s0, s1, op, tags))

        # Req 20.7：非 stale 节流 → HOLD（含 RECOVER 中）
        if "fresh" in tags:
            assert s1 is S.HOLD
        # Req 19.6：PROBE / RECOVER 中 stale 被忽略
        if "stale" in tags and s0 in (S.PROBE, S.RECOVER):
            assert s1 is s0 and i1 == i0

        # Req 20.4：升档只在 RECOVER 的 tick 上，相邻一级，间隔 ≥ recover_step_s
        if i1 > i0:
            assert op == "tick" and s0 is S.RECOVER, (op, s0)
            assert i1 == i0 + 1
            assert now - self.last_evt >= STEP_S - TOL, (now, self.last_evt)
            self.last_evt = now
        elif i1 < i0:
            if s1 is S.HOLD:
                assert i1 == 0
            else:
                assert "soft" in tags and i1 == i0 - 1 and i1 >= FLOOR_IDX

        # 更新模型
        if tags & {"fresh", "probe_ok", "soft"}:
            self.last_evt = now
        if "soft" in tags:
            self.last_soft = now
        if s1 is S.HOLD and s0 is not S.HOLD:
            self.hold_start = now

        # Req 20.5：探测成功 → 100% 的时长
        if "probe_ok" in tags:
            self.episode, self.clean = now, True
        elif op != "tick":
            self.clean = False
        if s1 is S.HOLD:
            self.episode = None
        elif s1 is S.NORMAL and self.episode is not None:
            dur = now - self.episode
            assert dur >= CFG.recover_full_s - TOL, dur
            if self.clean:
                assert dur <= CFG.recover_full_s + 1.0 + TOL, dur
            self.episode = None

    # ---- 操作 ----
    def tick_to(self, t: float) -> None:
        s0, i0 = self.th.state, LEVELS.index(self.th.level)
        self.clock.set(t)
        self.th.tick(t)
        self.check(s0, i0, "tick", set())

    def soft_tag(self, h: RateLimitHeaders | None, s0: ThrottleState) -> set[str]:
        low = h is not None and h.limit_tokens and h.remaining_tokens is not None and (
            h.remaining_tokens / h.limit_tokens < CFG.soft_down_ratio
        )
        if low and s0 in (S.NORMAL, S.RECOVER) and self.clock() - self.last_soft >= CFG.soft_down_cooldown_s:
            return {"soft"}
        return set()

    def event(self, ev: tuple) -> None:
        th = self.th
        kind = ev[0]
        base = self.clock()
        now = base + ev[1]
        self.clock.set(now)
        s0, i0 = th.state, LEVELS.index(th.level)
        tags: set[str] = set()
        if kind == "throttle":
            # 非 stale 的 429 只可能来自 NORMAL / RECOVER 中派发的请求（HOLD / PROBE 不派发普通请求）
            if s0 in (S.NORMAL, S.RECOVER):
                th.on_throttle(ev[2], now)
                tags.add("fresh")
        elif kind == "stale":
            if math.isfinite(self.hold_start):
                th.on_throttle(ev[2], now, dispatched_at=self.hold_start - 1.0)
                tags.add("stale")
        elif kind == "dispatch":
            if th.allowed_dispatch(now)[0] is None:
                for _ in range(ev[2]):
                    self.tickets.append(th.note_dispatch(now, is_probe=False))
        elif kind == "complete":
            if self.tickets:
                t = self.tickets.pop(0)
                what = ev[2]
                if what == "throttled":
                    tags.add("stale" if t.dispatched_at < self.hold_start else "fresh")
                    th.on_result(t, Res(Outcome.THROTTLED, 6.0), now)
                else:
                    h = {"ok_low": LOW, "ok_high": HIGH, "ok_none": None}[what]
                    tags |= self.soft_tag(h, s0)
                    th.on_result(t, Res(Outcome.OK, headers=h), now)
                th.note_done(t)
        elif kind == "probe":
            if self.probe is None and th.allowed_dispatch(now)[0] == 1:
                self.probe = th.note_dispatch(now, is_probe=True)
        elif kind == "probe_result":
            if self.probe is not None:
                outcome, ra = ev[2], ev[3]
                if outcome is Outcome.OK and s0 is S.PROBE:
                    tags.add("probe_ok")
                elif outcome is Outcome.THROTTLED:
                    tags.add("fresh")
                th.on_result(self.probe, Res(outcome, ra), now)
                th.note_done(self.probe)
                self.probe = None
        self.check(s0, i0, kind, tags)
        self.tick_to(base + 1.0)

    def to_probe(self, ra: float | None) -> None:
        th = self.th
        if th.state in (S.NORMAL, S.RECOVER):
            self.event(("throttle", 0.5, ra))
        for _ in range(200):
            if th.state is S.PROBE and th.allowed_dispatch(self.clock())[0] == 1:
                break
            self.tick_to(self.clock() + 1.0)
        self.event(("probe", 0.1))

    def run(self, events: list[tuple]) -> None:
        for ev in events:
            if ev[0] == "idle":
                for _ in range(ev[1]):
                    self.tick_to(self.clock() + 1.0)
            elif ev[0] == "to_probe":
                self.to_probe(ev[1])
                self.event(("probe_result", ev[3], ev[2], ev[1]))
            else:
                self.event(ev)


@settings(max_examples=200, deadline=None)
@given(start=st.floats(0.0, 1e5, allow_nan=False), events=st.lists(EVENT, min_size=10, max_size=80))
def test_recovery_invariants(start: float, events: list[tuple]) -> None:
    Sim(start).run(events)


@settings(max_examples=200, deadline=None)
@given(
    start=st.floats(0.0, 1e5, allow_nan=False),
    prefix=st.lists(EVENT, max_size=30),
    ra=RA,
    ok_off=OFF,
)
def test_probe_ok_reaches_full_in_60_to_61s(start: float, prefix: list[tuple], ra: float | None, ok_off: float) -> None:
    """任意前缀后：新节流 → HOLD → PROBE → OK，此后仅 1s tick，60s ≤ 回满时长 ≤ 61s。"""
    sim = Sim(start)
    sim.run(prefix)
    th, clock = sim.th, sim.clock
    if sim.probe is not None:  # 清掉前缀遗留的探测
        sim.event(("probe_result", 0.5, Outcome.SERVER_ERROR, None))
    while sim.tickets:
        sim.event(("complete", 0.5, "ok_none"))
    if th.state in (S.NORMAL, S.RECOVER):
        sim.event(("throttle", 0.5, ra))
        assert th.state is S.HOLD
    sim.to_probe(ra)
    assert sim.probe is not None and th.state is S.PROBE
    base = clock()
    clock.set(base + ok_off)
    t_ok = clock()
    s0 = th.state
    th.on_result(sim.probe, Res(Outcome.OK), t_ok)
    th.note_done(sim.probe)
    sim.probe = None
    sim.check(s0, 0, "probe_result", {"probe_ok"})
    assert th.state is S.RECOVER and th.level == LEVELS[0]
    t = base
    while th.state is not S.NORMAL:
        t += 1.0
        sim.tick_to(t)
        assert t - t_ok <= 62.0
    assert CFG.recover_full_s - TOL <= clock() - t_ok <= CFG.recover_full_s + 1.0 + TOL
