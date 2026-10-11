"""Property 10（控制器层）: HOLD 零派发、PROBE 单请求。

∀ 429 / 在途 stale 返回 / 探测结果的任意交错（FakeClock，tick 间隔 ≤ 1s）：
    - HOLD ⇒ allowed_dispatch == (0, 0)；
    - PROBE ⇒ 至多 1 个探测在途，探测在途时 max_new == 0；
    - 探测只在在途请求清空或进入 PROBE 满 probe_inflight_wait_s（30s）后发出；
    - 探测 THROTTLED → 新的 HOLD，hold_until 到达后再探测（次数不限）；
    - 探测 SERVER_ERROR / TIMEOUT / CLIENT_ERROR → 下一个探测不早于 probe_error_retry_s（5s）之后；
    - 探测 CONTENT_FILTER → 不进入 RECOVER，可立即再探。

Validates: Requirements 19.4, 19.7, 20.2, 20.3
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.clock import FakeClock
from loadtest.outcome import Outcome
from loadtest.settings import ThrottleSettings
from loadtest.throttle import DispatchTicket, ThrottleController, ThrottleState

S = ThrottleState
CFG = ThrottleSettings()
L0 = CFG.levels[0]
TOL = 1e-9
ERRORS = (Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR)


@dataclass
class Res:
    outcome: Outcome
    retry_after_s: float | None = None
    headers: None = None


RA = st.one_of(st.none(), st.floats(min_value=0.0, max_value=150.0, allow_nan=False))
DT = st.floats(min_value=0.01, max_value=1.0, allow_nan=False)
EVENT = st.one_of(
    st.tuples(st.just("idle"), st.integers(1, 35)),
    # 宏：(必要时)新节流 → tick 到可探测 → 派发探测 → 以给定结果返回
    st.tuples(st.just("to_probe"), RA, st.sampled_from(list(Outcome)), DT),
    st.tuples(st.just("throttle"), DT, RA),  # NORMAL / RECOVER 中派发的请求返回非 stale 429
    st.tuples(st.just("schedule"), DT, st.integers(1, 4)),
    st.tuples(
        st.just("complete"),
        DT,
        st.integers(0, 50),
        st.sampled_from([Outcome.OK, Outcome.THROTTLED, Outcome.SERVER_ERROR, Outcome.TIMEOUT]),
        RA,
    ),
    st.tuples(st.just("probe_result"), DT, st.sampled_from(list(Outcome)), RA),
)


class Sim:
    def __init__(self, start: float) -> None:
        self.clock = FakeClock(start)
        self.th = ThrottleController(CFG, self.clock)
        self.normal: list[DispatchTicket] = []
        self.probes: list[DispatchTicket] = []
        self.probe_phase_start = -math.inf
        self.err_at = -math.inf  # 最近一次探测错误的时刻
        self.dispatched_in_hold = 0

    def check(self) -> None:
        th, now = self.th, self.clock()
        a = th.allowed_dispatch(now)
        if th.state is S.HOLD:
            assert a == (0, 0.0)
        if th.state is S.PROBE:
            assert len(self.probes) <= 1
            if self.probes:
                assert a == (0, 0.0)
            if a[0] == 1:
                assert a[1] == L0
                assert not self.normal or now - self.probe_phase_start >= CFG.probe_inflight_wait_s - TOL
                assert now >= self.err_at + CFG.probe_error_retry_s - TOL
            else:
                assert a[0] == 0
        else:
            assert a[0] in (0, None)

    def advance(self, dt: float) -> None:
        # 调度器 tick 间隔 ≤ 1s
        s0 = self.th.state
        self.clock.advance(dt)
        self.th.tick(self.clock())
        if self.th.state is S.PROBE and s0 is not S.PROBE:
            self.probe_phase_start = self.clock()
        self.check()

    def schedule(self, k: int) -> None:
        th, now = self.th, self.clock()
        n, _lvl = th.allowed_dispatch(now)
        if th.state is S.HOLD:
            assert n == 0
        if n is None:
            for _ in range(k):
                self.normal.append(th.note_dispatch(now, is_probe=False))
        elif n == 1:
            assert th.state is S.PROBE and not self.probes
            self.probes.append(th.note_dispatch(now, is_probe=True))
            assert th.allowed_dispatch(now) == (0, 0.0)  # 探测在途：不再派发
        self.check()

    def probe_result(self, outcome: Outcome, ra: float | None) -> None:
        if not self.probes:
            return
        th, now = self.th, self.clock()
        s0, hu0 = th.state, th.hold_until
        p = self.probes.pop()
        th.on_result(p, Res(outcome, ra), now)
        th.note_done(p)
        if outcome is Outcome.OK:
            assert th.state is (S.RECOVER if s0 is S.PROBE else s0)
        elif outcome is Outcome.THROTTLED:
            assert th.state is S.HOLD
            assert th.hold_until >= max(hu0, now + th._cooldown(ra)) - TOL
            assert th.hold_until > now
        else:
            assert th.state is s0  # 保持 PROBE（或原状态），不进入 RECOVER
            if outcome in ERRORS:
                self.err_at = now
                if s0 is S.PROBE:
                    assert th.allowed_dispatch(now) == (0, 0.0)
            elif s0 is S.PROBE and (not self.normal or now - self.probe_phase_start >= CFG.probe_inflight_wait_s):
                # CONTENT_FILTER：立即可再探
                assert th.allowed_dispatch(now) == (1, L0)
        self.check()

    def complete(self, idx: int, outcome: Outcome, ra: float | None) -> None:
        if not self.normal:
            return
        th, now = self.th, self.clock()
        s0, hu0 = th.state, th.hold_until
        t = self.normal.pop(idx % len(self.normal))
        stale = t.dispatched_at < th.hold_started_at
        th.on_result(t, Res(outcome, ra), now)
        th.note_done(t)
        if outcome is Outcome.THROTTLED:
            if stale:
                assert th.state is s0
                if s0 is not S.HOLD:
                    assert th.hold_until == hu0
            else:
                assert th.state is S.HOLD
        self.check()

    def throttle(self, ra: float | None) -> None:
        th = self.th
        if th.state in (S.NORMAL, S.RECOVER):
            hu0 = th.hold_until
            th.on_throttle(ra, self.clock())
            assert th.state is S.HOLD and th.hold_until >= hu0
        self.check()

    def to_probe(self, ra: float | None) -> None:
        self.advance(0.5)
        self.throttle(ra)
        for _ in range(200):
            if self.th.state is S.PROBE and self.th.allowed_dispatch(self.clock())[0] == 1:
                break
            self.advance(1.0)
        self.schedule(1)

    def run(self, events: list[tuple]) -> None:
        for ev in events:
            kind = ev[0]
            if kind == "idle":
                for _ in range(ev[1]):
                    self.advance(1.0)
                continue
            if kind == "to_probe":
                self.to_probe(ev[1])
                self.advance(ev[3])
                self.probe_result(ev[2], ev[1])
                continue
            self.advance(ev[1])
            if kind == "throttle":
                self.throttle(ev[2])
            elif kind == "schedule":
                self.schedule(ev[2])
            elif kind == "complete":
                self.complete(ev[2], ev[3], ev[4])
            elif kind == "probe_result":
                self.probe_result(ev[2], ev[3])


@settings(max_examples=200, deadline=None)
@given(start=st.floats(0.0, 1e5, allow_nan=False), events=st.lists(EVENT, min_size=10, max_size=100))
def test_hold_zero_probe_single(start: float, events: list[tuple]) -> None:
    Sim(start).run(events)


@settings(max_examples=100, deadline=None)
@given(
    in_flight=st.integers(0, 5),
    first_ra=RA,
    fails=st.lists(st.tuples(RA, st.floats(0.0, 3.0, allow_nan=False)), min_size=1, max_size=15),
    others=st.lists(st.sampled_from([Outcome.CONTENT_FILTER, *ERRORS]), max_size=6),
)
def test_unlimited_reprobe(
    in_flight: int, first_ra: float | None, fails: list[tuple[float | None, float]], others: list[Outcome]
) -> None:
    """探测 THROTTLED → 新 HOLD → 再次 PROBE，次数不限；错误 / 内容过滤保持 PROBE。"""
    sim = Sim(0.0)
    th, clock = sim.th, sim.clock
    sim.schedule(in_flight)  # NORMAL：派发若干在途请求（后续一律 stale）
    sim.advance(0.5)
    th.on_throttle(first_ra, clock())
    sim.check()

    def wait_probe() -> None:
        for _ in range(400):
            if th.state is S.PROBE and th.allowed_dispatch(clock())[0] == 1:
                return
            sim.advance(1.0)
        raise AssertionError("probe never allowed")

    wait_probe()
    # 在途请求未清空时必须等满 30s
    if in_flight:
        assert clock() - sim.probe_phase_start >= CFG.probe_inflight_wait_s - TOL
    for k, (ra, delay) in enumerate(fails, start=1):
        sim.schedule(1)
        assert len(sim.probes) == 1
        if delay:
            sim.advance(delay)
        sim.probe_result(Outcome.THROTTLED, ra)
        assert th.state is S.HOLD
        assert th.stats.consecutive_probe_fail == k
        wait_probe()
    for o in others:
        sim.schedule(1)
        t_res = clock() + 0.5
        sim.advance(0.5)
        sim.probe_result(o, None)
        assert th.state is S.PROBE
        if o in ERRORS:
            wait_probe()
            assert clock() >= t_res + CFG.probe_error_retry_s - TOL
        else:
            assert th.allowed_dispatch(clock())[0] == 1
    assert th.stats.probes_sent == len(fails) + len(others)
    assert th.stats.max_consecutive_probe_fail == len(fails)
    assert th.stats.probe_ok == 0 and S.RECOVER not in [b for _, _, b in th.stats.timeline]
