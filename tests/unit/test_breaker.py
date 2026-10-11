"""CircuitBreaker 单元测试（Requirements 23.1, 23.2, 23.4, 23.6）。"""

from __future__ import annotations

import pytest

from loadtest.load.breaker import CircuitBreaker
from loadtest.outcome import Outcome

FAIL = Outcome.SERVER_ERROR


def _open(cb: CircuitBreaker, now: float = 0.0) -> None:
    for _ in range(10):
        cb.record(FAIL, now)
    assert cb.state(now) == "open"


@pytest.mark.parametrize("ignored", [Outcome.THROTTLED, Outcome.CONTENT_FILTER])
def test_ignored_outcomes_not_samples_and_do_not_reset(ignored: Outcome) -> None:
    cb = CircuitBreaker()
    for _ in range(9):
        cb.record(FAIL, 0.0)
    for _ in range(50):
        cb.record(ignored, 0.0)
    assert cb.consecutive == 9 and cb.state(0.0) == "closed"
    assert len(cb._window) == 9  # 不计入样本
    cb.record(FAIL, 0.0)
    assert cb.state(0.0) == "open"


def test_ignored_outcomes_do_not_dilute_error_rate() -> None:
    cb = CircuitBreaker()
    for i in range(19):
        cb.record(Outcome.OK if i % 2 else FAIL, 0.0)  # 10 失败 / 9 成功，未达 20 样本
        cb.record(Outcome.THROTTLED, 0.0)
    assert cb.state(0.0) == "closed"
    cb.record(Outcome.TIMEOUT, 0.0)  # 第 20 个样本：11/20 > 50%
    assert cb.state(0.0) == "open"


def test_ok_resets_consecutive() -> None:
    cb = CircuitBreaker()
    for _ in range(9):
        cb.record(Outcome.CLIENT_ERROR, 0.0)
    cb.record(Outcome.OK, 0.0)
    assert cb.consecutive == 0
    for _ in range(9):
        cb.record(Outcome.TIMEOUT, 1.0)
    # 19 个样本 18 失败，但样本 < 20 且连续 9 次 → 仍 closed
    assert cb.state(1.0) == "closed"


def test_error_rate_over_half_with_20_samples_opens() -> None:
    cb = CircuitBreaker()
    seq = [FAIL, Outcome.OK] * 9 + [FAIL]  # 19 个样本：10 失败
    for o in seq:
        cb.record(o, 0.0)
    assert cb.state(0.0) == "closed"
    cb.record(FAIL, 0.0)  # 20 样本 11 失败 = 55%
    assert cb.state(0.0) == "open" and cb.open_count == 1


def test_exactly_half_does_not_open() -> None:
    cb = CircuitBreaker()
    for o in [FAIL, Outcome.OK] * 10:
        cb.record(o, 0.0)
    assert cb.state(0.0) == "closed"


def test_window_prunes_old_samples() -> None:
    cb = CircuitBreaker()
    for _ in range(9):
        cb.record(FAIL, 0.0)
    cb.record(Outcome.OK, 0.0)
    for _ in range(10):
        cb.record(Outcome.OK, 30.0)
    # 61s 后旧样本失效：新增 9 失败 + 窗口内 10 OK = 19 样本
    for _ in range(9):
        cb.record(FAIL, 61.0)
    assert cb.state(61.0) == "closed"


def test_ten_consecutive_failures_open() -> None:
    cb = CircuitBreaker()
    for o in [Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR] * 3:
        cb.record(o, 0.0)
    assert cb.state(0.0) == "closed"
    cb.record(False, 0.0)  # bool 也可
    assert cb.state(0.0) == "open"
    assert cb.retry_at == 60.0
    assert not cb.allow_dispatch(0.0) and cb.half_open_remaining(0.0) == 0


def test_open_to_half_open_after_60s_then_close_after_5_ok() -> None:
    cb = CircuitBreaker()
    _open(cb, 0.0)
    assert cb.state(59.9) == "open"
    cb.record(Outcome.OK, 30.0)  # open 期间到达的在途结果被忽略
    assert cb.state(59.9) == "open"
    assert cb.state(60.0) == "half_open"
    assert cb.retry_at == 0.0
    for i in range(5):
        assert cb.allow_dispatch(60.0)
        cb.on_dispatch(60.0)
    for i in range(4):
        cb.record(Outcome.OK, 61.0)
        assert cb.state(61.0) == "half_open"
    cb.record(True, 61.0)
    assert cb.state(61.0) == "closed"
    assert cb.consecutive == 0 and cb.half_open_remaining(61.0) is None
    assert cb.open_count == 1


@pytest.mark.parametrize("fail", [Outcome.SERVER_ERROR, Outcome.TIMEOUT, Outcome.CLIENT_ERROR])
def test_failure_in_half_open_reopens(fail: Outcome) -> None:
    cb = CircuitBreaker()
    _open(cb, 0.0)
    assert cb.state(60.0) == "half_open"
    cb.on_dispatch(60.0)
    cb.on_dispatch(60.0)
    cb.record(Outcome.OK, 60.5)
    cb.record(fail, 61.0)
    assert cb.state(61.0) == "open" and cb.open_count == 2
    assert cb.retry_at == 121.0
    # 下一轮 half_open 名额重置
    assert cb.state(121.0) == "half_open" and cb.half_open_remaining(121.0) == 5


def test_half_open_ignores_throttled_and_content_filter() -> None:
    cb = CircuitBreaker()
    _open(cb, 0.0)
    cb.record(Outcome.THROTTLED, 60.0)
    cb.record(Outcome.CONTENT_FILTER, 60.0)
    assert cb.state(60.0) == "half_open"


def test_half_open_probe_cap_and_release() -> None:
    cb = CircuitBreaker()
    assert cb.half_open_remaining(0.0) is None and cb.allow_dispatch(0.0)
    cb.on_dispatch(0.0)  # closed 下不占名额
    _open(cb, 0.0)
    assert cb.half_open_remaining(60.0) == 5
    for k in range(5):
        cb.on_dispatch(60.0)
        assert cb.half_open_remaining(60.0) == 4 - k
    assert not cb.allow_dispatch(60.0)
    cb.release_probe()
    assert cb.allow_dispatch(60.0) and cb.half_open_remaining(60.0) == 1
    cb.on_dispatch(60.0)
    assert not cb.allow_dispatch(60.0)
    for _ in range(10):
        cb.release_probe()
    assert cb.half_open_remaining(60.0) == 5  # 不会超过上限


def test_open_count_increments_each_open() -> None:
    cb = CircuitBreaker()
    t = 0.0
    for n in range(1, 4):
        _open(cb, t) if n == 1 else cb.record(FAIL, t)
        assert cb.open_count == n and cb.state(t) == "open"
        t += 60.0
        assert cb.state(t) == "half_open"
    # 从 closed 重新累积也会递增
    for _ in range(5):
        cb.record(Outcome.OK, t)
    assert cb.state(t) == "closed"
    _open(cb, t)
    assert cb.open_count == 4
