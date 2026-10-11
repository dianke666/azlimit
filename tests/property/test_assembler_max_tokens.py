"""Property 5: 输出上限。

∀ 组装结果，max_tokens == 40，且 user 消息中的指令带有对应语言的固定后缀，位于正文之前或之后。

**Validates: Requirements 14.5**
"""
from __future__ import annotations

from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.load.assembler import SEP
from loadtest.load.tasks import REQUIRED_SUFFIX
from tests.property.test_assembler_precision import catalog, make_assembler


@settings(max_examples=60, deadline=None)
@given(target=st.integers(min_value=1500, max_value=6000), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_max_tokens_and_suffix(target: int, seed: int) -> None:
    req = make_assembler().assemble(target, Random(seed))
    assert req.max_tokens == 40
    task = next(t for t in catalog().for_lang(req.lang) if t.id == req.task_id)
    assert task.instruction.endswith(REQUIRED_SUFFIX[req.lang])
    system, user = req.messages
    assert system == {"role": "system", "content": task.system_prompt}
    assert user["role"] == "user"
    if req.instruction_position == "before":
        assert user["content"].startswith(task.instruction + SEP)
    else:
        assert req.instruction_position == "after"
        assert user["content"].endswith(SEP + task.instruction)


def test_both_instruction_positions_occur() -> None:
    asm = make_assembler()
    rng = Random(3)
    positions = {asm.assemble(2000, rng).instruction_position for _ in range(30)}
    assert positions == {"before", "after"}
