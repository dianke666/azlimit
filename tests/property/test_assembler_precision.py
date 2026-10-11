"""Property 3: 组装精度。

∀ target ∈ [1500, 6000]，|input_tokens − target| ≤ 20，且 input_tokens 与完整消息复算一致。

**Validates: Requirements 14.1, 14.8**

本模块同时提供合成语料池辅助函数（``synthetic_passages`` / ``make_index`` / ``catalog``），
供其他组装器测试复用。
"""
from __future__ import annotations

from functools import lru_cache
from random import Random

from hypothesis import given, settings
from hypothesis import strategies as st

from loadtest.corpus.index import PassageIndex
from loadtest.corpus.store import Passage
from loadtest.load.assembler import RequestAssembler
from loadtest.load.tasks import TaskCatalog
from loadtest.tokens import default_counter

# ---------------------------------------------------------------- 合成语料

_ZH_SUBJ = ["城市交通系统", "这家制造企业", "现代农业", "医疗服务机构", "在线教育平台", "物流行业", "新能源汽车", "地方政府",
            "研究团队", "零售商", "金融科技公司", "气象部门", "社区图书馆", "软件工程师", "餐饮连锁店", "港口运营方"]
_ZH_VERB = ["正在逐步优化", "持续投入资源改进", "重新评估了", "开始系统性地梳理", "引入新的方法管理", "认真分析了",
            "调整了长期规划中的", "通过试点项目验证了", "与合作伙伴共同完善了"]
_ZH_OBJ = ["数据采集流程", "成本控制策略", "用户反馈机制", "安全生产规范", "人才培养方案", "供应链协同模式", "能耗监测体系",
           "服务质量标准", "风险预警指标", "设备维护计划", "信息公开制度", "库存周转效率"]
_ZH_TAIL = ["，效果在半年内逐渐显现", "，并定期公布阶段性结果", "，以应对不断变化的市场需求", "，相关经验值得其他地区参考",
            "，但仍需要更多时间观察", "，同时兼顾了效率与公平", ""]
_EN_SUBJ = ["The city transit system", "This manufacturing firm", "Modern agriculture", "The regional hospital",
            "An online learning platform", "The logistics sector", "The research team", "A local retailer",
            "The weather service", "The public library", "A fintech startup", "The port operator"]
_EN_VERB = ["is gradually improving", "has invested heavily in", "recently reviewed", "began a systematic audit of",
            "introduced new tools for", "carefully analyzed", "redesigned", "piloted changes to"]
_EN_OBJ = ["its data collection workflow", "its cost control strategy", "the customer feedback loop",
           "workplace safety rules", "staff training programs", "supply chain coordination", "energy monitoring",
           "service quality standards", "risk indicators", "equipment maintenance schedules", "inventory turnover"]
_EN_TAIL = [", and early results look promising", ", publishing progress reports every quarter",
            " to keep pace with changing demand", ", although more evidence is still needed",
            ", balancing efficiency with fairness", ""]


def _sentence(rng: Random, lang: str) -> str:
    if lang == "zh":
        return rng.choice(_ZH_SUBJ) + rng.choice(_ZH_VERB) + rng.choice(_ZH_OBJ) + rng.choice(_ZH_TAIL) + "。"
    return f"{rng.choice(_EN_SUBJ)} {rng.choice(_EN_VERB)} {rng.choice(_EN_OBJ)}{rng.choice(_EN_TAIL)}. "


def _bucket(n: int) -> str:
    return "S" if n < 800 else ("M" if n < 1200 else "L")


def _make_text(rng: Random, lang: str, want: int) -> str:
    c = default_counter()
    sents: list[str] = []
    text = ""
    while c.count(text) < want + 20:
        sents.extend(_sentence(rng, lang) for _ in range(8))
        text = ("".join(sents)).strip()
    return c.trim_to_tokens(text, want)


@lru_cache(maxsize=8)
def synthetic_passages(n_per_lang: int = 200, seed: int = 7, start_id: int = 1) -> tuple[Passage, ...]:
    """每种语言 n_per_lang 个段落，均分到 S [500,800) / M [800,1200) / L [1350,1500]，token_len 精确。"""
    rng = Random(seed)
    c = default_counter()
    out: list[Passage] = []
    pid = start_id
    ranges = [(500, 799), (800, 1199), (1350, 1500)]
    for lang in ("zh", "en"):
        for i in range(n_per_lang):
            lo, hi = ranges[i % 3]
            text = _make_text(rng, lang, rng.randint(lo, hi))
            n = c.count(text)
            out.append(Passage(id=pid, category="test", lang=lang, token_len=n, use_count=0,
                               expires_at=None, text=text, length_bucket=_bucket(n)))
            pid += 1
    return tuple(out)


def make_index(passages=None, per_run_max_uses: int = 4) -> PassageIndex:
    return PassageIndex(synthetic_passages() if passages is None else passages, per_run_max_uses=per_run_max_uses)


@lru_cache(maxsize=1)
def catalog() -> TaskCatalog:
    return TaskCatalog.load()


def make_assembler(index: PassageIndex | None = None, **kw) -> RequestAssembler:
    return RequestAssembler(index if index is not None else make_index(), catalog(), default_counter(), **kw)


# ---------------------------------------------------------------- Property 3


@settings(max_examples=80, deadline=None)
@given(target=st.integers(min_value=1500, max_value=6000), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_assembly_precision(target: int, seed: int) -> None:
    asm = make_assembler()
    req = asm.assemble(target, Random(seed))
    assert abs(req.input_tokens - target) <= 20
    assert req.input_tokens == default_counter().count_messages_full(req.messages)
    assert req.target_tokens == target


def test_assembly_precision_extremes() -> None:
    """边界目标长度在多个种子下都满足精度（每个种子一个新索引，连续组装 6 次）。"""
    c = default_counter()
    for seed in range(15):
        asm = make_assembler()
        for target in (1500, 1501, 2000, 4000, 5980, 6000):
            req = asm.assemble(target, Random(seed * 1000 + target))
            assert abs(req.input_tokens - target) <= 20, (seed, target, req.input_tokens)
            assert req.input_tokens == c.count_messages_full(req.messages)
