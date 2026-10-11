# Implementation Plan: Azure GPT 压力测试系统（azure-gpt-load-tester）

## Overview

使用 Python 3.11+ asyncio 按设计文档逐步实现：先搭建项目骨架与配置，再实现语料子系统（存储、话题、过滤、去重、规划、Gemini 调用、warmup），然后实现压测子系统（长度分布、索引、组装、限流、爬坡、节流、熔断、Azure 客户端、成本守卫），最后接入补充任务、调度器、运行控制、指标与 CLI。测试使用 pytest + pytest-asyncio + hypothesis（属性测试）+ respx（httpx mock），时钟与随机数全部注入（FakeClock、Random(seed)）。

说明：Gemini 客户端依赖 DualTokenBucket、ThrottleController 与 CostGuard，因此这三个共享组件放在 Gemini 步骤之前实现（任务 6），Azure 侧直接复用。ThrottleController 是同一实现的两种模式：Azure 使用 `mode="probe"`（NORMAL / HOLD / PROBE / RECOVER：429 → HOLD 遵守 Retry-After → 单个 Throttle_Probe → 探测成功后 10% → 25% → 50% → 75% → 100% 每档 15s，约 60s 回满），Gemini 使用 `mode="level"`（立即 10% + hold，10% → 50% → 100% 每档 10s，无探测）。节流统计（ThrottleStats：探测计数、各状态累计时长、状态时间线）进入控制台、JSONL、报告与 runs 表。

## Tasks

- [x] 1. 项目骨架与配置
  - [x] 1.1 创建项目骨架、依赖锁定与密钥忽略规则
    - 创建 `pyproject.toml`（入口 `loadtest = loadtest.cli:app`）、`requirements.lock`（按设计"依赖"表锁定精确版本）
    - 创建 `.gitignore`（包含 `.env`、凭据目录、`data/`、`runs/`）、`.env.example`（仅占位：`AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`、`GOOGLE_APPLICATION_CREDENTIALS`）、`config.example.yaml`（全部非密钥默认值）
    - 创建 `src/loadtest/` 包目录结构（corpus/、load/、cost/、metrics/、db/）与 `tests/unit`、`tests/property`、`tests/integration`，配置 pytest 与 pytest-asyncio
    - 实现 `src/loadtest/clock.py`：`Clock` 协议、`RealClock`、`FakeClock`（可推进时间、可注入 sleep）
    - _Requirements: 1.1, 29.6_

  - [x] 1.2 实现 Settings 加载与校验（`src/loadtest/settings.py`）
    - 按设计实现 AzureSettings、GeminiSettings、LoadSettings、ThrottleSettings、CorpusSettings、CostSettings、ModelPrice 与顶层 Settings（pydantic-settings，YAML + `.env`）
    - 密钥使用 `SecretStr`；缺失/类型不合法时输出参数名并以非零退出码终止，且不发起外部调用
    - 校验 `lang_ratio`（键恰好 {zh, en}、值 ∈ [0,1]、|Σ−1| ≤ 1e-9）、`startup_ramp`、单价 ≥ 0、预算上限 > 0
    - 校验 ThrottleSettings（Azure `throttle` 与 `gemini.throttle` 各自校验）：`levels` 严格递增、首项 0.10、末项 1.00；`recover_full_s > 0`；`soft_down_floor ∈ levels`；`min_cooldown_s ≤ max_retry_after_s`；`probe_inflight_wait_s > 0`、`probe_error_retry_s > 0`；Azure `mode == "probe"`（默认 levels [0.10, 0.25, 0.50, 0.75, 1.00]、recover_full_s 60）、Gemini `mode == "level"`（默认 levels [0.10, 0.50, 1.00]、recover_full_s 20）
    - 校验 `GOOGLE_APPLICATION_CREDENTIALS` 文件存在且可读，错误信息只含参数名不含文件内容
    - 提供 `redact()` / 配置快照函数，用掩码替换所有密钥值（供日志、`runs.config_json`、JSONL、CSV、报告使用）
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x]* 1.3 编写 Settings 单元测试（`tests/unit/test_settings.py`）
    - 覆盖缺失密钥、非法 lang_ratio / startup_ramp / 单价 / 预算、凭据文件不存在、默认值、配置快照中原始 key 子串出现次数为 0
    - 覆盖非法 ThrottleSettings：levels 非递增 / 首末项错误、soft_down_floor ∉ levels、min_cooldown_s > max_retry_after_s、probe 参数 ≤ 0、Azure 配为 level 或 Gemini 配为 probe；默认 recover_step_s 为 Azure 15s、Gemini 10s
    - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [x] 1.4 实现 token 工具（`src/loadtest/tokens.py`）
    - 封装 tiktoken `o200k_base`：`count()`、`count_messages()`（含消息开销）、`trim_to_tokens()`（先在句子边界裁剪，再按 token 截断）
    - _Requirements: 5.3, 14.1, 14.8_

  - [x]* 1.5 编写 token 工具单元测试（`tests/unit/test_tokens.py`）
    - 验证中英文句子边界裁剪、截断后 token 数不超过预算、消息开销计数稳定
    - _Requirements: 14.1, 14.8_

- [x] 2. SQLite 存储层
  - [x] 2.1 实现 schema 与 CorpusStore（`src/loadtest/db/schema.sql`、`src/loadtest/corpus/store.py`）
    - 按设计创建 passages / rejections / quarantine / runs / ledger 表与 ledger_totals 视图，开启 WAL、busy_timeout
    - 实现 `insert`（grounded 时 `expires_at = created_at + 72h`，否则 NULL；记录 `gen_model`）、`quarantine`、`expire_due`、`flush_use_counts`、`stats()`→PoolStats、`record_rejection`（只记类别、语言、模型、原因、时间）、runs 记录读写（含 `throttled`、`probes` 计数列与 `throttle_json` ThrottleStats 快照列）、ledger 批量写入（失败重试 3 次后抛出致命错误）、`ledger_totals` 读取、当日 UTC grounding 计数
    - 定义 `Passage` 数据类与 `load_active_index` 所需的查询（排除过期与非 active）
    - _Requirements: 4.1, 4.2, 4.5, 5.10, 9.4, 12.6, 24.1, 26.1, 26.4, 26.5, 28.4_

  - [x]* 2.2 编写 CorpusStore 单元测试（`tests/unit/test_store.py`）
    - 验证 expires_at 规则、过期置 expired、rejections 不含正文、use_count 累加、ledger 写入失败重试 3 次后报错、runs 记录 probes / throttle_json 往返一致
    - _Requirements: 4.1, 4.5, 5.10, 9.4, 26.5, 28.4_

- [x] 3. 语料数据文件与话题分类树
  - [x] 3.1 创建 `taxonomy.yaml` 与 TopicTaxonomy（`src/loadtest/corpus/taxonomy.py`）
    - 至少 14 个大类（含 13 个指定类别与扩展类），每类 10–30 个子话题、类别权重，仅 daily_news / daily_economy / sports_events 标记 grounding；排除敏感话题
    - 定义 `TopicSpec`；实现加载校验（列出全部不合规类别名）、`sample()`（语言、长度桶、target_len 落在桶区间、8 种体裁之一）、`penalize(subtopic, factor)`、`coverage()`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [x] 3.2 创建任务指令库 `src/loadtest/load/tasks.yaml` 与 TaskCatalog（`src/loadtest/load/tasks.py`）
    - 中文 ≥ 30 条、英文 ≥ 30 条；按语言均匀抽样；中文附加"只输出答案，不要解释，不超过 30 字。"，英文附加 "Answer only, no explanation, under 20 words."
    - _Requirements: 14.4, 14.6, 14.7_

  - [x] 3.3 创建 `src/loadtest/corpus/blocklist.txt` 与提示词模板（`src/loadtest/corpus/prompts.py`）
    - 中英关键词黑名单；中/英提示词模板（3–6 段、只写公认事实、不编造、无 Markdown），zh 目标字数 ∈ target_len × [1.0, 1.3]，en 目标词数 = target_len × 0.75；grounding 类别附加"仅依据检索报道、注明事件日期"
    - _Requirements: 3.2, 3.3, 3.4, 5.7_

  - [x]* 3.4 编写数据文件与提示词单元测试（`tests/unit/test_taxonomy_tasks_prompts.py`）
    - 验证类别数与子话题数、非法 taxonomy 拒绝加载并列出类别名、grounding 标记、sample 输出合法、任务指令数量、提示词长度换算
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 3.3, 14.7_

- [x] 4. 安全过滤与去重
  - [x] 4.1 实现 SafetyFilter（`src/loadtest/corpus/safety.py`）
    - 按 finish_reason → length → language → refusal → blocklist → structure → grounding 顺序执行，首个失败即停止并返回该原因
    - SAFETY / 空文本 → `safety_block`（由调用方将子话题权重 × 0.5）；非 STOP/SAFETY → `structure`；500–1500 token 并归入 S/M/L；字符类别占比语言检测（zh CJK ≥ 60%；en 字母 ≥ 80% 且 CJK < 2%）；拒答正则；黑名单；≥ 3 段、无 `#` 开头行、无规范化后重复段落；grounded 需 ≥ 1 来源且含日期
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.12_

  - [x]* 4.2 编写 SafetyFilter 单元测试（`tests/unit/test_safety.py`）
    - 每个规则各自的通过/拒绝样例，以及多规则同时失败时只记录首个原因
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.12_

  - [x] 4.3 实现 Deduplicator 并接入 CorpusStore.insert（`src/loadtest/corpus/dedup.py`、`store.py`）
    - 规范化（删除空白与 Unicode 标点）后 SHA256 精确匹配；自实现 64 位 SimHash，按 band 索引查询汉明距离 ≤ 3；对所有状态段落生效
    - 重复时不插入并写入 rejections（`duplicate`）
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x]* 4.4 编写去重属性测试（`tests/property/test_dedup.py`）
    - **Property 17: 去重**
    - **Validates: Requirements 6.1, 6.2, 6.3**

- [x] 5. 语料供应规划
  - [x] 5.1 实现 SupplyPlanner 缺口计算与任务输出（`src/loadtest/corpus/planner.py`）
    - `plan()`：语言 × 类别 × 长度桶（S/M/L = 30/40/30）目标，取整后调整使各级之和等于上级，Σn = max(0, 目标 − active 未过期总量)
    - `next_spec()`：优先级 隔离替换 > 过期替换 > 语言 > 类别 > 长度桶；类别权重与缺口成正比；当日 grounding 达上限时移除 grounding 类别；子话题样本 ≥ 20 且拒绝率 > 30% 时降权 50%；支持类别/语言过滤（供 generate 使用）
    - _Requirements: 3.5, 5.11, 7.1, 7.2, 7.3, 7.4, 10.7_

  - [x] 5.2 实现运行前供应检查 `check_for_run`（`src/loadtest/corpus/planner.py`）
    - 预期请求数 = target_tpm × duration_s / 60 / 3040；所需库存 = ceil(预期请求数 × 2.5 × lang_ratio × (1 + supply_margin) / per_run_max_uses)
    - refuse / warn（总量 < healthy_size、语言占比偏离 > 5pp、桶占比 < 15%）/ ok，输出告警与建议 warmup 命令
    - _Requirements: 8.1, 8.2, 8.4, 8.5_

  - [x]* 5.3 编写 SupplyPlanner 单元测试（`tests/unit/test_planner.py`）
    - 验证缺口总和、各级取整之和、优先级顺序、grounding 上限移除、降权逻辑
    - _Requirements: 3.5, 5.11, 7.1, 7.2, 7.3_

  - [x]* 5.4 编写供应检查属性测试（`tests/property/test_supply_check.py`）
    - **Property 25: 供应检查**（判定部分；"refuse 时 0 个 Azure 请求"部分在 16.3 验证）
    - **Validates: Requirements 8.1, 8.2, 8.4, 8.5**

- [x] 6. 共享限流、节流与成本组件
  - [x] 6.1 实现 DualTokenBucket（`src/loadtest/bucket.py`）
    - 单一实例同时扣 cost 个 token 与 1 个请求；tok_cap = max(TPM×5/60, 6040)、req_cap = max(RPM×5/60, 1)；初始存量 = 2s 速率（请求 ≥ 1）；`set_rates` 降速时立即截断存量；全局锁 FIFO 放行，等待步长 ≤ 0.25s；注入 Clock
    - _Requirements: 11.1, 17.1, 17.2, 17.4, 17.5, 17.6_

  - [x]* 6.2 编写限流上界属性测试（`tests/property/test_bucket_bound.py`）
    - **Property 6: 限流不超预算**
    - **Validates: Requirements 17.1, 17.2, 17.6, 11.1**

  - [x]* 6.3 编写降档生效属性测试（`tests/property/test_bucket_downshift.py`）
    - **Property 8: 降档生效**
    - **Validates: Requirements 17.4**

  - [x] 6.4 实现 ThrottleController（`src/loadtest/throttle.py`，Azure / Gemini 共用实现，两种模式）
    - 定义 `ThrottleState`（NORMAL / HOLD / PROBE / RECOVER）、`DispatchTicket(dispatched_at, is_probe)`、`ThrottleStats`（throttle_events、stale_throttles、probes_sent、probe_ok、probe_throttled、probe_error、probe_content_filter、consecutive_probe_fail、max_consecutive_probe_fail、soft_downs、time_in_state、timeline）；`recover_step_s = recover_full_s / (len(levels) − 1)`
    - `on_throttle(retry_after, now, dispatched_at)`：cooldown = min(max(retry_after 或 0, min_cooldown_s), max_retry_after_s)；`dispatched_at < hold_started_at` 的在途旧请求计 stale_throttles，HOLD 中只令 hold_until 取 max，PROBE / RECOVER 中忽略；否则计 throttle_events 并进入 HOLD（含 RECOVER 中的新节流），hold_until 只增不减，level = levels[0]
    - `on_probe_result`：OK → RECOVER、level = levels[0]、重置升档计时；THROTTLED → 新 HOLD 周期（新 Retry-After）并累加连续探测失败数；SERVER_ERROR / TIMEOUT / CLIENT_ERROR → 保持 PROBE，`probe_ready_at = now + probe_error_retry_s`（5s）；CONTENT_FILTER → 保持 PROBE，立即可再探
    - `on_success(headers, now)`：仅 NORMAL / RECOVER；remaining/limit < 10% 时降一级（不低于 soft_down_floor 0.50、不进入 HOLD、5s 冷却，缺失头不判定），NORMAL → RECOVER，重置升档计时
    - `on_result(ticket, res, now)` 统一入口（探测 → on_probe_result；THROTTLED → on_throttle(dispatched_at)；OK → on_success）；`note_dispatch(now, is_probe)` 返回 ticket 并维护 in_flight / probe_in_flight / probes_sent；`note_done(ticket)`
    - `allowed_dispatch(now) → (max_new, level)`：HOLD → (0, 0)；PROBE 时探测在途、未到 probe_ready_at、或仍有在途请求且等待未满 `probe_inflight_wait_s`（30s）→ (0, 0)，否则 (1, levels[0])；NORMAL / RECOVER → (None, level)
    - `tick(now)`：累加 time_in_state；HOLD 到期 → probe 模式进入 PROBE（记录 probe_phase_start），level 模式直接进入 RECOVER；RECOVER 中距最近事件 ≥ recover_step_s 升相邻一级，到 1.0 → NORMAL；`is_holding(now)`；`_set_state` 记录 timeline
    - mode="level"（Gemini）：无 PROBE，hold 结束直接 RECOVER，10% → 50% → 100% 每档 10s
    - _Requirements: 1.5, 11.2, 19.1, 19.2, 19.4, 19.6, 19.7, 20.1, 20.2, 20.3, 20.4, 20.5, 20.7, 21.1, 21.2, 21.3, 21.4_

  - [x]* 6.5 编写探测后阶梯回满属性测试（`tests/property/test_throttle_recovery.py`）
    - **Property 11: 探测成功后 60 秒阶梯回满**（mode=probe 下 RECOVER 只能由 OK 探测进入、每次只升相邻一级且间隔 ≥ 15s、探测成功到 100% ∈ [60s, 61s]、RECOVER 中新节流回到 HOLD；"tick 调用间隔 ≤ 1s"部分在 15.4 调度器模拟中验证）
    - **Validates: Requirements 20.1, 20.4, 20.5, 20.6, 20.7**

  - [x]* 6.6 编写主动减速属性测试（`tests/property/test_throttle_soft_down.py`）
    - **Property 12: 主动减速下限**（无节流头序列下 state 永不为 HOLD / PROBE、level ≥ 0.50、每次只降一级、5s 内最多一次、HOLD / PROBE 中不触发、缺失头维度不判定）
    - **Validates: Requirements 21.1, 21.2, 21.3, 21.4**

  - [x] 6.7 实现定价与 CostGuard（`src/loadtest/cost/pricing.py`、`src/loadtest/cost/guard.py`）
    - `reserve`（输入 × 1.02 + max_tokens 计价，检查 cap − safety_margin 与可选 per_run_cap）、`settle`（按 usage，缓存输入按 cached_input_per_m；无 usage 按预留）、`release`、`remaining_ratio`、`estimate_run_cost`、按 provider/model 累积待刷写账本、启动时从 ledger_totals 读取历史花费（Gemini 含 gemini 与 gemini_grounding）、grounding 按次计价
    - 默认上限 Azure $180、Gemini $250
    - _Requirements: 25.1, 25.2, 25.3, 25.4, 25.6, 26.2_

  - [x]* 6.8 编写成本守卫属性测试（`tests/property/test_cost_guard.py`）
    - **Property 15: 成本守卫**（"reserve 返回 None 后调度器不再派发"部分在 15.7 验证）
    - **Validates: Requirements 25.2, 25.4, 25.5, 25.6, 11.5**

  - [x]* 6.9 编写账本持久化属性测试（`tests/property/test_ledger.py`）
    - **Property 16: 账本持久化**
    - **Validates: Requirements 26.1, 26.2, 26.3**

  - [x]* 6.10 编写探测状态机属性测试（`tests/property/test_throttle_probe.py`）
    - **Property 10: HOLD 零派发、PROBE 单请求**（控制器层：任意 429 / 在途返回 / 探测结果交错下，HOLD 期间 allowed_dispatch == (0, 0)；PROBE 期间探测在途时 max_new == 0、每次至多 1 个探测；在途清空或等待满 30s 才放行探测；探测 THROTTLED → 新 HOLD 后再探、次数不限；探测 ERROR → 下一探测 ≥ 5s 后；探测 CONTENT_FILTER → 不进入 RECOVER 且可立即再探；调度器层部分在 15.4 验证）
    - **Validates: Requirements 19.4, 19.7, 20.2, 20.3**

  - [x]* 6.11 编写 ThrottleController 单元测试（`tests/unit/test_throttle.py`）
    - mode="level"：429 → 10% + hold，hold 结束直接 RECOVER，10% → 50% → 100% 每档 10s、无 PROBE
    - mode="probe"：设计第 6 节时间线示例（t=0 429、t=3 在途 429 延长 hold 至 13、t=13 PROBE、t=15 探测 OK、t=75 回满）与探测再次 429 分支
    - ThrottleStats：各计数、time_in_state 总和 == 经过时长、timeline 记录每次状态切换
    - _Requirements: 11.2, 19.7, 20.1, 20.2, 20.3, 20.5, 28.4_

- [x] 7. Gemini 调用链
  - [x] 7.1 实现 GeminiRateLimiter（`src/loadtest/corpus/gemini_limiter.py`）
    - 基于 DualTokenBucket：RPM = quota_rpm × usage_ratio × Gemini_Throttle 档位，TPM 同理；档位变化时更新速率；warmup 与 Replenisher（≤ 30 RPM）各自独立实例
    - Gemini_Throttle 为 `ThrottleController(gemini.throttle)`，固定 `mode="level"`（levels [0.10, 0.50, 1.00]、每档 10s、无 PROBE）；limiter 使用其 `level` 与 `is_holding`，不调用 allowed_dispatch / 探测接口
    - _Requirements: 10.1, 11.1, 11.2_

  - [x] 7.2 实现 ModelChain（`src/loadtest/corpus/model_chain.py`）
    - 默认链 3.1-flash-lite → 3.5-flash-lite → 3.8-flash；连续失败 5 次切换（末位保持并告警）；成功清零；600s 后探测主模型，连续 3 次成功恢复，探测失败回到备用并重新计时；预算剩余 < 20% 时不选 gemini-3.8-flash
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5_

  - [x]* 7.3 编写模型链属性测试（`tests/property/test_model_chain.py`）
    - **Property 27: 模型链切换**
    - **Validates: Requirements 12.1, 12.2, 12.3, 12.4, 12.5**

  - [x] 7.4 实现 GeminiClient 与 gemini_call（`src/loadtest/corpus/gemini_client.py`）
    - google-genai Vertex AI 模式 + 服务账号凭据；grounding 类别启用 Google Search 工具并提取来源 URL；返回 GeneratedText（正文、模型、token、finish_reason、grounded、sources、attempts）
    - 每次尝试：`wait_if_holding`（按 Gemini_Throttle.is_holding 以 ≤ 1s 步长等待并调用 tick）→ limiter.acquire → CostGuard.reserve（拒绝则抛 GeminiBudgetExhausted）→ 60s 超时调用 → settle（成功按实际模型与 usage；超时/5xx 按预留）
    - 429/RESOURCE_EXHAUSTED → Gemini_Throttle.on_throttle（level 模式：立即 10% + hold，无探测）+ 退避重试；5xx/超时 → 退避重试；其他 4xx 与 SAFETY 不重试；退避 = min(cap, max(Retry-After 或 0, Uniform(0, min(cap, base × 2^n))))；尝试耗尽后向 ModelChain 报告失败
    - 生成成功后执行 SafetyFilter + Dedup + 入库（SAFETY 时子话题降权），失败写 rejections
    - _Requirements: 3.1, 3.4, 3.6, 5.1, 11.2, 11.3, 11.4, 11.5, 11.6, 11.7, 12.6_

  - [x]* 7.5 编写 Gemini 重试与退避属性测试（`tests/property/test_gemini_retry.py`）
    - **Property 26: Gemini 重试与退避**（使用可编程假 genai 客户端与 FakeClock；Gemini_Throttle 为 level 模式，429 后档位立即 == 0.10 且 hold 期间尝试数为 0）
    - **Validates: Requirements 11.2, 11.3, 11.4, 11.6, 11.7**

  - [x]* 7.6 编写 GeminiClient 单元测试（`tests/unit/test_gemini_client.py`）
    - 验证 grounding 工具配置与来源提取、GeneratedText 字段、SAFETY 不重试并降权、预留被拒时不发起调用
    - _Requirements: 3.1, 3.4, 3.6, 5.1, 11.6_

- [x] 8. warmup 与 generate 流程
  - [x] 8.1 实现 WarmupRunner 与定向 generate（`src/loadtest/corpus/warmup.py`）
    - 开始时 `store.expire_due`；缺口为 0 时不调用 Gemini 并输出库存；以 `--concurrency`（默认 32）并发生成；结束条件：缺口填满 / Gemini CostGuard 拒绝 / 本次花费达 `--max-cost`
    - generate：仅生成 `--category` 所列类别（可选 `--lang`），入库数 ≤ `--count`，grounding 受每日上限约束
    - 写入 runs（kind = warmup / generate）与 ledger；输出按语言/类别/长度桶的库存、拒绝率与花费报告
    - _Requirements: 3.5, 3.7, 4.5, 4.6, 7.6, 26.1_

  - [x]* 8.2 编写 warmup / generate 单元测试（`tests/unit/test_warmup.py`）
    - 使用假 Gemini 客户端验证三种结束条件、缺口为 0 时零调用、generate 类别过滤与数量上限
    - _Requirements: 3.7, 4.6, 7.6_

- [x] 9. Checkpoint - 语料子系统
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. 长度分布、段落索引与请求组装
  - [x] 10.1 实现 LengthDistribution（`src/loadtest/load/distribution.py`）
    - 80% 截断正态 N(3000, (3500−2500)/3.33) 拒绝采样并取整；10% 均匀 [1500, 2500)；10% 均匀 (3500, 6000]
    - _Requirements: 13.1, 13.2, 13.3_

  - [x]* 10.2 编写长度范围属性测试（`tests/property/test_length_range.py`）
    - **Property 1: 长度范围**
    - **Validates: Requirements 13.1**

  - [x]* 10.3 编写长度分布属性测试（`tests/property/test_length_distribution.py`）
    - **Property 2: 长度分布**
    - **Validates: Requirements 13.2, 13.3**

  - [x] 10.4 实现 PassageIndex（`src/loadtest/corpus/index.py`）
    - 按 (lang, bucket) 分桶；`pick`：先选包含 near（或最近）的桶，排除 exclude 与 run_uses ≥ per_run_max_uses，按 1/(1+use_count+run_uses)^α 前缀和加权抽样，逐级放宽到其他桶，全部无候选返回 None；`pick` 时过滤 `expires_at ≤ now`
    - `mark_used`、`remove`、`ingest_pending`（get_nowait，最多 64 条）、`expire_due`、`inventory`、`run_uses` 导出
    - _Requirements: 4.2, 4.3, 4.4, 9.1, 9.2, 9.5, 10.3, 24.2_

  - [x]* 10.5 编写过期段落属性测试（`tests/property/test_expiry.py`）
    - **Property 24: 过期段落不被选中**
    - **Validates: Requirements 4.2, 4.3, 4.4**

  - [x] 10.6 实现 RequestAssembler（`src/loadtest/load/assembler.py`）
    - 按 lang_ratio 选语言；同语言任务指令；按预算分档确定初始段落数（< 2000 → 1–2；[2000, 4000) → 2–3；否则 3–4）；随机打乱段落、随机指令前/后位置
    - 超出预算：句子边界裁剪最后一段再按 token 截断；不足 > 20：段落 < 4 且有候选时追加截断片段，否则重新裁剪最后一段来源段落补足
    - 复算完整消息 token，保证 |input_tokens − target| ≤ 20；max_tokens = 40；对所用段落 mark_used；无候选抛 CorpusExhausted
    - _Requirements: 9.3, 9.5, 14.1, 14.2, 14.3, 14.4, 14.5, 14.6, 14.8, 14.9, 15.1, 15.2_

  - [x]* 10.7 编写组装精度属性测试（`tests/property/test_assembler_precision.py`）
    - **Property 3: 组装精度**
    - **Validates: Requirements 14.1, 14.8**

  - [x]* 10.8 编写段落约束属性测试（`tests/property/test_assembler_passages.py`）
    - **Property 4: 段落约束**
    - **Validates: Requirements 14.2, 14.3, 14.9**

  - [x]* 10.9 编写输出上限属性测试（`tests/property/test_assembler_max_tokens.py`）
    - **Property 5: 输出上限**
    - **Validates: Requirements 14.5**

  - [x]* 10.10 编写单次运行复用上限属性测试（`tests/property/test_reuse_cap.py`）
    - **Property 21: 单次运行复用上限**
    - **Validates: Requirements 9.1, 9.3, 9.5**

  - [x]* 10.11 编写语言配比属性测试（`tests/property/test_lang_ratio.py`）
    - **Property 22: 语言配比**
    - **Validates: Requirements 15.1, 15.2**

  - [x] 10.12 实现内容过滤隔离处理（`src/loadtest/load/quarantine.py`）
    - CONTENT_FILTER 时对请求涉及全部段落调用 `store.quarantine`（原因 `azure_content_filter`，记录命中的过滤类别）并立即 `index.remove`
    - _Requirements: 24.1, 24.2_

  - [x]* 10.13 编写隔离属性测试（`tests/property/test_quarantine.py`）
    - **Property 18: 隔离**（含重新加载索引后的跨运行验证）
    - **Validates: Requirements 24.1, 24.2**

- [x] 11. 启动爬坡、熔断与错误分类
  - [x] 11.1 实现 StartupRamp 与 Effective_Level（`src/loadtest/load/ramp.py`）
    - 阶梯函数，默认 [(0, 0.5), (10, 1.0)]；`--no-ramp` 时恒为 1.0；`effective_level = min(ramp, throttle.level)`
    - _Requirements: 18.1, 18.2, 18.3, 18.4_

  - [x]* 11.2 编写启动爬坡属性测试（`tests/property/test_ramp.py`）
    - **Property 13: 启动爬坡**（含桶初始存量 = 2s 速率）
    - **Validates: Requirements 18.1, 18.2, 18.3, 18.5, 17.5**

  - [x]* 11.3 编写档位上界属性测试（`tests/property/test_effective_level.py`）
    - **Property 14: 档位上界**（节流、探测结果、减速、爬坡任意交错：throttle.level ∈ levels 且 ≤ 1.0；NORMAL / RECOVER 下 min(ramp, level) ∈ levels ∪ ramp 档位；HOLD 下有效派发为 0；PROBE 下探测档位 == levels[0]）
    - **Validates: Requirements 18.4**

  - [x] 11.4 实现 CircuitBreaker（`src/loadtest/load/breaker.py`）
    - 只统计 OK 与 SERVER_ERROR/TIMEOUT/CLIENT_ERROR；60s 窗口样本 ≥ 20 且错误率 > 50%，或连续失败 10 次 → open 60s；half_open 最多 5 个探测，全部成功 → closed，任一失败 → open；记录 open 次数
    - _Requirements: 23.1, 23.2, 23.4, 23.6_

  - [x]* 11.5 编写 CircuitBreaker 单元测试（`tests/unit/test_breaker.py`）
    - 验证 THROTTLED/CONTENT_FILTER 不计样本也不重置连续失败、open/half_open/closed 转换、half_open 探测上限
    - _Requirements: 23.1, 23.2, 23.4, 23.6_

  - [x] 11.6 实现 classify / parse_retry_after / is_quota_error（`src/loadtest/load/azure_client.py`）
    - 按设计第 7 节实现 Outcome 分类（超时 → TIMEOUT；无状态码 → SERVER_ERROR；429/503/403-quota → THROTTLED；400 content_filter 或 200 finish_reason=content_filter → CONTENT_FILTER 等）
    - Retry-After：retry-after-ms → retry-after（秒或 HTTP 日期）；无法解析/负值视为缺失；过去日期取 0；> 120 截断并告警
    - _Requirements: 19.3, 22.1, 22.2, 22.5_

  - [x]* 11.7 编写节流立即进入 HOLD 属性测试（`tests/property/test_throttle_immediate.py`）
    - **Property 9: 节流立即进入 HOLD**（NORMAL / RECOVER / PROBE 及爬坡期间；parse_retry_after 结果 ∈ [0, 120] ∪ {None}；非在途 Throttle_Event 后 state == HOLD、allowed_dispatch == (0, 0)、hold_until 下界成立且只增不减；HOLD 前派发的在途 429 在 HOLD 中只令 hold_until 取 max、在 PROBE / RECOVER 中被忽略，不改变状态、不重置恢复进度）
    - **Validates: Requirements 19.1, 19.2, 19.3, 19.6**

  - [x]* 11.8 编写错误分类属性测试（`tests/property/test_classify.py`）
    - **Property 19: 错误分类全覆盖**（"重试次数为 0"部分在 12.2 用 respx 验证）
    - **Validates: Requirements 22.1, 22.2, 22.5, 19.5, 24.4**

- [x] 12. Azure 客户端
  - [x] 12.1 实现 AzureClient.send（`src/loadtest/load/azure_client.py`）
    - 封装 AsyncAzureOpenAI（max_retries = 0、timeout 30s、连接池 = max_concurrency），使用 with_raw_response 读取 x-ratelimit-remaining/limit-tokens/requests 与 retry-after(-ms) 头（构造 `RateLimitHeaders` 供 on_success），提取 usage（含 cached_tokens）与内容过滤类别，返回 AzureResult；THROTTLED / CONTENT_FILTER 不重试（Throttle_Probe 也不重试，失败后的再探测是新组装的请求）
    - _Requirements: 19.5, 22.2, 22.3, 24.4_

  - [x]* 12.2 编写 AzureClient respx 单元测试（`tests/unit/test_azure_client.py`）
    - 覆盖 200（usage 与 ratelimit 头）、429 带各种 Retry-After、503、403 quota/非 quota、400 content_filter、500/502/504、超时、连接失败；断言每个请求只发送 1 次
    - _Requirements: 19.3, 19.5, 22.2, 22.3, 22.5, 24.4_

- [x] 13. Checkpoint - 压测核心组件
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. 后台语料补充
  - [x] 14.1 实现 Replenisher（`src/loadtest/corpus/replenisher.py`）
    - 独立 asyncio 任务：独立 GeminiRateLimiter（≤ 30 RPM）、并发 ≤ 4、独立 aiosqlite 连接；优先替换隔离/过期段落，再按缺口补充
    - 入库后 put_nowait 到容量 1024 的队列，满则丢弃通知；任意异常捕获记录后 5s 继续；GeminiBudgetExhausted 时记录 1 条告警并停止调用；停止信号后不再发起新调用
    - _Requirements: 10.1, 10.2, 10.4, 10.6, 10.7, 10.8_

  - [x]* 14.2 编写 Replenisher 单元测试（`tests/unit/test_replenisher.py`）
    - 验证队列满时段落仍入库、异常后继续、预算耗尽后零调用、替换优先级、停止后不再调用
    - _Requirements: 10.2, 10.4, 10.6, 10.7, 10.8_

- [x] 15. 调度器
  - [x] 15.1 实现请求 worker 结果处理（`src/loadtest/load/worker.py`）
    - `worker(planned, reservation, ticket: DispatchTicket)`：发送 PlannedRequest；结算（THROTTLED → 0；OK → 按 usage；超时/无 usage → 按预留）；在处理响应时同步调用 `throttle.on_result(ticket, res, now)`（探测 → on_probe_result；普通 THROTTLED → on_throttle(retry_after, now, ticket.dispatched_at) 以识别在途旧请求；OK → on_success(headers)）
    - CONTENT_FILTER → 隔离并记录 60s 窗口命中（探测命中同样隔离）；OK 与 SERVER_ERROR/TIMEOUT/CLIENT_ERROR 记入熔断（含探测请求）；计数节流
    - `finally` 中调用 `throttle.note_done(ticket)` 并释放并发信号量；向指标传递 `is_probe`
    - _Requirements: 19.1, 19.5, 19.6, 19.7, 20.1, 20.3, 21.1, 23.1, 24.1, 24.4, 25.3_

  - [x] 15.2 实现 Scheduler 主循环（`src/loadtest/load/scheduler.py`）
    - 按设计第 4 节：ingest_pending、每 60s 过期检查、第 60s 起每 60s 随机游走 N(0, 30k²) 并截断到区间（`--tpm T` 时 [T×0.875, T×1.0625]）、所有等待经 `sleep_ticking` 拆成 ≤ 1s 小段并调用 throttle.tick
    - `blocked(t)`：`throttle.allowed_dispatch(t)` 的 max_new == 0（HOLD、PROBE 探测在途 / 未到重探时刻 / 等待在途请求清空 ≤ 30s）、熔断 open 或内容过滤暂停（60s 内 > 5 次 → 暂停 30s 并告警）时不派发，按 ≤ 1s 步长轮询；half_open 最多 5 个探测
    - NORMAL / RECOVER：eff_level = min(ramp, throttle.level)；PROBE：eff_level = levels[0]，跳过泊松等待，只派发 1 个 Throttle_Probe（正常组装、经 CostGuard 与令牌桶）
    - λ = min(eff_tpm / EWMA, eff_rpm) / 60，EWMA α = 0.05 初值 3040；eff_rpm = min(rpm_max, quota_rpm × 0.5) × eff_level；派发前 set_rates
    - 派发顺序：组装 → CostGuard.reserve（None → stop_reason=budget）→ bucket.acquire → 并发信号量（上限 64）；获取期间停止、进入阻塞状态或 PROBE 状态发生变化则 release 预留与信号量、不派发；CorpusExhausted → stop_reason=corpus_exhausted
    - 派发时 `ticket = throttle.note_dispatch(now, is_probe=probing)` 并把 ticket 交给 worker；仅实际派发后更新 EWMA
    - 爬坡零点 = 主循环开始时刻；停止信号后不再派发
    - _Requirements: 4.3, 10.3, 10.5, 16.1, 16.2, 16.3, 16.4, 16.6, 16.7, 17.3, 18.4, 18.5, 19.4, 20.2, 20.3, 20.6, 23.3, 23.6, 24.3, 25.5, 27.3, 27.8_

  - [x]* 15.3 编写全局 60 秒窗口属性测试（`tests/property/test_global_window.py`）
    - **Property 7: 全局 60 秒窗口**（FakeClock 离散事件模拟，含随机游走、爬坡与 `--tpm T`）
    - **Validates: Requirements 17.3, 16.2, 16.3, 16.6, 16.7, 16.8**

  - [x]* 15.4 编写 HOLD 零派发与 PROBE 单请求调度器属性测试（`tests/property/test_hold_no_dispatch.py`）
    - **Property 10: HOLD 零派发、PROBE 单请求**（调度器层 FakeClock 离散事件模拟，任意 429 / 在途返回 / 探测结果交错：HOLD 期间新派发为 0；PROBE 期间每次至多派发 1 个探测且结果返回前不派发其他请求、不做泊松等待；在途清空或等待满 30s 后才发探测；JSONL 记录中 is_probe 与实际探测一致）
    - 同时断言调度器调用 tick 的间隔（含 HOLD、PROBE 与泊松间隔内）≤ 1s，补充 **Property 11** 的 tick 部分
    - **Validates: Requirements 19.4, 19.7, 20.2, 20.3, 20.6**

  - [x]* 15.5 编写补充任务不阻塞调度器属性测试（`tests/property/test_replenisher_isolation.py`）
    - **Property 23: 补充任务不阻塞调度器**（随机挂起/抛异常/队列满的假补充任务与真实调度器同一事件循环）
    - **Validates: Requirements 10.4, 10.5**

  - [x]* 15.6 编写调度器单元测试（`tests/unit/test_scheduler.py`）
    - 验证 reserve 返回 None 后零派发且 stop_reason=budget、CorpusExhausted 停止、内容过滤暂停 30s、熔断 open 期间零派发、half_open 探测上限、EWMA 更新
    - 验证获取令牌/信号量期间进入 HOLD 或 PROBE 状态变化时释放预留（CostGuard.reserved 回到原值）且不派发；PROBE 下派发的 ticket.is_probe 为真
    - _Requirements: 16.1, 16.4, 20.2, 23.3, 23.6, 24.3, 25.5, 27.8_

- [x] 16. 运行控制器
  - [x] 16.1 实现 RunController（`src/loadtest/load/runner.py`）
    - 启动顺序：配置校验 → pid 单实例锁（存活拒绝；失效覆盖；失败退出时删除）→ `--tpm` / `--rpm-max` 上界校验 → 供应检查（refuse 且无 `--allow-low-supply` → stop_reason=supply_refused、0 个 Azure 请求、输出缺口与 warmup 命令；warn → 输出告警）→ 成本预估（超限且无 `--allow-partial` → 拒绝）→ 启动 Probe_Request（经 CostGuard；401/404 退出；THROTTLED → `throttle.on_throttle(retry_after)` 后启动，主循环从 HOLD → PROBE 开始；5xx/超时 5s 间隔重试 3 次；`--dry-run` 跳过；与运行中的 Throttle_Probe 区分）→ 加载 PassageIndex → 启动 Scheduler 与 Replenisher
    - 停止：SIGINT/SIGTERM/`runs/STOP`/duration 到期 → stop_reason（stop_cmd/signal/duration）；Drain ≤ 30s，未完成按预留结算；第二次 SIGINT 立即结算退出；熔断第 3 次 open → circuit_open；每 10s 刷写账本与 use_count，账本写失败 → stop_reason=error 非零退出
    - 收尾：run_uses 累加回 use_count、刷写账本、写 runs 记录（含 throttled、probes 与 ThrottleStats 快照 throttle_json）、生成报告、删除 pid 文件
    - `--dry-run`：只运行调度与组装（Azure 调用替换为模拟结果），不写 Azure 账本
    - _Requirements: 8.3, 8.6, 8.7, 9.4, 16.7, 16.8, 22.4, 22.6, 22.7, 22.8, 22.9, 23.5, 25.7, 26.4, 26.5, 27.1, 27.2, 27.4, 27.5, 27.6, 27.7, 27.8, 27.9_

  - [x]* 16.2 编写优雅停止属性测试（`tests/property/test_graceful_stop.py`）
    - **Property 20: 优雅停止**
    - **Validates: Requirements 27.3, 27.4, 27.5, 9.4**

  - [x]* 16.3 编写供应拒绝零请求测试（`tests/unit/test_runner_supply.py`）
    - 补充 **Property 25**：refuse 且无 `--allow-low-supply` 时 Azure 请求数（含探测）为 0；`--allow-low-supply` 时启动并告警；warn 时启动并输出告警
    - _Requirements: 8.3, 8.6, 8.7_

  - [x]* 16.4 编写 RunController 单元测试（`tests/unit/test_runner.py`）
    - 验证 pid 锁（存活/失效）、`--tpm` 越界拒绝、成本预估拒绝、启动探测 401/404/THROTTLED（主循环开始时 state == HOLD）/5xx 路径、stop_reason 判定、二次 SIGINT、账本写失败退出、runs.probes 与 throttle_json 已写入
    - _Requirements: 16.8, 22.4, 22.8, 22.9, 25.7, 26.5, 27.2, 27.6, 27.9_

- [x] 17. 指标与报告
  - [x] 17.1 实现指标收集器（`src/loadtest/metrics/collector.py`）
    - 60s 滑动窗口 TPM/RPM、固定桶直方图 P50/P95（输入、延迟）、平均输出 token、各 Outcome 累计计数（区分普通请求与探测）、语言计数、派发时长、复用分布
    - 读取 ThrottleController.stats 快照：当前状态与档位、throttle_events / stale_throttles、探测计数（sent / ok / 429 / err / cf、连续失败最大值）、soft_downs、time_in_state、timeline
    - _Requirements: 15.3, 16.5, 28.1_

  - [x] 17.2 实现控制台、JSONL 与 CSV 输出（`src/loadtest/metrics/console.py`、`src/loadtest/metrics/writers.py`）
    - 每 10s rich 控制台输出全部字段，并显示 throttle 状态（NORMAL / HOLD / PROBE / RECOVER）、档位、ramp、节流事件与 stale 数、探测计数（ok / 429 / err / cf）、各状态累计时长
    - 每请求一行 JSONL（含 `is_probe` 字段，不含原文与密钥）；每 10s 一行 CSV（含 throttle 状态、档位、探测计数与各状态累计时长）；所有输出经 redact
    - _Requirements: 1.7, 28.1, 28.2, 28.3_

  - [x] 17.3 实现运行报告（`src/loadtest/metrics/report.py`）
    - 生成 `report.json` 与 `report.md`：stop_reason、请求与 token 统计、实际平均 TPM/RPM、节流次数、zh/en 占比（1 位小数）、复用分布、补充入库数、花费；平均输出 token 不在 20–40 时输出偏离提示
    - 报告包含全部 ThrottleStats 字段与每次 Throttle_Event 的时间线（进入 HOLD 时刻、探测次数与结果、回满时刻）；提供写入 `runs.probes` 与 `runs.throttle_json` 的序列化函数
    - _Requirements: 15.3, 16.5, 28.4, 28.5_

  - [x]* 17.4 编写指标与报告单元测试（`tests/unit/test_metrics.py`）
    - 验证窗口统计、百分位、报告字段、偏离提示、JSONL/CSV/报告中原始 API key 子串出现 0 次
    - 验证 JSONL is_probe 字段、控制台/CSV 中 throttle 状态与探测计数、report.md 中 ThrottleStats 与时间线、throttle_json 可反序列化
    - _Requirements: 1.7, 15.3, 16.5, 28.1, 28.2, 28.3, 28.4, 28.5_

  - [x] 17.5 将指标接入 worker、Scheduler 与 RunController
    - 修改 `worker.py`、`scheduler.py`、`runner.py`：记录每请求结果（含 is_probe）、每 10s 输出与刷写（含 ThrottleStats 快照）、结束时生成报告并把 probes / throttle_json 写入 runs（任意 stop_reason）
    - _Requirements: 28.1, 28.2, 28.3, 28.4_

- [x] 18. CLI 接线
  - [x] 18.1 实现 typer CLI（`src/loadtest/cli.py`）
    - 命令 warmup / generate / run / stop / status / budget / report / pool-stats 及设计中的全部选项；非法选项输出选项名并非零退出
    - stop：写 `runs/STOP` 并向 pid 发 SIGTERM，无运行中压测则提示并非零退出；status：依据 pid 存活输出 run_id 与已运行时间；budget：分模型已用、上限、20 分钟预估与剩余可运行次数；report [RUN_ID]：不存在则非零退出；pool-stats：设计"供应监控"中的全部统计
    - 不暴露任何网络端口
    - _Requirements: 3.7, 4.6, 7.5, 25.8, 27.2, 27.10, 29.1, 29.2, 29.3, 29.4, 29.5, 29.6, 29.7_

  - [x]* 18.2 编写 CLI 单元测试（`tests/unit/test_cli.py`）
    - 使用 typer CliRunner 验证命令与选项存在、非法选项退出码、stop/status/report/budget/pool-stats 输出
    - _Requirements: 25.8, 27.10, 29.1, 29.2, 29.3, 29.4, 29.5, 29.7_

- [x] 19. Checkpoint - 完整系统接线
  - Ensure all tests pass, ask the user if questions arise.

- [x] 20. 集成测试
  - [x]* 20.1 编写 dry-run 集成测试（`tests/integration/test_dry_run.py`）
    - `loadtest run --dry-run --duration 2m`（FakeClock）不调用 Azure、不发探测、不写 Azure 账本，输出预期 TPM/RPM、爬坡曲线、语言配比、复用与长度分布
    - _Requirements: 22.7, 27.7_

  - [x]* 20.2 编写 Mock Azure 端到端集成测试（`tests/integration/test_mock_azure.py`）
    - respx 可编程返回 200（usage 与 ratelimit 头）/429（Retry-After）/503/400 content_filter/5xx，FakeClock 驱动
    - 验证爬坡；429 → HOLD 期间零派发 → 恰好 1 个 Throttle_Probe → 探测成功后 10% → 25% → 50% → 75% → 100% 约 60s 回满；探测再次 429 → 回到 HOLD 并再探（循环）；探测 5xx → 5s 后重探；探测 content_filter → 隔离并立即重探；PROBE 前等待在途请求清空（≤ 30s）；HOLD 前派发的在途 429 只延长 hold；RECOVER 中新 429 → HOLD
    - 验证隔离、内容过滤暂停、熔断、优雅停止，以及控制台 / JSONL（is_probe）/ 报告中的 ThrottleStats
    - _Requirements: 18.1, 19.1, 19.4, 19.6, 19.7, 20.1, 20.2, 20.3, 20.5, 20.7, 23.2, 23.5, 24.1, 24.3, 27.3, 27.4, 28.4_

  - [x]* 20.3 编写 Mock Gemini warmup 集成测试（`tests/integration/test_mock_gemini.py`）
    - 注入 429、RESOURCE_EXHAUSTED、503、超时、SAFETY、语言错误、过短文本，验证缺口填充、重试、模型切换、过滤与去重
    - _Requirements: 3.7, 5.1, 6.4, 11.2, 11.3, 12.2_

  - [x]* 20.4 编写供应场景集成测试（`tests/integration/test_supply_scenarios.py`）
    - 构造 zh 不足 / en 不足 / 长度桶失衡 / 含过期段落的池，验证 refuse / warn / 过期过滤
    - _Requirements: 4.2, 8.2, 8.3, 8.4_

- [x] 21. 服务器部署文件
  - [x] 21.2 多 GCP 赠金账号预算与 Gemini 3 grounding 计费
    - ledger 增加 account 列（旧库自动迁移）；Gemini CostGuard 只累计当前账号（`cost.gemini_budget_account`，缺省 `gemini.project`）；`budget` 单列其他账号
    - grounding 预留 `grounding_reserve_queries` 次查询，成功后按 `web_search_queries` 去重数结算；`grounding_daily_limit` 单位改为查询数（默认 600）
    - _Requirements: 3.5, 25.1, 26.1, 26.2, 26.6_
  - [x] 21.3 上线前自检 `loadtest doctor`（`src/loadtest/doctor.py`）
    - tokenizer / 数据库库存 / Azure 鉴权、deployment、部署 TPM·RPM 限流头 / Gemini 链中每个模型（thinking_level）/ grounding 来源与计费查询数；花费经 CostGuard 记账（run_id = doctor-*）；任一 FAIL 退出码 1
    - systemd 单元设置 `TIKTOKEN_CACHE_DIR=/opt/a_azure_limit/data/tiktoken`（ProtectHome=yes 下 ~/.cache 不可用）
    - _Requirements: 29.1_
  - [x] 21.1 创建 systemd 单元与部署说明（`deploy/`）
    - `deploy/loadtest-run.service`（KillSignal=SIGTERM、TimeoutStopSec=60）、`deploy/loadtest-refresh.service` 与 `deploy/loadtest-refresh.timer`（每日 06:00 generate grounding 类别 300 条）
    - `deploy/README.md`：安装、`.env` 与凭据 `chmod 600`、warmup、tmux 中运行与 `loadtest stop` 的命令示例
    - _Requirements: 4.6, 27.1, 27.2_

- [x] 22. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests (hypothesis) validate the 27 universal correctness properties; all use FakeClock and seeded Random, no real network
- Unit tests validate specific examples and edge cases; respx mocks Azure HTTP, a programmable fake genai client mocks Gemini
- Shared DualTokenBucket / ThrottleController / CostGuard are implemented in task 6 before the Gemini client because both Gemini and Azure sides depend on them
- ThrottleController has two modes: Azure `probe` (HOLD → single Throttle_Probe → 60s stepped RECOVER) and Gemini `level` (10% + hold → 10s steps, no probe); Property 10 is tested at controller level (6.10) and scheduler level (15.4)
- "Probe_Request" (task 16.1) is the one-off startup check before the main loop; "Throttle_Probe" is the single request sent in PROBE state after each HOLD

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "1.4"] },
    { "id": 2, "tasks": ["1.3", "1.5", "2.1", "3.1", "3.2", "6.1", "6.4", "10.1", "11.1", "11.4", "11.6"] },
    { "id": 3, "tasks": ["2.2", "3.3", "4.3", "6.2", "6.3", "6.5", "6.6", "6.10", "6.11", "6.7", "10.2", "10.3", "10.4", "11.2", "11.3", "11.5", "11.7", "11.8"] },
    { "id": 4, "tasks": ["3.4", "4.1", "4.4", "5.1", "6.8", "6.9", "7.1", "7.2", "10.5", "10.6", "12.1"] },
    { "id": 5, "tasks": ["4.2", "5.2", "7.3", "7.4", "10.7", "10.8", "10.9", "10.10", "10.11", "10.12", "12.2"] },
    { "id": 6, "tasks": ["5.3", "5.4", "7.5", "7.6", "8.1", "10.13", "14.1", "17.1"] },
    { "id": 7, "tasks": ["8.2", "14.2", "15.1", "17.2", "17.3"] },
    { "id": 8, "tasks": ["15.2", "17.4"] },
    { "id": 9, "tasks": ["15.3", "15.4", "15.5", "15.6", "16.1"] },
    { "id": 10, "tasks": ["16.2", "16.3", "16.4", "17.5"] },
    { "id": 11, "tasks": ["18.1"] },
    { "id": 12, "tasks": ["18.2", "20.1", "20.2", "20.3", "20.4", "21.1"] }
  ]
}
```
