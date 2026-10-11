# Requirements Document

Azure GPT 压力测试系统（azure-gpt-load-tester）

## Introduction

本系统用于对用户自有的 Azure OpenAI 部署进行可控压力测试。系统使用 Google Cloud Vertex AI 上的低价 Gemini 模型预先生产高质量、多行业的中文与英文语料（zh 60% / en 40%），并将语料组装成长输入、短输出的请求，以泊松到达与目标随机游走的方式发送给 Azure 低价非推理模型（默认 gpt-4.1-nano）。

目标负载：每分钟约 800k token（配额 1M TPM），200–500 次请求/分钟（配额 1000 RPM），输入 1500–6000 token 且约 80% 落在 2500–3500，输出 20–40 token。系统启动时陡峭爬坡（前 10 秒 50%，之后 100%）。Azure 侧节流采用探测模式：收到 429 / 403 配额 / 503 时立即进入 HOLD 暂停全部新派发，遵守 Retry-After（至少 10 秒、最多 120 秒）；冷却结束后进入 PROBE，只发送 1 个探测请求；探测成功后进入 RECOVER，按 10% → 25% → 50% → 75% → 100% 每档 15 秒阶梯恢复（约 60 秒回满）；探测再次被节流则回到 HOLD。Gemini 侧节流保持简单档位模式（立即 10% + 冷却，10% → 50% → 100% 每档 10 秒，无探测）。系统内置成本守卫与跨运行持久化账本；支持手动启动、手动停止与固定时长运行。

本文档的需求由已批准的设计文档（design.md）推导而来。

## Glossary

- **Load_Tester**：本系统整体，包括 CLI、语料子系统与压测子系统。
- **Settings**：从 `config.yaml` 与 `.env` 加载并校验的全部配置参数。
- **CLI**：基于 typer 的命令行入口（`loadtest` 命令）。
- **Topic_Taxonomy**：行业大类 → 子话题 → 体裁三级话题分类树，存于 `taxonomy.yaml`。
- **TopicSpec**：一条语料生成任务的规格（类别、子话题、体裁、语言、是否 grounding、长度桶、目标长度）。
- **Gemini_Client**：通过 Vertex AI 调用 Gemini 模型生成语料的组件。
- **Grounding**：Vertex AI 的 Google Search 检索增强，用于生成真实新闻类语料。
- **Grounding_Category**：需要 Grounding 的类别：`daily_news`、`daily_economy`、`sports_events`。
- **Model_Chain**：Gemini 主模型与备用模型的有序列表及其切换逻辑。
- **Gemini_Rate_Limiter**：Gemini 侧的 RPM/TPM 双桶限流器。
- **Gemini_Throttle**：Gemini 侧的节流控制器实例（与 Throttle_Controller 实现相同、实例独立），以 Level_Mode 运行，档位集合默认 {10%, 50%, 100%}。
- **Safety_Filter**：入库前对生成文本做长度、语言、拒答、黑名单、结构、grounding 校验的组件。
- **Deduplicator**：基于 SHA256 精确匹配与 64 位 SimHash（汉明距离 ≤ 3）的去重组件。
- **Corpus_Store**：基于 SQLite（WAL 模式）的持久化存储，包含 passages、rejections、quarantine、runs、ledger 表。
- **Passage**：语料池中的一个段落，token 长度 500–1500，属于且仅属于一种语言（zh 或 en）。
- **Length_Bucket**：段落长度桶，S = 500–799、M = 800–1199、L = 1200–1500 token。
- **Passage_Index**：运行期间按（语言, 长度桶）分桶的内存段落索引。
- **Run_Uses**：单次运行内某段落被组装使用的次数。
- **Use_Count**：段落跨运行累计被使用的次数（持久化）。
- **Supply_Planner**：按语言 × 类别 × 长度桶计算库存缺口并输出生成任务的组件。
- **Supply_Check**：运行前的语料库存检查，结果为 ok / warn / refuse。
- **Replenisher**：压测运行期间在后台低速生成并补充语料的独立任务。
- **Length_Distribution**：输入目标长度的混合分布采样器。
- **Request_Assembler**：将段落与任务指令组装为 Azure 请求的组件。
- **Planned_Request**：已组装、待发送的请求，含消息、输入 token 数、max_tokens、段落 id 列表与语言。
- **Task_Catalog**：中英文任务指令库（`tasks.yaml`），中文与英文各至少 30 条。
- **Scheduler**：按泊松到达与目标随机游走调度 Azure 请求的主循环。
- **Dual_Token_Bucket**：全局唯一的 token 桶 + 请求桶双桶限流器。
- **Startup_Ramp**：启动阶梯爬坡，默认 [(0s, 50%), (10s, 100%)]。
- **Throttle_Controller**：Azure 侧节流控制器，以 Probe_Mode 运行，状态为 NORMAL / HOLD / PROBE / RECOVER，档位集合（`throttle.levels`）默认 {10%, 25%, 50%, 75%, 100%}。
- **Probe_Mode**：Throttle_Controller 的运行模式：Throttle_Event → HOLD → PROBE（发送 1 个 Throttle_Probe）→ 探测成功后 RECOVER → 档位回到 100% 后 NORMAL。
- **Level_Mode**：Gemini_Throttle 的运行模式：Throttle_Event → 档位 10% 并进入 HOLD → HOLD 结束直接 RECOVER（无探测）→ NORMAL。
- **NORMAL**：Throttle_Controller 档位为 100% 且无冷却、无探测的状态。
- **HOLD**：冷却状态，起止为 [进入时刻, hold_until)，期间 Scheduler 不派发任何新请求。
- **PROBE**：HOLD 结束后的探测状态，期间 Scheduler 只派发 1 个 Throttle_Probe。
- **RECOVER**：探测成功后（或主动减速后）的阶梯恢复状态，档位逐级升高至 100%。
- **Hold**：处于 HOLD 状态的冷却期；hold_until 为冷却结束时间。
- **Throttle_Probe**：节流探测请求，即 PROBE 状态下由 Scheduler 正常组装、经 Cost_Guard 与 Dual_Token_Bucket 派发的单个 Azure 请求，用于确认配额已恢复；与 Probe_Request 不同。
- **Throttle_Event**：Azure 返回 429、403（配额类错误）或 503。
- **Stale_Throttle**：在当前 HOLD 周期开始之前派发的在途请求所返回的 Throttle_Event。
- **recover_step_s**：RECOVER 每档最短停留时长，等于 recover_full_s / (len(levels) − 1)；Azure 默认 60 / 4 = 15 秒，Gemini 默认 20 / 2 = 10 秒。
- **ThrottleStats**：节流统计，包括 throttle_events、stale_throttles、probes_sent、probe_ok、probe_throttled、probe_error、probe_content_filter、consecutive_probe_fail、max_consecutive_probe_fail、soft_downs、各状态累计时长与状态切换时间线。
- **Effective_Level**：有效档位；Throttle_Controller 处于 NORMAL 或 RECOVER 时等于 min(Startup_Ramp 档位, Throttle_Controller 档位)，处于 PROBE 时 Throttle_Probe 使用 levels[0]，处于 HOLD 时不派发。
- **Circuit_Breaker**：基于错误率与连续失败次数的熔断器，状态为 closed / open / half_open。
- **Azure_Client**：封装 AsyncAzureOpenAI 的组件，SDK 自带重试关闭（max_retries = 0）。
- **Outcome**：Azure 请求结果分类：OK、THROTTLED、CONTENT_FILTER、CLIENT_ERROR、SERVER_ERROR、TIMEOUT。
- **Probe_Request**：启动探测请求，即 `loadtest run` 在供应检查之后、主循环开始之前发送的单个 Azure 请求，用于验证 endpoint、key 与 deployment；与 Throttle_Probe 不同。
- **Cost_Guard**：按 token 与单价估算花费、在放行前检查预算上限的组件；Azure 与 Gemini 各一个实例。
- **Ledger**：Corpus_Store 中跨运行累计的花费账本（ledger 表与 ledger_totals 视图）。
- **Metrics_Reporter**：负责控制台输出、CSV/JSONL 记录与运行报告的组件。
- **Run_Controller**：负责运行生命周期、信号处理、pid 文件与 STOP 文件的组件。
- **Drain**：停止时不再派发新请求、等待在途请求结算的收尾过程，默认上限 30 秒。
- **Fake_Clock**：测试用可注入时钟。

## Requirements

### Requirement 1: 配置加载与密钥保护

**User Story:** 作为运维人员，我希望通过配置文件与环境变量统一管理参数和密钥，以便安全、可重复地运行压测。

#### Acceptance Criteria

1. WHEN Load_Tester 启动任一 CLI 命令，THE Settings SHALL 从 `config.yaml` 加载非密钥参数，并从环境变量或 `.env` 加载 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT` 与 `GOOGLE_APPLICATION_CREDENTIALS`。
2. IF 必需参数（包括 `AZURE_OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT`、`GOOGLE_APPLICATION_CREDENTIALS` 与 `gemini.project`）缺失或类型不合法，THEN THE Settings SHALL 以非零退出码终止命令并输出缺失或不合法的参数名，且在终止前不发起任何 Azure 或 Gemini 调用。
3. IF `lang_ratio` 的键不是恰好 {zh, en}、任一值不在 [0, 1] 内，或各值之和 Σ 满足 |Σ − 1| > 1e-9，THEN THE Settings SHALL 拒绝该配置并输出校验错误。
4. IF `startup_ramp` 的时间不严格递增、首步时间不为 0、档位不在 (0, 1] 内或末步档位不等于 1.0，THEN THE Settings SHALL 拒绝该配置并输出校验错误。
5. IF 任一节流配置（Azure `throttle` 或 `gemini.throttle`）的 `levels` 不严格递增、首项不等于 0.10、末项不等于 1.00、`recover_full_s` 小于等于 0、`soft_down_floor` 不属于 `levels`、`min_cooldown_s` 大于 `max_retry_after_s`、`probe_inflight_wait_s` 小于等于 0 或 `probe_error_retry_s` 小于等于 0，或 Azure `throttle.mode` 不等于 `probe`，或 `gemini.throttle.mode` 不等于 `level`，THEN THE Settings SHALL 拒绝该配置并输出校验错误。
6. IF 任一单价小于 0 或任一预算上限小于等于 0，THEN THE Settings SHALL 拒绝该配置并输出校验错误。
7. THE Load_Tester SHALL 在日志、控制台输出、配置快照（`runs.config_json`）、JSONL、CSV 与报告中以掩码形式代替所有密钥值，使原始 API key 作为子串在上述输出中出现的次数为 0。
8. THE Settings SHALL 提供设计文档"参数设置总表"中列出的全部默认值，包括 Azure 配额 1,000,000 TPM / 1,000 RPM、目标 800,000 TPM、默认部署 gpt-4.1-nano、默认运行时长 20 分钟。
9. IF `GOOGLE_APPLICATION_CREDENTIALS` 指向的凭据文件不存在或不可读，THEN THE Settings SHALL 以非零退出码终止命令并输出参数名 `GOOGLE_APPLICATION_CREDENTIALS`，且不输出该文件的任何内容。

### Requirement 2: 话题分类树

**User Story:** 作为测试设计者，我希望语料覆盖广泛的行业话题，以便压测流量接近真实业务。

#### Acceptance Criteria

1. THE Topic_Taxonomy SHALL 至少包含 14 个大类，其中包括：影视技巧、多媒体技巧、中国小说、人物传记、中外文学欣赏、学术论文、每日新闻、每日经济、体育赛事、厨艺、自然、电商、运动，以及至少 1 个扩展类别。
2. THE Topic_Taxonomy SHALL 为每个大类提供 10–30 个子话题，并使每个 TopicSpec 的体裁属于以下 8 种之一：说明文、评论、教程、案例分析、摘要、对话、书评/影评、新闻综述。
3. WHEN Supply_Planner 请求采样，THE Topic_Taxonomy SHALL 返回一个语言属于 {zh, en} 之一、length_bucket 属于 {S, M, L} 之一且 target_len 落在该长度桶区间内的 TopicSpec。
4. THE Topic_Taxonomy SHALL 仅对 Grounding_Category 中的类别将 `TopicSpec.grounding` 设为真。
5. THE Topic_Taxonomy SHALL 排除政治、宗教、暴力、色情、违法活动、医疗诊断、投资建议与争议性话题。
6. IF `taxonomy.yaml` 中任一大类的子话题数不在 10–30 内，或任一非 Grounding_Category 的类别被标记为需要 Grounding，THEN THE Topic_Taxonomy SHALL 拒绝加载并列出全部不合规的类别名。

### Requirement 3: Gemini 语料生成

**User Story:** 作为测试设计者，我希望用低价 Gemini 模型生成高质量、真实的语料，以便为 Azure 压测提供输入。

#### Acceptance Criteria

1. THE Gemini_Client SHALL 使用服务账号 JSON 凭据以 Vertex AI 模式调用 Gemini，默认主模型为 gemini-3.1-flash-lite；对 Gemini 3 及以后模型按 `thinking_levels` 设置 thinking_level（Flash-Lite 默认 minimal，未配置时 low），对 gemini-2.5 模型设置 thinking_budget=0。
2. WHEN 生成 zh 语料，THE Gemini_Client SHALL 使用中文提示词模板并要求简体中文正文；WHEN 生成 en 语料，THE Gemini_Client SHALL 使用英文提示词模板并要求英文正文。
3. THE Gemini_Client SHALL 在提示词中要求 3–6 个自然段、只写公认事实、不编造人名/数据/引语/文献、不使用 Markdown 标题/列表/表格，并按长度桶目标换算目标长度：zh 目标字数落在 target_len × [1.0, 1.3] 内，en 目标词数为 target_len × 0.75；入库判定以实测 token 数为准。
4. WHERE TopicSpec 属于 Grounding_Category，THE Gemini_Client SHALL 启用 Google Search Grounding，并要求正文仅依据检索到的报道撰写且注明事件日期。
5. THE Gemini_Client SHALL 每次 Grounding 尝试在 Cost_Guard 中预留 `grounding_reserve_queries`（默认 3）次搜索查询，成功后按响应 `grounding_metadata.web_search_queries` 去重后的实际查询数结算（无该字段时按 1 次），超时/5xx 按预留数结算；WHEN 当前 UTC 日内已计费的 Grounding 查询数达到 `grounding_daily_limit`（默认 600），THE Supply_Planner SHALL 从采样中移除 Grounding_Category 直到下一个 UTC 00:00，且不以非 Grounding 方式生成这些类别。
6. WHEN 一次生成成功，THE Gemini_Client SHALL 返回正文、实际使用的模型名、输入/输出 token 数、finish_reason、是否 grounded、来源 URL 列表与尝试次数。
7. WHEN 用户执行 `loadtest warmup`，THE Load_Tester SHALL 以配置的并发（默认 32）生成语料，并在以下任一条件满足时结束：(a) 相对 `--target`（默认 8,000）的缺口全部填满；(b) Gemini Cost_Guard 拒绝预留；(c) 本次 warmup 的花费达到 `--max-cost`（默认 $20）；结束后输出按语言/类别/长度桶统计的库存、拒绝率与花费报告。

### Requirement 4: 时效语料过期

**User Story:** 作为测试设计者，我希望新闻类语料保持时效，以便不使用过时内容。

#### Acceptance Criteria

1. WHEN 一个段落入库，THE Corpus_Store SHALL 在该段落为 grounded 时将 `expires_at` 设为 `created_at` 加 72 小时，在该段落为非 grounded 时将 `expires_at` 设为 NULL。
2. WHEN Corpus_Store 构建 Passage_Index，THE Corpus_Store SHALL 排除 `expires_at` 小于等于当前时间的段落。
3. WHILE 压测运行中，THE Scheduler SHALL 每 60 秒执行一次过期检查，使任一满足 `expires_at` ≤ t 的段落在时刻 t + 60 秒之前被移出 Passage_Index。
4. WHEN Passage_Index 执行段落选取，THE Passage_Index SHALL 仅返回 `expires_at` 为空或大于当前时间的段落。
5. WHEN `warmup` 或 `generate` 命令开始，THE Corpus_Store SHALL 将所有已过期段落的状态置为 `expired`。
6. WHEN 用户执行 `loadtest generate --category <类别列表> --count N`，THE Load_Tester SHALL 仅生成所列类别的语料，使本次入库段落数小于等于 N，且 Grounding_Category 的生成受 `grounding_daily_limit` 约束。

### Requirement 5: 安全过滤与质量校验

**User Story:** 作为测试设计者，我希望入库语料干净合规，以便不触发 Azure 内容风控并保持质量。

#### Acceptance Criteria

1. IF 生成结果的 finish_reason 为 SAFETY 或正文为空，THEN THE Safety_Filter SHALL 丢弃该结果、不重试、以原因 `safety_block` 写入 rejections，并将该子话题的采样权重乘以 0.5。
2. IF 生成结果的 finish_reason 既不是 STOP 也不是 SAFETY，THEN THE Safety_Filter SHALL 以原因 `structure` 拒绝该结果。
3. IF 正文 token 长度（o200k_base）小于 500 或大于 1500，THEN THE Safety_Filter SHALL 以原因 `length` 拒绝该结果。
4. WHEN 正文通过长度校验，THE Safety_Filter SHALL 将正文按 token 长度归入 S、M 或 L 长度桶。
5. IF zh 规格的正文中 CJK 字符占非空白字符的比例低于 60%，或 en 规格的正文中 ASCII 字母 [A-Za-z] 占非空白字符的比例低于 80%，或 en 规格的正文中 CJK 字符占非空白字符的比例不低于 2%，THEN THE Safety_Filter SHALL 以原因 `language` 拒绝该结果。
6. IF 正文匹配拒答或免责模板，THEN THE Safety_Filter SHALL 以原因 `refusal` 拒绝该结果。
7. IF 正文包含 `blocklist.txt` 中的任一中英关键词，THEN THE Safety_Filter SHALL 以原因 `blocklist` 拒绝该结果。
8. IF 正文的段落（以换行分隔的非空文本块）少于 3 个、含 Markdown 标题堆砌（即存在以 `#` 开头的行），或含重复段落（两个段落经空白与标点规范化后完全相同），THEN THE Safety_Filter SHALL 以原因 `structure` 拒绝该结果。
9. IF grounded 规格的生成结果没有至少 1 个来源，或正文不含至少 1 个"年-月-日"或"月-日"形式的日期，THEN THE Safety_Filter SHALL 以原因 `grounding` 拒绝该结果。
10. THE Corpus_Store SHALL 在 rejections 表中只记录类别、语言、模型、原因与时间，不记录正文。
11. WHEN 某子话题在至少 20 个样本中拒绝率超过 30%，THE Supply_Planner SHALL 将该子话题的采样权重降低 50%。
12. THE Safety_Filter SHALL 按 finish_reason → length → language → refusal → blocklist → structure → grounding → duplicate 的顺序执行校验，在第一个失败的校验处停止，并只记录该首个失败原因。

### Requirement 6: 语料去重

**User Story:** 作为测试设计者，我希望语料池中没有重复或近似重复的段落，以便请求多样化。

#### Acceptance Criteria

1. WHEN 一段文本被插入 Corpus_Store，THE Deduplicator SHALL 对该文本此后的重复检查返回"重复"。
2. WHEN 待入库文本删除全部空白字符与 Unicode 标点后的 SHA256 与 Corpus_Store 中任一状态（active、quarantined 或 expired）段落的对应值相同，THE Deduplicator SHALL 判定为重复。
3. WHEN 待入库文本的 64 位 SimHash 与 Corpus_Store 中任一状态（active、quarantined 或 expired）段落的汉明距离小于等于 3，THE Deduplicator SHALL 判定为重复。
4. IF 待入库文本被判定为重复，THEN THE Corpus_Store SHALL 不插入该文本并以原因 `duplicate` 写入 rejections。

### Requirement 7: 语料供应规划

**User Story:** 作为测试设计者，我希望按缺口驱动地生成语料，以便语料池在语言、类别与长度上保持均衡。

#### Acceptance Criteria

1. WHEN Supply_Planner 计算缺口，THE Supply_Planner SHALL 输出 (语言, 类别, 长度桶, 数量) 列表，且数量之和等于 max(0, 目标规模 − 当前 active 且未过期的段落总量)。
2. THE Supply_Planner SHALL 以 目标规模 × lang_ratio[语言] 作为各语言目标，以 语言目标 × 该类别在 Topic_Taxonomy 中的权重 作为类别目标，以 S/M/L = 30% / 40% / 30% 作为长度桶目标，并对取整结果做调整，使每一级子目标之和等于其上级目标。
3. WHEN Supply_Planner 输出下一条生成任务，THE Supply_Planner SHALL 按以下优先级选择：隔离替换、过期替换、语言缺口、类别缺口、长度桶缺口；同级内类别采样权重与库存缺口成正比。
4. THE Load_Tester SHALL 将 `min_runnable` 默认设为 2,000、`healthy_size` 默认设为 8,000、`target_size` 默认设为 15,000。
5. WHEN 用户执行 `loadtest pool-stats`，THE CLI SHALL 输出 active / quarantined / expired 数量，按语言、长度桶、use_count、类别的分布，近 7 天拒绝原因占比，分模型 Gemini 花费，以及按 duration = 1200 秒计算的 Supply_Check 判定与各语言所需库存和可用库存。
6. WHEN `warmup` 或 `generate` 计算出的缺口总数为 0，THE Load_Tester SHALL 不发起任何 Gemini 调用并输出当前库存。

### Requirement 8: 运行前供应检查

**User Story:** 作为运维人员，我希望在压测前确认语料足够，以便运行不会因语料耗尽而中途失败。

#### Acceptance Criteria

1. WHEN 用户执行 `loadtest run`，THE Supply_Planner SHALL 以 预期请求数 = target_tpm × duration_s / 60 / 3040 计算预期请求数，并以 ceil(预期请求数 × 2.5 × lang_ratio[语言] × (1 + supply_margin) / per_run_max_uses) 计算每种语言的所需库存。
2. IF active 未过期段落总数小于 `min_runnable`，或任一语言库存小于该语言所需库存，THEN THE Supply_Check SHALL 返回 refuse。
3. WHEN Supply_Check 返回 refuse 且未指定 `--allow-low-supply`，THE Run_Controller SHALL 发送 0 个 Azure 请求（含 Probe_Request），以 stop_reason = `supply_refused` 结束并以非零退出码退出，同时输出各语言缺口与建议的 warmup 命令。
4. WHEN Supply_Check 未返回 refuse 且（总量小于 `healthy_size`、任一语言占比偏离目标超过 5 个百分点，或任一语言任一长度桶占该语言库存低于 15%），THE Supply_Check SHALL 返回 warn 并列出每条告警。
5. WHEN Supply_Check 未返回 refuse 或 warn，THE Supply_Check SHALL 返回 ok。
6. WHEN 指定 `--allow-low-supply` 且 Supply_Check 返回 refuse，THE Run_Controller SHALL 启动运行并输出可能以 `corpus_exhausted` 提前结束的告警。
7. WHEN Supply_Check 返回 warn，THE Run_Controller SHALL 启动运行，并输出每条告警与建议的 warmup 命令。

### Requirement 9: 段落复用策略

**User Story:** 作为测试设计者，我希望控制段落在单次运行中的复用次数，以便请求内容足够多样。

#### Acceptance Criteria

1. WHILE 单次运行进行中，THE Passage_Index SHALL 只选取 Run_Uses 小于 `per_run_max_uses`（默认 4）的段落。
2. WHEN Passage_Index 以目标长度 near 选取段落，THE Passage_Index SHALL 先在区间包含 near（无包含时取区间与 near 距离最近）的长度桶内，从排除本请求已选段落后的候选中按权重 1 / (1 + Use_Count + Run_Uses)^α（α = `use_weight_alpha`，默认 1.0）加权抽样；目标长度桶无候选时按与 near 的距离由近到远依次放宽到其他长度桶；全部长度桶无候选时返回空。
3. IF 请求语言下没有满足复用上限的候选段落，THEN THE Request_Assembler SHALL 抛出 CorpusExhausted，且不突破复用上限。
4. WHEN 运行结束（任意 stop_reason），THE Corpus_Store SHALL 将本次运行各段落的 Run_Uses 恰好一次累加到对应段落的 Use_Count。
5. WHEN Request_Assembler 将一个段落（含补足用截断片段的来源段落）用于一个 Planned_Request，THE Passage_Index SHALL 将该段落的 Run_Uses 加 1。

### Requirement 10: 后台语料补充

**User Story:** 作为运维人员，我希望在压测期间后台补充语料，同时补充过程不影响 Azure 侧调度。

#### Acceptance Criteria

1. WHILE 压测运行中且未指定 `--no-replenish`，THE Replenisher SHALL 作为独立异步任务运行，使用独立的 Gemini_Rate_Limiter（不超过 30 RPM）、并发不超过 4、使用独立数据库连接。
2. WHEN Replenisher 成功插入新段落，THE Replenisher SHALL 以非阻塞方式把该段落放入容量为 `replenish_queue_size`（默认 1024）的队列；IF 队列已满，THEN THE Replenisher SHALL 丢弃该通知并保留已入库的段落。
3. WHILE 压测运行中，THE Scheduler SHALL 每次循环以非阻塞方式从队列读取最多 64 个新段落加入 Passage_Index。
4. IF Replenisher 内部发生任何异常，THEN THE Replenisher SHALL 在任务内捕获并记录该异常，5 秒后继续，且该异常不终止压测运行。
5. WHILE Replenisher 处于任意延迟、挂起或异常状态且未向队列交付任何段落，THE Scheduler SHALL 产生与关闭 Replenisher 时相同的派发序列（相同随机种子与相同初始 Passage_Index），且不等待 Replenisher 的任何操作。
6. IF Gemini Cost_Guard 拒绝预留，THEN THE Replenisher SHALL 在本次运行内不再发起 Gemini 调用并记录 1 条告警，且 Scheduler 继续派发。
7. WHEN 存在被隔离或过期且尚未替换的段落，THE Replenisher SHALL 优先按这些段落的类别、语言与长度桶生成替代段落，再按缺口补充。
8. WHEN 优雅停止开始，THE Replenisher SHALL 不再发起新的 Gemini 调用。

### Requirement 11: Gemini 限流、节流、重试与超时

**User Story:** 作为运维人员，我希望 Gemini 调用稳定且不超配额，以便语料供应不中断。

#### Acceptance Criteria

1. THE Gemini_Rate_Limiter SHALL 以 RPM 速率 = quota_rpm（默认 600）× usage_ratio（默认 0.5）× Gemini_Throttle 当前档位、TPM 速率 = quota_tpm（默认 800,000）× usage_ratio × Gemini_Throttle 当前档位放行调用，使任意区间 [t0, t1] 内放行的调用数与 token 数分别小于等于 对应容量 + 对应速率 × (t1 − t0)（与 Requirement 17.2 相同的上界形式）。
2. WHEN Gemini 返回 429 或 RESOURCE_EXHAUSTED，THE Gemini_Throttle SHALL 以 Level_Mode 处理：立即将档位设为 10% 并进入 HOLD，将 hold_until 设为 max(现有 hold_until, 当前时间 + min(max(Retry-After, 10 秒), 120 秒))；HOLD 结束后不发送 Throttle_Probe，直接进入 RECOVER，并在无新的 429 或 RESOURCE_EXHAUSTED 时按 10% → 50% → 100% 每 10 秒升一级（recover_full_s = 20 秒）。
3. IF Gemini 调用返回 429、RESOURCE_EXHAUSTED、5xx 或单次尝试超过 60 秒超时，THEN THE Gemini_Client SHALL 以退避重试；IF 调用返回其他 4xx 或 finish_reason 为 SAFETY，THEN THE Gemini_Client SHALL 不重试；WHEN 单次生成任务的 `max_retries` + 1（默认 4）次尝试均失败，THE Gemini_Client SHALL 放弃该任务、返回空结果，并向 Model_Chain 报告 1 次失败。
4. THE Gemini_Client SHALL 将第 n 次（n 从 0 开始）退避时长取为 min(backoff_cap_s, max(Retry-After 或 0, Uniform(0, min(backoff_cap_s, backoff_base_s × 2^n)))) 秒（默认 backoff_base_s = 2、backoff_cap_s = 30），使结果在 [0, 30] 内，且存在 Retry-After 时结果大于等于 min(Retry-After, 30)。
5. WHEN 每次 Gemini 尝试（含重试与备用模型）开始，THE Gemini_Client SHALL 在 Gemini Cost_Guard 中预留该次尝试的最大花费；WHEN 尝试超时或返回 5xx，THE Gemini_Client SHALL 按预留金额结算；WHEN 尝试成功，THE Gemini_Client SHALL 按实际 usage 与实际使用的模型结算。
6. IF Gemini Cost_Guard 拒绝预留，THEN THE Gemini_Client SHALL 不发起该次调用并报告 Gemini 预算耗尽。
7. WHILE Gemini_Throttle 处于 Hold，THE Gemini_Client SHALL 不发起任何 Gemini 尝试。

### Requirement 12: Gemini 备用模型链

**User Story:** 作为运维人员，我希望主模型故障时自动切换备用模型，以便语料生成持续进行。

#### Acceptance Criteria

1. THE Model_Chain SHALL 默认按 gemini-3.1-flash-lite → gemini-3.5-flash-lite → gemini-3.8-flash 排序，且当前模型始终属于该列表。
2. WHEN 当前模型在重试耗尽后的连续失败次数达到 `failover_after_failures`（默认 5），THE Model_Chain SHALL 切换到列表中的下一个模型并记录告警；IF 当前模型已是列表中最后一个可选模型，THEN THE Model_Chain SHALL 保持当前模型并记录告警。
3. WHEN 某模型的一次调用成功，THE Model_Chain SHALL 将该模型的连续失败计数清零。
4. WHEN 切换到备用模型后经过 `failback_after_s`（默认 600 秒），THE Model_Chain SHALL 将下一次尝试发往主模型；WHEN 主模型连续 3 次调用成功，THE Model_Chain SHALL 恢复主模型为当前模型；IF 主模型探测失败，THEN THE Model_Chain SHALL 回到备用模型并重新开始 `failback_after_s` 计时。
5. WHILE Gemini 预算剩余比例 (cap − spent − reserved) / cap 小于 20%，THE Model_Chain SHALL 不选择 `expensive_models` 中的模型（默认 gemini-3.8-flash）；当切换目标为高价模型时保持当前模型。
6. WHEN 段落入库，THE Corpus_Store SHALL 在 `gen_model` 字段记录实际生成该段落的模型名。

### Requirement 13: 输入长度分布

**User Story:** 作为测试设计者，我希望请求输入长度符合指定分布，以便 token 消耗集中在输入端。

#### Acceptance Criteria

1. WHEN Length_Distribution 采样目标输入长度，THE Length_Distribution SHALL 返回 1500 到 6000（含）之间的整数。
2. THE Length_Distribution SHALL 以 80% 的概率从截断到 [2500, 3500] 的正态分布 N(μ = 3000, σ = (core_high − core_low) / 3.33 ≈ 300.3) 采样并四舍五入取整，使对任意随机种子采样 20,000 次时落在 [2500, 3500] 的比例在 [0.78, 0.82] 内。
3. THE Length_Distribution SHALL 以 10% 的概率从 [1500, 2500) 均匀采样，以 10% 的概率从 (3500, 6000] 均匀采样，使采样 20,000 次时两个尾部区间各自的比例在 [0.09, 0.11] 内。

### Requirement 14: 请求组装

**User Story:** 作为测试设计者，我希望每个请求由多段语料与任务指令精确组装，以便输入长度可控、内容多样、输出短小。

#### Acceptance Criteria

1. WHEN Request_Assembler 以目标长度 target（1500–6000）组装请求，THE Request_Assembler SHALL 使 Planned_Request 的 input_tokens（以 tiktoken o200k_base 对完整消息复算）与 target 之差的绝对值小于等于 20。
2. THE Request_Assembler SHALL 使每个 Planned_Request 包含 1–4 个互不重复的段落，且全部段落的语言等于该请求的语言。
3. THE Request_Assembler SHALL 仅使用状态为 active、未过期、未隔离的段落。
4. THE Request_Assembler SHALL 从 Task_Catalog 中与请求语言相同的指令子集里均匀抽取任务指令，随机打乱段落顺序，并随机将指令置于正文之前或之后，使两种指令位置均出现。
5. THE Request_Assembler SHALL 将每个 Planned_Request 的 max_tokens 设为 40。
6. THE Request_Assembler SHALL 在中文指令后附加"只输出答案，不要解释，不超过 30 字。"，在英文指令后附加 "Answer only, no explanation, under 20 words."。
7. THE Task_Catalog SHALL 提供中文至少 30 条、英文至少 30 条任务指令。
8. WHEN 段落总长超出预算，THE Request_Assembler SHALL 在句子边界裁剪最后一段，再按 token 截断到预算；WHEN 段落总长低于预算超过 20 token，THE Request_Assembler SHALL 在段落数少于 4 且存在候选段落时追加一个截断片段，否则重新裁剪最后一段的来源段落以补足，且补足结果满足 Requirement 14.2 与 9.1。
9. THE Request_Assembler SHALL 按段落预算确定初始段落数（不含 Requirement 14.8 的补足片段）：预算小于 2000 token 时为 1–2 段，预算在 [2000, 4000) 时为 2–3 段，其余为 3–4 段。

### Requirement 15: 语言配比

**User Story:** 作为测试设计者，我希望请求语言按 zh 60% / en 40% 分布，以便贴近目标业务。

#### Acceptance Criteria

1. WHEN Request_Assembler 选择请求语言，THE Request_Assembler SHALL 以 `lang_ratio["zh"]`（默认 0.60）的概率选择 zh，否则选择 en，使组装 10,000 个请求时 zh 占比在 [0.58, 0.62] 内。
2. THE Request_Assembler SHALL 只生成语言为 zh 或 en 的请求，不生成中英混合请求。
3. WHEN 运行结束，THE Metrics_Reporter SHALL 在报告中输出 zh 与 en 请求分别占已派发请求数的比例，以保留 1 位小数的百分比表示。

### Requirement 16: 请求调度与目标随机游走

**User Story:** 作为测试设计者，我希望请求时间随机分布并在目标附近波动，以便模拟真实业务流量。

#### Acceptance Criteria

1. WHILE 压测运行中，THE Scheduler SHALL 以指数分布的间隔（泊松到达）尝试派发请求，到达率 λ = min(有效 TPM / 平均请求 token 的 EWMA, 有效 RPM) / 60，其中有效 TPM = 目标 TPM × Effective_Level，EWMA 为已派发请求 (input_tokens + max_tokens) 的指数加权平均（平滑系数 `ewma_alpha` = 0.05，初值 3040）。
2. WHEN 自主循环开始起第 60 秒及此后每经过 60 秒，THE Scheduler SHALL 将目标 TPM 加上 N(0, 30,000²) 的随机步长并截断到 [tpm_walk_low, tpm_walk_high]（默认 [700,000, 850,000]），使目标 TPM 始终位于该区间内，初始目标为 800,000。
3. THE Scheduler SHALL 将有效 RPM 限制为 min(rpm_max, quota_rpm × 0.5) × Effective_Level，其中 rpm_max 默认为 500。
4. THE Scheduler SHALL 在派发请求前依次完成：组装、Cost_Guard 预留、Dual_Token_Bucket 获取（成本 = input_tokens + max_tokens）、并发信号量获取（上限 `max_concurrency`，默认 64）。
5. WHEN 运行结束，THE Metrics_Reporter SHALL 报告实际平均 TPM = Σ(已派发请求的 input_tokens + max_tokens) / 派发时长（分钟）与实际平均 RPM = 已派发请求数 / 派发时长（分钟），其中派发时长为主循环开始到停止派发的时间，以便确认 TPM 约 800k、RPM 在 200–500 范围内。
6. WHEN Effective_Level 或目标 TPM 发生变化，THE Scheduler SHALL 在下一次派发前更新 Dual_Token_Bucket 的速率。
7. WHEN 用户以 `--tpm T` 启动运行，THE Scheduler SHALL 以 T 作为初始目标 TPM，并以 [T × 0.875, T × 1.0625] 作为随机游走区间。
8. IF T × 1.0625 × (1 + 5/60) > 0.95 × quota_tpm（默认配额下即 T 大于约 825,000），或 `--rpm-max` × (1 + 5/60) > 0.95 × quota_rpm（默认配额下即大于约 876），THEN THE CLI SHALL 拒绝启动运行并以非零退出码退出。

### Requirement 17: 全局双令牌桶限流

**User Story:** 作为运维人员，我希望所有请求共享一个全局限流器，以便证明任何时间窗口内都不超过配额。

#### Acceptance Criteria

1. THE Dual_Token_Bucket SHALL 以单一全局实例为所有 worker 放行请求，每次放行同时扣除 cost 个 token 与 1 个请求额度。
2. THE Dual_Token_Bucket SHALL 使任意区间 [t0, t1] 内放行的 token 总量小于等于 tok_cap + tok_rate × (t1 − t0)，放行请求数小于等于 req_cap + req_rate × (t1 − t0)，其中 tok_cap = max(TPM × 5 / 60, 6,040)、req_cap = max(RPM × 5 / 60, 1)。
3. WHILE 压测运行中，THE Load_Tester SHALL 使任一 60 秒滑动窗口内放行的 Σ(input_tokens + max_tokens) 小于等于 tpm_walk_high × (1 + 5/60)（默认约 920,800，不超过配额的 95%），且放行请求数小于等于 rpm_max × (1 + 5/60)。
4. WHEN Dual_Token_Bucket 的速率被下调，THE Dual_Token_Bucket SHALL 立即将存量截断到新容量，使此后任一区间的放行量小于等于新容量 + 新速率 × 时长。
5. WHEN Dual_Token_Bucket 初始化，THE Dual_Token_Bucket SHALL 将 token 初始存量设为 `bucket_initial_fill_s`（默认 2 秒）× tok_rate，将请求初始存量设为 max(1, bucket_initial_fill_s × req_rate)。
6. WHILE 存量不足以放行等待中的请求，THE Dual_Token_Bucket SHALL 按 FIFO 顺序放行等待者，并以不超过 0.25 秒的间隔重新检查存量。

### Requirement 18: 启动爬坡

**User Story:** 作为运维人员，我希望压测启动时快速但不突发地到达目标负载。

#### Acceptance Criteria

1. WHEN 压测启动且未指定 `--no-ramp`，THE Startup_Ramp SHALL 在已运行时间位于 [0, 10) 秒时返回档位 50%，在已运行时间大于等于 10 秒时返回档位 100%。
2. WHERE 指定了 `--no-ramp`，THE Startup_Ramp SHALL 自启动起返回档位 100%。
3. THE Startup_Ramp SHALL 返回 (0, 1.0] 内且关于已运行时间单调不减的档位，且已运行时间大于等于最后一步时间时返回 1.0。
4. WHILE Throttle_Controller 处于 NORMAL 或 RECOVER，THE Scheduler SHALL 使用 Effective_Level = min(Startup_Ramp 档位, Throttle_Controller 档位)，该值始终在 (0, 1.0] 内且属于两者档位取值集合的并集；WHILE Throttle_Controller 处于 PROBE，THE Scheduler SHALL 以 levels[0] 作为 Throttle_Probe 的档位。
5. THE Startup_Ramp SHALL 以 Scheduler 主循环开始时刻（供应检查与 Probe_Request 完成之后）作为已运行时间的零点。

### Requirement 19: 节流事件立即暂停（HOLD 与 PROBE）

**User Story:** 作为运维人员，我希望收到 429 或其他配额超限反馈时系统立即暂停派发、冷却后只用一个请求确认配额已恢复，以便遵守 Azure 配额且不在恢复期反复撞限。

#### Acceptance Criteria

1. WHEN Azure_Client 收到非 Stale_Throttle 的 Throttle_Event（包括 Startup_Ramp 期间，以及 Throttle_Controller 处于 NORMAL 或 RECOVER 时），THE Throttle_Controller SHALL 在处理该响应时同步进入 HOLD 状态，使此后 allowed_dispatch 返回 (0, 0)。
2. WHEN Throttle_Controller 因 Throttle_Event 进入或停留在 HOLD，THE Throttle_Controller SHALL 将 hold_until 设为 max(现有 hold_until, 当前时间 + min(max(Retry-After, 10 秒), 120 秒))，使 hold_until 只增不减。
3. WHEN 解析 Retry-After，THE Azure_Client SHALL 依次使用 `retry-after-ms`（换算为秒）与 `retry-after`（秒数或 HTTP 日期），将无法解析或为负值的头视为缺失并继续尝试下一个头，将早于当前时间的 HTTP 日期取为 0 秒，在两者都缺失时使用 10 秒，并在值超过 120 秒时截断为 120 秒且记录告警。
4. WHILE Throttle_Controller 处于 HOLD（区间 [进入 HOLD 时刻, hold_until)），THE Scheduler SHALL 派发 0 个新请求。
5. WHEN 一个请求（含 Throttle_Probe）的 Outcome 为 THROTTLED，THE Azure_Client SHALL 不重试该请求，THE Metrics_Reporter SHALL 将其计入节流计数，THE Cost_Guard SHALL 将其花费结算为 0。
6. IF Azure_Client 收到 Stale_Throttle，THEN THE Throttle_Controller SHALL 将其计入 stale_throttles，不改变当前状态与恢复进度，并仅在处于 HOLD 时将 hold_until 设为 max(现有 hold_until, 当前时间 + min(max(Retry-After, 10 秒), 120 秒))，在处于 PROBE 或 RECOVER 时忽略该事件。
7. WHILE Throttle_Controller 处于 PROBE（hold_until 到达后进入），THE Scheduler SHALL 在在途请求清空或自进入 PROBE 起已满 `probe_inflight_wait_s`（默认 30 秒）之后，派发恰好 1 个经 Cost_Guard 与 Dual_Token_Bucket 的 Throttle_Probe，且在该 Throttle_Probe 结果返回前派发 0 个其他新请求。

### Requirement 20: 节流探测结果与阶梯恢复

**User Story:** 作为运维人员，我希望探测确认配额恢复后以约 60 秒阶梯回满负载，以便压测稳定地回到目标水平。

#### Acceptance Criteria

1. WHEN Throttle_Probe 的 Outcome 为 OK，THE Throttle_Controller SHALL 进入 RECOVER、将档位设为 levels[0]（10%）并以该时刻作为升档计时起点；在 Probe_Mode 下，OK 的 Throttle_Probe 是 Throttle_Controller 从 HOLD 或 PROBE 进入 RECOVER 的唯一途径。
2. WHEN Throttle_Probe 的 Outcome 为 THROTTLED，THE Throttle_Controller SHALL 回到 HOLD，按 Requirement 19.2 以该响应的 Retry-After 更新 hold_until，将 consecutive_probe_fail 加 1，并在 hold_until 到达后再次进入 PROBE，探测次数不设上限。
3. IF Throttle_Probe 的 Outcome 为 SERVER_ERROR、TIMEOUT 或 CLIENT_ERROR，THEN THE Throttle_Controller SHALL 保持 PROBE 并使下一个 Throttle_Probe 不早于 `probe_error_retry_s`（默认 5 秒）之后派发，且 THE Circuit_Breaker SHALL 将该结果计为失败样本；IF Throttle_Probe 的 Outcome 为 CONTENT_FILTER，THEN THE Throttle_Controller SHALL 保持 PROBE、不进入 RECOVER，并立即允许派发新组装的下一个 Throttle_Probe（涉及段落按 Requirement 24 隔离）。
4. WHILE Throttle_Controller 处于 RECOVER，THE Throttle_Controller SHALL 每次只将档位升高到 levels 中相邻的下一级（默认 10% → 25% → 50% → 75% → 100%），且仅在自 max(最近节流时间, 最近主动减速时间, 最近升档时间, 最近 Throttle_Probe 成功时间) 起已经过 recover_step_s（默认 15 秒）时升档。
5. WHILE 无新的 Throttle_Event 或主动减速事件，THE Throttle_Controller SHALL 使从 Throttle_Probe 成功到档位达到 100% 的时长大于等于 `recover_full_s`（默认 60 秒）且小于等于 recover_full_s + 1 秒，并在档位达到 100% 时进入 NORMAL。
6. WHILE 压测运行中（含 HOLD、PROBE、熔断 open、内容过滤暂停期间与泊松到达间隔内），THE Scheduler SHALL 以不超过 1 秒的间隔调用 Throttle_Controller 的 tick。
7. WHEN Throttle_Controller 处于 RECOVER 时收到非 Stale_Throttle 的 Throttle_Event，THE Throttle_Controller SHALL 回到 HOLD，并按 HOLD → PROBE → RECOVER 的顺序重新开始恢复。

### Requirement 21: 基于响应头的主动减速

**User Story:** 作为运维人员，我希望在配额接近耗尽时提前减速，以便减少 429 的发生。

#### Acceptance Criteria

1. WHILE Throttle_Controller 处于 NORMAL 或 RECOVER，WHEN 成功响应的 `x-ratelimit-remaining-tokens` / `x-ratelimit-limit-tokens` 或 `x-ratelimit-remaining-requests` / `x-ratelimit-limit-requests` 低于 `soft_down_ratio`（默认 10%），THE Throttle_Controller SHALL 将档位降低到 levels 中相邻的下一级（默认 100% → 75% → 50%），并在原状态为 NORMAL 时进入 RECOVER。
2. THE Throttle_Controller SHALL 使主动减速后的档位大于等于 `soft_down_floor`（默认 50%），在当前档位的下一级低于 soft_down_floor 时保持当前档位，且主动减速不使状态进入 HOLD 或 PROBE。
3. THE Throttle_Controller SHALL 自最近一次满足主动减速条件的时刻起 5 秒内不再执行主动减速，无论该次是否实际降档。
4. IF 某一维度的 remaining 头或 limit 头缺失，THEN THE Throttle_Controller SHALL 不对该维度进行主动减速判定。

### Requirement 22: Azure 错误分类与启动探测

**User Story:** 作为运维人员，我希望每个 Azure 响应被明确分类，以便系统做出正确的处理。

#### Acceptance Criteria

1. WHEN Azure_Client 得到响应或异常，THE Azure_Client SHALL 为每个 (状态码, 响应体, 异常) 组合返回唯一的 Outcome。
2. THE Azure_Client SHALL 将 429、503 与响应体表明 quota 或 rate limit exceeded 的 403 分类为 THROTTLED，将其他 403 分类为 CLIENT_ERROR；将 400 且 code = content_filter，或 200 且 finish_reason = content_filter 分类为 CONTENT_FILTER；将 finish_reason 不为 content_filter 的 200 分类为 OK；将 500/502/504 分类为 SERVER_ERROR；将 30 秒内未收到完整响应分类为 TIMEOUT；将其他 4xx 分类为 CLIENT_ERROR。
3. THE Azure_Client SHALL 以 SDK 重试次数为 0 的方式发送请求，并读取 `x-ratelimit-remaining-tokens`、`x-ratelimit-remaining-requests`、`x-ratelimit-limit-tokens`、`x-ratelimit-limit-requests`、`retry-after`、`retry-after-ms` 响应头。
4. WHEN `loadtest run` 在供应检查之后、压测开始之前，THE Run_Controller SHALL 发送 1 个 Probe_Request；IF Probe_Request 返回 401 或 404，THEN THE Run_Controller SHALL 以非零退出码终止并提示配置错误。
5. IF 请求因没有状态码的网络异常（如连接失败）结束，THEN THE Azure_Client SHALL 将其分类为 SERVER_ERROR。
6. THE Run_Controller SHALL 通过 Azure Cost_Guard 预留并结算 Probe_Request 的花费。
7. WHERE 指定了 `--dry-run`，THE Run_Controller SHALL 不发送 Probe_Request。
8. IF Probe_Request 的 Outcome 为 THROTTLED，THEN THE Run_Controller SHALL 启动运行，且 Throttle_Controller 在主循环开始前已按该 Probe_Request 的 Retry-After 进入 HOLD，使主循环从 HOLD → PROBE 开始。
9. IF Probe_Request 的 Outcome 为 SERVER_ERROR 或 TIMEOUT，THEN THE Run_Controller SHALL 以 5 秒间隔重试 Probe_Request 最多 3 次；IF 重试均失败，THEN THE Run_Controller SHALL 以非零退出码终止。

### Requirement 23: 熔断器

**User Story:** 作为运维人员，我希望在 Azure 持续出错时暂停压测，以便避免无效消耗。

#### Acceptance Criteria

1. THE Circuit_Breaker SHALL 只以 Outcome 为 OK 或计入失败（SERVER_ERROR、TIMEOUT、CLIENT_ERROR）的结果（含 Throttle_Probe）作为样本；THROTTLED 与 CONTENT_FILTER 既不计入错误率的分子与分母，也不重置连续失败计数；仅 OK 重置连续失败计数。
2. WHEN 60 秒窗口内样本数不少于 20 且错误率超过 50%，或连续失败达到 10 次，THE Circuit_Breaker SHALL 进入 open 状态 60 秒。
3. WHILE Circuit_Breaker 处于 open 状态，THE Scheduler SHALL 不派发新请求。
4. WHEN open 持续 60 秒后，THE Circuit_Breaker SHALL 进入 half_open 状态；WHEN half_open 下 5 个探测请求成功，THE Circuit_Breaker SHALL 回到 closed；IF half_open 下有请求失败，THEN THE Circuit_Breaker SHALL 回到 open。
5. WHEN 单次运行中 Circuit_Breaker 第 3 次进入 open，THE Run_Controller SHALL 立即以 stop_reason = `circuit_open` 停止运行。
6. WHILE Circuit_Breaker 处于 half_open 状态，THE Scheduler SHALL 最多派发 5 个探测请求。

### Requirement 24: 内容过滤隔离

**User Story:** 作为测试设计者，我希望触发 Azure 内容过滤的段落被立即隔离，以便后续请求不再触发风控。

#### Acceptance Criteria

1. WHEN 一个请求的 Outcome 为 CONTENT_FILTER，THE Corpus_Store SHALL 将该请求涉及的全部段落状态置为 `quarantined`，并在 quarantine 表记录原因 `azure_content_filter` 与过滤类别（hate、sexual、violence、self_harm、jailbreak 中的命中项；无法获得时为空）。
2. WHEN 段落被隔离，THE Passage_Index SHALL 立即移除该段落，使同一运行及之后所有运行的组装结果都不包含该段落。
3. IF 任一 60 秒滑动窗口内 CONTENT_FILTER 次数超过 5 次，THEN THE Scheduler SHALL 在随后 30 秒内不派发新请求，并输出包含该窗口内命中次数的告警。
4. WHEN 一个请求的 Outcome 为 CONTENT_FILTER，THE Azure_Client SHALL 不重试该请求，THE Metrics_Reporter SHALL 将其计入内容过滤计数。

### Requirement 25: 成本守卫

**User Story:** 作为账号所有者，我希望花费不会超过预算，以便控制成本。

#### Acceptance Criteria

1. THE Cost_Guard SHALL 将 Azure 预算上限默认设为 $180，将 Gemini 预算上限默认设为每个 GCP 账号 $250（含 grounding 与备用模型）。
2. WHEN Cost_Guard 收到预留请求，THE Cost_Guard SHALL 仅在 已花费 + 在途预留 + 本次最大花费（按输入 token × 1.02 与 max_tokens 计价）小于等于 上限 − safety_margin（默认 $0.5）时放行；WHERE 配置了 `per_run_cap_usd`，THE Cost_Guard SHALL 同时要求 本次运行已花费 + 在途预留 + 本次最大花费 小于等于 per_run_cap_usd。
3. WHEN 请求完成且有 usage，THE Cost_Guard SHALL 按实际 usage 与单价结算，其中缓存输入 token 按 `cached_input_per_m` 计价（未配置时按 `input_per_m` 计价）；WHEN 请求超时或无 usage，THE Cost_Guard SHALL 按预留金额结算。
4. WHILE 每个请求的实际 usage 小于等于其预留量，THE Cost_Guard SHALL 使累计花费小于等于上限。
5. IF Azure Cost_Guard 拒绝预留，THEN THE Scheduler SHALL 不再派发任何请求，并以 stop_reason = `budget` 停止运行。
6. THE Gemini Cost_Guard SHALL 按实际使用的模型分别计价，使各模型花费之和等于 Gemini 总花费；Grounding 按次计入 provider `gemini_grounding`。
7. WHEN `loadtest run` 启动，THE Cost_Guard SHALL 按 target_tpm、预期 RPM 与 duration 输出本次运行预估花费；IF 已花费 + 预估花费超过上限且未指定 `--allow-partial`，THEN THE Run_Controller SHALL 以非零退出码拒绝启动，且不发送任何 Azure 请求。
8. WHEN 用户执行 `loadtest budget`，THE CLI SHALL 输出 Azure 与 Gemini 分模型的已用金额、上限、单次 20 分钟运行预估，以及剩余可运行次数 = floor((上限 − safety_margin − 已花费) / 单次运行预估)。

### Requirement 26: 持久化账本

**User Story:** 作为账号所有者，我希望花费跨运行累计记录，以便预算控制在多次运行间有效。

#### Acceptance Criteria

1. THE Corpus_Store SHALL 在 ledger 表按 run_id、provider、model、account 记录输入 token、输出 token、计次单位与花费；旧库打开时自动补 account 列（旧行 account = ''）。
2. WHEN 新运行启动，每个 Cost_Guard SHALL 从 ledger 读取自身 provider 的历史累计花费作为已花费初值（Gemini Cost_Guard 包含 `gemini` 与 `gemini_grounding`，且只累计当前账号 = `cost.gemini_budget_account`，未设置时为 `gemini.project`；Azure 不区分账号）。
6. WHEN 用户更换 GCP 赠金账号（`gemini.project` 或 `cost.gemini_budget_account` 改变），THE Gemini Cost_Guard SHALL 从 0 开始计算新账号的花费，`loadtest budget` SHALL 单列其他账号的历史花费且不计入当前上限。
3. THE Load_Tester SHALL 使 ledger_totals 的累计花费等于各次运行 `runs.cost_usd` 之和（误差小于 1e-6）。
4. WHILE 运行进行中，THE Load_Tester SHALL 每 10 秒批量刷写账本与 Use_Count，使进程异常退出时最多丢失最近 10 秒的记录；WHEN 运行结束或收到停止信号，THE Load_Tester SHALL 强制刷盘。
5. IF 账本写入失败，THEN THE Load_Tester SHALL 在首次失败后最多重试 3 次；IF 3 次重试均失败，THEN THE Run_Controller SHALL 以 stop_reason = `error` 停止运行并以非零退出码退出。

### Requirement 27: 运行控制与优雅停止

**User Story:** 作为运维人员，我希望可以手动启动、手动停止或按固定时长运行压测，并在停止时完整收尾。

#### Acceptance Criteria

1. WHEN 用户执行 `loadtest run`，THE Run_Controller SHALL 启动压测并运行 `--duration` 指定的时长（默认 20 分钟），同时写入 `runs/loadtest.pid`。
2. WHEN 收到 SIGINT、SIGTERM、`loadtest stop`（写入 `runs/STOP` 并向 pid 发送 SIGTERM）或 duration 到期，THE Run_Controller SHALL 开始优雅停止并记录 stop_reason：存在 `runs/STOP` 时为 `stop_cmd`，其余 SIGINT/SIGTERM 为 `signal`，duration 到期为 `duration`。
3. WHEN 优雅停止开始，THE Scheduler SHALL 不再派发新请求。
4. WHEN 优雅停止开始，THE Run_Controller SHALL 等待在途请求最多 `drain_timeout_s`（默认 30 秒），并将未完成的请求按预留金额计入账本。
5. WHEN Drain 结束，THE Run_Controller SHALL 将 Run_Uses 累加回 Use_Count、刷写账本、生成报告并删除 `runs/loadtest.pid`。
6. IF 优雅停止期间再次收到 SIGINT，THEN THE Run_Controller SHALL 不再等待在途请求，将未完成请求按预留金额结算，刷写账本后立即退出。
7. WHERE 指定了 `--dry-run`，THE Run_Controller SHALL 只运行调度与组装，不发送 Probe_Request、不调用 Azure、不写入 Azure 账本条目，并输出预期 TPM/RPM、爬坡曲线、语言配比、复用分布与长度分布。
8. IF 运行中 Request_Assembler 抛出 CorpusExhausted，THEN THE Run_Controller SHALL 以 stop_reason = `corpus_exhausted` 优雅停止。
9. IF `runs/loadtest.pid` 存在且其记录的进程存活，THEN THE Run_Controller SHALL 拒绝启动新的 run 并以非零退出码退出，以防止两个压测叠加超出配额。
10. IF 执行 `loadtest stop` 时没有运行中的压测，THEN THE CLI SHALL 输出提示信息并以非零退出码退出。

### Requirement 28: 指标与报告

**User Story:** 作为运维人员，我希望实时看到压测状态并在结束后获得报告，以便评估部署表现。

#### Acceptance Criteria

1. WHILE 压测运行中，THE Metrics_Reporter SHALL 每 10 秒在控制台输出：已运行时间、Throttle_Controller 当前状态（NORMAL / HOLD / PROBE / RECOVER）、有效档位、爬坡状态、Throttle_Event 次数、stale_throttles 次数、Throttle_Probe 计数（发送 / 成功 / 被节流 / 错误 / 内容过滤）、各状态累计时长、目标 TPM、实际 TPM、RPM、在途请求数、各 Outcome 计数、语言占比、输入 P50/P95、平均输出 token、延迟 P50/P95、语料池 active 数与本次新增数、最大 Run_Uses、补充队列长度与花费；其中实际 TPM、RPM、各百分位与平均输出 token 按最近 60 秒窗口计算，各 Outcome 计数为本次运行累计值。
2. THE Metrics_Reporter SHALL 为每个请求写一行 JSONL，内容包含时间戳、请求 id、语言、task_id、段落 id、input/output token、Outcome、状态码、延迟与是否为 Throttle_Probe（`is_probe`），且不包含语料原文与密钥。
3. THE Metrics_Reporter SHALL 每 10 秒写一行 CSV 汇总，包含时间戳与 Requirement 28.1 中的全部数值字段。
4. WHEN 运行结束（任意 stop_reason），THE Metrics_Reporter SHALL 生成 `report.json` 与 `report.md`，包含 stop_reason、请求与 token 统计、实际平均 TPM 与 RPM、ThrottleStats 全部字段（含各状态累计时长）、每次 Throttle_Event 的时间线（进入 HOLD、Throttle_Probe 次数与结果、回到 100% 的时刻）、语言实际配比、段落复用分布、补充入库数与花费。
5. IF 平均输出 token（Σ completion_tokens / OK 请求数）不在 20–40 范围内，THEN THE Metrics_Reporter SHALL 在报告中输出偏离提示。

### Requirement 29: CLI 命令

**User Story:** 作为运维人员，我希望通过一组 CLI 命令完成全部操作。

#### Acceptance Criteria

1. THE CLI SHALL 提供命令 `warmup`、`generate`、`run`、`stop`、`status`、`budget`、`report`、`pool-stats`、`doctor`（上线前以极少量真实调用自检 Azure 与 Gemini 各依赖，任一检查失败时以非零退出码退出）。
2. THE CLI SHALL 为 `run` 提供选项 `--duration`、`--tpm`、`--rpm-max`、`--deployment`、`--no-replenish`、`--no-ramp`、`--allow-low-supply`、`--allow-partial`、`--dry-run`。
3. THE CLI SHALL 为 `warmup` 提供选项 `--target`、`--concurrency`、`--max-cost`，为 `generate` 提供选项 `--count`、`--category`、`--lang`。
4. WHEN 用户执行 `loadtest status`，THE CLI SHALL 依据 `runs/loadtest.pid` 记录的进程是否存活，输出当前是否有运行中的压测及其 run_id 与已运行时间。
5. WHEN 用户执行 `loadtest report [RUN_ID]`，THE CLI SHALL 输出指定运行（缺省为最近一次）的报告；IF RUN_ID 不存在，THEN THE CLI SHALL 输出错误信息并以非零退出码退出。
6. THE Load_Tester SHALL 不暴露任何网络监听端口。
7. IF CLI 选项值不合法，THEN THE CLI SHALL 输出不合法的选项名并以非零退出码退出。
