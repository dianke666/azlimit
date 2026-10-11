-- Corpus / run / ledger schema（SQLite，WAL 模式）。
-- 所有时间为 ISO8601 UTC 字符串，固定格式 "YYYY-MM-DDTHH:MM:SS.ffffff+00:00"，可按字典序比较。
-- 使用 IF NOT EXISTS，使每次打开数据库时执行本脚本都是幂等的。

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;

CREATE TABLE IF NOT EXISTS passages (
    id            INTEGER PRIMARY KEY,
    category      TEXT    NOT NULL,
    subtopic      TEXT    NOT NULL,
    genre         TEXT    NOT NULL,
    lang          TEXT    NOT NULL CHECK (lang IN ('zh','en')),
    token_len     INTEGER NOT NULL CHECK (token_len BETWEEN 500 AND 1500),
    length_bucket TEXT    NOT NULL CHECK (length_bucket IN ('S','M','L')),
    text          TEXT    NOT NULL,
    sha256        TEXT    NOT NULL UNIQUE,
    simhash       INTEGER NOT NULL,          -- 64 位，存为有符号整数
    grounded      INTEGER NOT NULL DEFAULT 0,
    sources       TEXT,                      -- JSON 数组
    gen_model     TEXT    NOT NULL,          -- 实际生成模型（主或备用）
    created_at    TEXT    NOT NULL,          -- ISO8601 UTC
    expires_at    TEXT,                      -- grounded=1 时为 created_at + 72h
    status        TEXT    NOT NULL DEFAULT 'active'
                  CHECK (status IN ('active','quarantined','expired')),
    use_count     INTEGER NOT NULL DEFAULT 0,
    filter_hits   INTEGER NOT NULL DEFAULT 0,
    CHECK (grounded = 0 OR expires_at IS NOT NULL)
);
CREATE INDEX IF NOT EXISTS idx_passages_pick   ON passages(status, lang, length_bucket, use_count);
CREATE INDEX IF NOT EXISTS idx_passages_cat    ON passages(status, lang, category);
CREATE INDEX IF NOT EXISTS idx_passages_expire ON passages(status, expires_at) WHERE expires_at IS NOT NULL;
-- SimHash 4 × 16 位 band 表达式索引（鸽巢原理：汉明距离 ≤ 3 必有一个 band 完全相同）。
-- 查询时表达式须与此处逐字一致才能命中索引（见 CorpusStore._find_duplicate）。
CREATE INDEX IF NOT EXISTS idx_passages_simhash_band  ON passages((simhash & 65535));
CREATE INDEX IF NOT EXISTS idx_passages_simhash_band1 ON passages(((simhash >> 16) & 65535));
CREATE INDEX IF NOT EXISTS idx_passages_simhash_band2 ON passages(((simhash >> 32) & 65535));
CREATE INDEX IF NOT EXISTS idx_passages_simhash_band3 ON passages(((simhash >> 48) & 65535));

CREATE TABLE IF NOT EXISTS rejections (     -- 生成被拒统计（不存原文）
    id          INTEGER PRIMARY KEY,
    run_id      TEXT,
    category    TEXT NOT NULL,
    lang        TEXT NOT NULL CHECK (lang IN ('zh','en')),
    model       TEXT NOT NULL,
    reason      TEXT NOT NULL,             -- length|language|refusal|blocklist|structure|grounding|duplicate|safety_block
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejections_time ON rejections(created_at);

CREATE TABLE IF NOT EXISTS quarantine (
    passage_id  INTEGER NOT NULL REFERENCES passages(id),
    run_id      TEXT,
    reason      TEXT    NOT NULL,           -- 'azure_content_filter' | 'manual'
    detail      TEXT,                       -- hate/sexual/violence/self_harm/jailbreak
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('warmup','generate','load')),
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    stop_reason   TEXT,                     -- duration|signal|stop_cmd|budget|circuit_open|corpus_exhausted|supply_refused|error
    config_json   TEXT NOT NULL,            -- 去除密钥后的配置快照
    requests_ok   INTEGER DEFAULT 0,
    requests_err  INTEGER DEFAULT 0,
    throttled     INTEGER DEFAULT 0,
    probes        INTEGER DEFAULT 0,
    throttle_json TEXT,                     -- ThrottleStats 快照（含各状态累计时长）
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    passages_added INTEGER DEFAULT 0,
    cost_usd      REAL    DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ledger (
    id          INTEGER PRIMARY KEY,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    provider    TEXT NOT NULL CHECK (provider IN ('azure','gemini','gemini_grounding')),
    model       TEXT NOT NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    units       INTEGER NOT NULL DEFAULT 0,
    cost_usd    REAL NOT NULL,
    flushed_at  TEXT NOT NULL,
    account     TEXT NOT NULL DEFAULT ''    -- 预算归属（Gemini：GCP 项目/赠金账号；Azure 为空）
);
CREATE INDEX IF NOT EXISTS idx_ledger_provider_time ON ledger(provider, flushed_at);
-- idx_ledger_account 在 CorpusStore.open() 迁移（旧库补 account 列）之后创建

CREATE VIEW IF NOT EXISTS ledger_totals AS
    SELECT provider, model, SUM(cost_usd) AS total_usd,
           SUM(input_tokens) AS in_tok, SUM(output_tokens) AS out_tok, SUM(units) AS units
    FROM ledger GROUP BY provider, model;
