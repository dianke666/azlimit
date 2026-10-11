# 部署说明

目标目录 `/opt/a_azure_limit`，以专用非 root 用户 `loadtest` 运行。工具不监听任何网络端口。

## 1. 安装

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin loadtest
sudo mkdir -p /opt/a_azure_limit && sudo chown loadtest:loadtest /opt/a_azure_limit
# 将代码放入 /opt/a_azure_limit 后：
cd /opt/a_azure_limit
sudo -u loadtest python3.11 -m venv .venv
sudo -u loadtest .venv/bin/pip install -r requirements.lock
sudo -u loadtest .venv/bin/pip install --no-deps -e .
sudo -u loadtest mkdir -p data runs credentials
```

以下命令默认在 `/opt/a_azure_limit` 下以 `loadtest` 用户执行，并已 `source .venv/bin/activate`、加载 `.env`（如 `set -a; . ./.env; set +a`），且：

```bash
export TIKTOKEN_CACHE_DIR=/opt/a_azure_limit/data/tiktoken   # 与 systemd 单元一致（ProtectHome=yes 下 ~/.cache 不可用）
```

tiktoken 编码文件首次使用时需联网下载一次，之后离线可用；`loadtest doctor` 会完成这一步。

## 2. 配置

```bash
cp config.example.yaml config.yaml     # 修改 gemini.project 为你的 GCP 项目 ID
cp .env.example .env                   # 填写 AZURE_OPENAI_API_KEY / AZURE_OPENAI_ENDPOINT / GOOGLE_APPLICATION_CREDENTIALS
# 服务账号 JSON 放到 credentials/ 下
sudo chown loadtest:loadtest .env credentials/*.json
sudo chmod 600 .env credentials/*.json
sudo chmod 700 credentials
```

`.env` 与凭据文件不得提交到仓库（已在 `.gitignore` 中忽略）。

### 自检（真实调用，花费 < $0.05）

```bash
loadtest doctor                 # 全部检查；任何 FAIL 时退出码 1
loadtest doctor --skip-azure    # 只查 Gemini（换 GCP 账号后）
```

检查项：tokenizer、数据库、Azure 鉴权/deployment/部署 TPM 与 RPM 是否覆盖目标、Gemini 链中每个模型是否可用且 thinking_level 被接受、grounding 是否返回来源及计费查询数。全部 OK（或只有可解释的 WARN）后再 warmup。

## 3. 预热语料

```bash
loadtest warmup --target 8000 --concurrency 32 --max-cost 20   # 约 30 分钟，≈ $14（gemini-3.1-flash-lite）
loadtest pool-stats                                            # 库存、语言/长度桶分布、拒绝率、Gemini 花费
loadtest budget                                                # 已用/上限、20 分钟预估、剩余可运行次数
```

## 4. 运行压测

```bash
# 运行前核对（不调用 Azure、不写 Azure 账本）
loadtest run --dry-run --duration 2m

# 小流量冒烟
loadtest run --duration 1m --tpm 50000

# 正式运行：在 tmux 中
tmux new -s loadtest
loadtest run --duration 20m
# Ctrl-b d 脱离；tmux attach -t loadtest 重新连接

# 另一个终端
loadtest status          # run_id 与已运行时长
loadtest stop            # 写 runs/STOP 并发送 SIGTERM，优雅停止
loadtest report          # 最近一次报告；loadtest report <RUN_ID> 查看指定 run
```

同一时间只允许一个 run（`runs/loadtest.pid` 锁）。

## 5. systemd（可选）

```bash
sudo cp deploy/loadtest-run.service deploy/loadtest-refresh.service deploy/loadtest-refresh.timer /etc/systemd/system/
sudo systemctl daemon-reload

# 每日 06:00 刷新新闻/经济/体育 grounding 语料（300 条，72h 过期）
sudo systemctl enable --now loadtest-refresh.timer
systemctl list-timers loadtest-refresh.timer

# 压测只手动启动，不要 enable
sudo systemctl start loadtest-run
sudo systemctl stop loadtest-run          # 发送 SIGTERM，60s 内完成收尾
journalctl -u loadtest-run -f
journalctl -u loadtest-refresh
```

单元使用 `ProtectSystem=strict`，仅 `data/` 与 `runs/` 可写；若修改 `db_path` / `log_dir`，需同步调整 `ReadWritePaths`。

## 6. 日志与报告

- `runs/<run_id>/`：每次运行的日志与报告
- `runs/loadtest.pid`、`runs/STOP`：运行锁与停止标记
- `data/corpus.db`：语料库与费用账本

## 7. 预算

- 累计上限：Azure $180、Gemini 每个 GCP 账号 $250（`config.yaml` 的 `cost.*_budget_cap_usd`），超出预估时拒绝启动。
- `cost.prices` 中的单价于 2026-10 对照官方价格页核对；部署前请再核对一次（Vertex 价格与模型下线日期变化较快）。
- Gemini 默认链：gemini-3.1-flash-lite → gemini-3.5-flash-lite → gemini-3.8-flash（gemini-2.x 已或即将下线）。确认 GCP 项目在 `global` 端点可用这些模型。
- 预估：预热 8,000 段 ≈ $14；每次 20 分钟运行的后台补充 ≈ $1；grounding 按实际搜索查询数 × $0.014 记账，每日上限 600 次查询（≤ $8.4/天）。
- Gemini 预算按账号独立：默认以 `gemini.project` 为账号。换新 GCP 赠金账号时，改 `config.yaml` 的 `gemini.project` 并替换 `credentials/` 下的服务账号 JSON，预算从 0 重新计；语料库（`data/corpus.db`）保留，不需要重新 warmup。`loadtest budget` 会单列旧账号的花费。

## 8. 安全提示

- 不开放任何端口；服务器仅需出站 HTTPS。
- 密钥只放在 `.env` 与 `credentials/`，权限 600，不入库、不写入日志。
- 不要把 `openai` / `httpx` 的日志级别设为 DEBUG，否则可能输出请求头中的密钥。
