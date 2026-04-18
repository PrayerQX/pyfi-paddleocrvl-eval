# AGENTS.md

这是给 AI 编程代理使用的项目说明。修改代码、运行评测、调用外部 API 前，必须先阅读本文件。

本分支用于 PyFi / PaddleOCR-VL 实验，重点是用 **远程 PaddleOCR-VL API** 作为文档/图表解析底座，再用 ERNIE 4.5 等 selector 做多选题推理评测。

## 分支硬性要求

1. Python 环境必须使用 `uv` 管理，并与全局 Python 隔离。
   - 远程项目目录：`remote_paddleocr_app/`
   - 推荐命令：

     ```powershell
     cd remote_paddleocr_app
     C:\Users\小周\.local\bin\uv.exe sync
     C:\Users\小周\.local\bin\uv.exe run python -m pytest
     ```

   - 不要用全局 `pip install` 修改本机全局环境。
   - 不要在仓库根目录误创建 `.venv` 或 `uv.lock`。远程项目的虚拟环境和 lock file 应保留在 `remote_paddleocr_app/` 内。

2. 本分支优先使用远程 PaddleOCR。
   - 实验和评测默认走 `remote_paddleocr_app`。
   - PaddleOCR-VL 解析必须通过远程 API：

     ```text
     PADDLEOCR_API_URL=https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing
     ```

   - 除非用户明确要求，不使用本地 PaddleOCR / 本地 PaddleOCR-VL 模型加载。
   - 不要把本地 `runs/pyfi_paddleocrvl15_*_artifacts` 当成远程 PaddleOCR-VL 结果报告。
   - 远程 artifacts 应写入 `remote_paddleocr_app/output/...`，并且不提交运行产物。

3. 必须符合学术评测要求。
   - 不允许使用 gold answer 参与 prompt、adapter、evidence builder、veto、fallback 或任何推理逻辑。
   - 不允许针对任何选项字母做特殊处理，例如 A/B 优先、D avoidance、first-option fallback、按选项分布校准等。
   - 所有选项必须一视同仁，只能根据远程 PaddleOCR 证据、题目和选项文本判断。
   - 不允许基于测试集结果做 stacking、capability-specific routing、按能力类别调权或测试集校准。
   - 可以做统一适用于所有样本的改动，例如统一 prompt、统一结构化证据格式、统一解析鲁棒性改进。
   - 任何可能影响分数的策略必须能在不看 gold answer 的情况下预先定义。

## 密钥与安全

- 严禁硬编码 API key、access token 或服务凭证。
- 使用运行时环境变量：

  ```powershell
  $env:PADDLEOCR_API_TOKEN="..."
  $env:ERNIE_API_KEY="..."
  $env:ERNIE_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
  ```

- 不要打印密钥，不要把包含密钥的完整命令写入文档、配置或提交。
- `.env`、API 日志、原始响应大文件和运行结果不得提交。

## 当前主要实验链路

### 远程严格基线

```text
remote PaddleOCR-VL markdown + ERNIE 4.5 single-pass selector
```

特点：

- 远程 PaddleOCR-VL 解析图片。
- ERNIE 4.5 只做一次 forced-choice 选择。
- 不使用 fallback、stacking、选项偏置或 capability 分流。

### 远程 layout blocks 链路

```text
remote PaddleOCR-VL markdown + layout blocks + ERNIE 4.5
```

特点：

- 在 markdown 之外，统一加入远程 PaddleOCR-VL 的 `prunedResult.parsing_res_list` 版面块文本和 bbox。
- 仍然对所有样本使用同一 prompt 和同一证据格式。
- 当前远程严格口径下表现较好。

### 实验 3：远程 PaddleOCR-VL 结构底座

```text
remote PaddleOCR-VL layout parsing
-> structured intermediate evidence
-> ERNIE 4.5 selector
```

要求：

- 阶段 A 只能使用远程 PaddleOCR-VL 返回的 `layoutParsingResults`、`prunedResult`、`markdown` 等内容。
- 阶段 B 只能读取题目、选项、image background 和阶段 A 的结构化证据。
- 不允许使用本地 PaddleOCR-VL artifacts 冒充远程结果。
- 不允许根据 capability 切换 prompt 或证据结构。

## 推荐运行命令

进入远程项目：

```powershell
cd remote_paddleocr_app
```

运行测试和 lint：

```powershell
C:\Users\小周\.local\bin\uv.exe run python -m pytest
C:\Users\小周\.local\bin\uv.exe run ruff check .
```

远程实验 smoke：

```powershell
C:\Users\小周\.local\bin\uv.exe run remote-paddleocr eval-pyfi `
  --dataset ..\data\pyfi\pyfi_eval_301.jsonl `
  --images-root ..\data\pyfi `
  --out output\smoke_remote_exp3_structured_ernie45_5.jsonl `
  --artifacts-dir output\eval_remote_chart_ernie5_50_artifacts `
  --limit 5 `
  --use-chart-recognition `
  --structured-intermediate `
  --disable-web-search `
  --ernie-model ernie-4.5-turbo-128k-preview `
  --max-completion-tokens 256 `
  --retry-attempts 5 `
  --retry-base-sleep 8 `
  --sleep-between-records 0.5 `
  --progress-every 1
```

完整 301 评测：

```powershell
C:\Users\小周\.local\bin\uv.exe run remote-paddleocr eval-pyfi `
  --dataset ..\data\pyfi\pyfi_eval_301.jsonl `
  --images-root ..\data\pyfi `
  --out output\eval_remote_exp3_structured_ernie45_301.jsonl `
  --artifacts-dir output\eval_remote_chart_ernie5_50_artifacts `
  --use-chart-recognition `
  --structured-intermediate `
  --disable-web-search `
  --ernie-model ernie-4.5-turbo-128k-preview `
  --max-completion-tokens 256 `
  --retry-attempts 5 `
  --retry-base-sleep 8 `
  --sleep-between-records 0.5 `
  --progress-every 25
```

如需减少重复远程 selector 调用，可以启用 selector 缓存和小规模并发：

```powershell
C:\Users\小周\.local\bin\uv.exe run remote-paddleocr eval-pyfi `
  --dataset ..\data\pyfi\pyfi_eval_301.jsonl `
  --images-root ..\data\pyfi `
  --out output\eval_remote_exp3_structured_cached_ernie45_301.jsonl `
  --artifacts-dir output\eval_remote_chart_ernie5_50_artifacts `
  --selector-cache-dir output\selector_cache_exp3_structured_ernie45 `
  --selector-concurrency 4 `
  --use-chart-recognition `
  --structured-intermediate `
  --disable-web-search `
  --ernie-model ernie-4.5-turbo-128k-preview `
  --max-completion-tokens 256 `
  --retry-attempts 5 `
  --retry-base-sleep 8 `
  --resume `
  --progress-every 25
```

缓存和并发只允许用于工程加速，不得改变 prompt、证据、选项处理或评分逻辑。首次运行仍受远程服务限流影响；缓存命中后的复跑才会明显加速。

## 数据和产物策略

不要提交：

- `data/`
- `runs/`
- `remote_paddleocr_app/output/`
- `PyFi-main/`
- `.env`
- `*.zip`
- `*.egg-info/`
- `__pycache__/`
- 模型文件、OCR 生成产物、API 日志、截图、大型临时文件

可以提交：

- `src/`
- `remote_paddleocr_app/src/`
- `tests/`
- `remote_paddleocr_app/tests/`
- `configs/`
- `docs/`
- `AGENTS.md`

## 评测报告规范

重要实验必须记录：

- dataset 路径和 SHA256
- 是否为本地可复用 split，不能宣称它是 PyFi 官方 leaderboard split
- model / adapter / prompt profile
- PaddleOCR 来源：必须明确是远程还是本地
- selector model 和 base URL，不记录 API key
- artifact 目录
- 输出 JSONL 路径
- total、correct、accuracy、invalid、invalid rate
- 按 capability 和 complexity 的分项准确率

## 修改代码前

1. 先检查工作区：

   ```powershell
   git status --short
   ```

2. 先读相关文件，不要凭印象改：

   - `remote_paddleocr_app/src/remote_paddleocr_app/eval_pyfi.py`
   - `remote_paddleocr_app/src/remote_paddleocr_app/paddleocr_client.py`
   - `remote_paddleocr_app/src/remote_paddleocr_app/ernie_client.py`
   - `remote_paddleocr_app/tests/test_payload.py`
   - `src/finvl_eval/models.py`
   - `src/finvl_eval/runner.py`
   - `src/finvl_eval/evidence.py`

3. 长时间 API 评测前必须先跑小样本 smoke。

4. 代码改动后至少运行：

   ```powershell
   C:\Users\小周\.local\bin\uv.exe run python -m pytest
   C:\Users\小周\.local\bin\uv.exe run ruff check .
   ```

   如果改到主 harness，也运行：

   ```powershell
   python -m compileall -q src tests
   python -m unittest discover -s tests -v
   ```
