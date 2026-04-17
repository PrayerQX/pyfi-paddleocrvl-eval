# AGENTS.md

这是给 AI 编程代理使用的项目说明。修改代码、跑长时间评测、调用外部模型 API 前，先阅读本文件。

本项目是一个本地 PyFi / PaddleOCR-VL 评测仓库，目标是评估并提升 PaddleOCR-VL 1.5 在金融文档、图表理解、多选题 VQA 上的贡献和准确率。

---

## 写代码前

1. 先检查当前工作区：

   ```powershell
   git status --short
   ```

2. 不要凭印象改代码。按任务先读相关文件：

   - `src/finvl_eval/models.py`：模型适配器、selector 调用逻辑
   - `src/finvl_eval/runner.py`：统一评测 CLI
   - `src/finvl_eval/scoring.py`：答案归一化和指标聚合
   - `src/finvl_eval/evidence.py`：PaddleOCR-grounded 证据包构建
   - `configs/*.json`：可复现实验配置
   - `docs/*.md`：实验报告和复现说明

3. 未经明确要求，不要覆盖本地数据和运行产物。`data/`、`runs/`、`PyFi-main/` 都是本地/生成目录，体积可能很大。

4. 严禁硬编码 API key、access token 或服务凭证。使用运行时环境变量，例如：

   ```powershell
   $env:FINVL_SELECTOR_API_KEY="..."
   $env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
   ```

5. 保持改动范围收敛。本仓库是评测 harness，不是 PaddleOCR 或 PyFi 的上游源码仓库。

---

## 项目结构

```text
PaddleOCR-VL 1.5/
|-- src/finvl_eval/
|   |-- runner.py             评测入口
|   |-- models.py             模型适配器和 selector 调用
|   |-- evidence.py           PaddleOCR-grounded 证据包构建
|   |-- scoring.py            答案归一化和指标聚合
|   |-- pyfi.py               PyFi CSV/JSONL 解析
|   |-- records.py            标准化 EvalRecord 结构
|   |-- prompts.py            直接多选题 prompt 构造
|   |-- sample_pyfi.py        可复用本地 split 抽样
|   |-- download_pyfi.py      PyFi 元数据/图片下载辅助
|   `-- audit_invalid.py      invalid 答案审计报告生成
|-- tests/
|   `-- test_pyfi.py          解析、prompt、scoring 单元测试
|-- configs/                  可复现实验配置
|-- docs/                     报告和复现说明
|-- data/                     本地 PyFi CSV/图片/split，不提交
`-- runs/                     本地模型输出/artifacts/metrics，不提交
```

---

## 评测 Split

当前可复用本地 split：

```text
data/pyfi/pyfi_eval_301.jsonl
records: 301
sha256: ecf3bfc14c44f26eacc6dd3be152438373d1679daeb85d5c59dc1394d807e146
sampling seed: 20260412
stratify: capability
require image: true
```

从本地 PyFi CSV 和图片目录重新生成：

```powershell
python -m finvl_eval.sample_pyfi `
  --csv data/pyfi/PyFi-600K-dataset.csv `
  --out data/pyfi/pyfi_eval_301.jsonl `
  --limit 301 `
  --seed 20260412 `
  --stratify capability `
  --images-root data/pyfi `
  --require-image
```

注意：这是本地可复用 split。它不保证等同于 PyFi README 表格里的官方 301 样本，因为当前 checkout 中没有可复用的官方 split 文件。

---

## 模型链路

### Baseline

- `first-option`：固定选择第一个选项。
- `random-option`：带 seed 的随机选择 baseline。
- `openai-compatible-vlm`：直接图像 VLM baseline。
- `paddleocr-text-docqa`：传统 PaddleOCR 文本 + selector。
- `paddleocr-vl-docqa`：PaddleOCR-VL markdown + selector。

### PaddleOCR 重点链路

- `paddleocr-vl-hybrid-docqa`
  - 当前本地分数最高的 Paddle 链路。
  - 使用 PaddleOCR-VL markdown + 传统 PaddleOCR 文本。
  - 让 selector 读取完整 Paddle 证据并强制输出一个合法选项。
  - 适合冲本地 leaderboard。

- `paddleocr-vl-grounded-docqa`
  - 更可审计，更突出 PaddleOCR 贡献。
  - 从 Paddle 表格行、OCR 行、选项命中、数字候选和 raw Paddle 附录构建 evidence packet。
  - 要求 selector 返回答案和短证据字段。
  - 适合证明 PaddleOCR 的作用，即使准确率可能低于 hybrid 链路。

---

## 常用命令

安装本地包：

```powershell
python -m pip install -e .
```

运行测试：

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

在 local-301 split 上运行当前最佳 PaddleOCR 链路：

```powershell
$env:FINVL_SELECTOR_API_KEY="your-aistudio-token"
$env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_local301_paddleocrvl15_hybrid_ernie45.jsonl `
  --require-image `
  --progress-every 25
```

运行更可审计的 grounded 链路：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-grounded-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_grounded_ernie45_301.jsonl `
  --require-image `
  --progress-every 25
```

从已有 JSONL 输出生成可复用本地 leaderboard 报告：

```powershell
python -m finvl_eval.compare_runs `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --out docs/pyfi_local301_reusable_leaderboard.md `
  --run "Random option=runs/pyfi_random_option_301.jsonl" `
  --run "First option=runs/pyfi_first_option_301.jsonl" `
  --run "Direct GLM-4V=runs/pyfi_direct_glm4v_flash_301.jsonl" `
  --run "PaddleOCR text + GLM=runs/pyfi_paddleocr_text_glm301.jsonl" `
  --run "PaddleOCR-VL + GLM=runs/pyfi_paddleocrvl15_glm301_merged.jsonl" `
  --run "PaddleOCR-VL hybrid + ERNIE 4.5=runs/pyfi_local301_paddleocrvl15_hybrid_ernie45.jsonl"
```

---

## 新增模型适配器

新 adapter 放在 `src/finvl_eval/models.py`。

必须满足：

1. 实现 `predict(record, image_path, prompt) -> str | None`。
2. 返回原始模型输出，答案归一化交给 `normalize_prediction`。
3. 不要硬编码选项字母；通过 `record.valid_options` 和 scoring 逻辑处理。
4. 只有 CLI 明确传入 artifact 目录时，才把可复现产物写到 `runs/`。
5. 不打印密钥，也不要打印包含凭证的完整 prompt。

如果 adapter 调用 OpenAI-compatible API：

- 使用 `FINVL_SELECTOR_API_KEY` 或 `OPENAI_API_KEY`。
- 使用 `FINVL_SELECTOR_BASE_URL` 或 `OPENAI_BASE_URL`。
- API key 不进入 config、doc、代码。
- 单条样本异常应由 `runner.py` 记录到结果 JSONL，不应让整次评测直接中断。

---

## 评分规则

严格多选题准确率：

```text
correct = normalized_prediction == gold_answer
invalid = normalized_prediction is None
```

`src/finvl_eval/scoring.py` 中的 `normalize_answer` 支持：

- 原始字母：`A`
- 短文本：`The answer is B.`
- JSON：`{"answer":"C"}`
- fenced JSON block

不要放宽成从长推理文本里随便抽取单独字母。否则会把无效推理误算成有效答案，污染 leaderboard。

---

## 实验报告规范

重要实验必须记录：

- dataset 路径和 SHA256
- model adapter
- selector model 和 base URL，不记录 API key
- 使用的 artifact 目录
- 输出 JSONL 路径
- total、correct、accuracy、invalid、invalid rate
- 按 capability 和 complexity 的准确率
- 结果是本地 split，还是能和外部 leaderboard 对比

`docs/` 放简洁报告，`configs/` 放可复用运行配置。

推荐命名：

```text
configs/pyfi_<split>_<chain>_<selector>.json
docs/pyfi_<split>_<chain>_<selector>_report.md
runs/pyfi_<split>_<chain>_<selector>.jsonl
```

---

## 数据和产物策略

不要提交：

- `data/`
- `runs/`
- `PyFi-main/`
- `*.zip`
- `*.egg-info/`
- `__pycache__/`
- 模型下载文件、OCR 生成产物、API 日志、截图、大型临时文件

可以提交：

- `src/` 里的源码
- `tests/` 里的聚焦测试
- `configs/` 里的小型 JSON 配置
- `docs/` 里的简洁报告
- 本文件 `AGENTS.md`

---

## 测试要求

代码改动后运行：

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
```

涉及以下内容时，应添加或更新测试：

- PyFi 解析
- 答案归一化
- 评分聚合
- 会影响确定性行为的 prompt 或 evidence 格式
- CLI 参数行为

长时间 API 评测前先跑小样本 smoke：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/smoke_hybrid_ernie45.jsonl `
  --limit 5 `
  --require-image `
  --progress-every 1
```

---

## 硬性规则

- 提交文件中不得出现 API key。
- 除非用户明确要求，不使用破坏性 git 命令。
- 不把 `data/` 或 `runs/` 当成源码内容提交。
- 不宣称本地 split 是 PyFi 官方 leaderboard split。
- 没有精确报告 split、hash、模型链路和输出路径时，不做 leaderboard 声明。
- prompt、adapter、heuristic、evidence builder 中不得使用 gold answer。
- 做模型改进时，不进行无关的大型重构。

---

## 当前最佳本地结果

在可复用 local-301 split 上：

```text
PaddleOCR-VL hybrid + ERNIE 4.5
output: runs/pyfi_local301_paddleocrvl15_hybrid_ernie45.jsonl
total: 301
correct: 197
accuracy: 65.45%
invalid: 0
```

这是当前仓库里最强的 PaddleOCR-centered 链路。后续改进应以它作为要超越的 baseline。
