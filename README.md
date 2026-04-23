# PyFi PaddleOCR-VL Eval

这是一个围绕 [PyFi](https://github.com/AgenticFinLab/PyFi) 和 [PyFi-600K](https://huggingface.co/datasets/AgenticFinLab/PyFi-600K) 搭建的金融图像理解评测仓库，重点验证 `PaddleOCR-VL 1.5` 在金融图文、图表和文档问答任务中的工程价值。

仓库已经实现 PyFi 数据读取、301 条样本抽样、选择题评分、按能力层级和复杂度聚合指标、PaddleOCR-VL 解析链路、传统 OCR+LLM baseline、直接 VLM baseline、纯远程 PaddleOCR-VL + ERNIE 4.5 链路，以及 invalid 样本审计。

## 项目结论

当前 README 只把可复用、可默认开启的通用链路作为推荐方案，不再把依赖特化规则的集成策略作为主结论。

当前推荐路径是：**纯远程 PaddleOCR-VL 1.5 markdown + ERNIE 4.5 selector**。

关键结论：

- 在 `data/pyfi/pyfi_eval_301.jsonl` 上，纯远程单模式里最稳的配置是 `promptLabel=spotting`，实验结果为 **63.12%** accuracy、**0.33%** invalid rate。
- 在当前代码上的最终验证里，推荐命令对应的 `spotting` 结果为 **62.46%** accuracy、**0.66%** invalid rate。
- `promptLabel=table` 只作为**明显偏计算型题目**的手动覆盖，不作为默认模式。
- 不默认启用自动路由，也不把依赖特定误差模式的规避策略写成推荐方案。
- 对照 PyFi leaderboard，`ernie-4.5-turbo-vl` 直接作为 VLM 仅有 **34.47%**；接入 PaddleOCR-VL 解析后，通用远程链路可以稳定提升到 **62%+**。

## 结果演进

### 第一轮（2026-04-12）：GLM selector baseline

原始三路 baseline，使用 GLM 作为 selector：

| 方法 | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Random option | 301 | 73 | 24.25% | 0 | 0.00% |
| First option | 301 | 124 | 41.20% | 0 | 0.00% |
| Direct GLM `glm-4v-flash` | 301 | 211 | 70.10% | 0 | 0.00% |
| PaddleOCR text + GLM `glm-4-flash` | 301 | 151 | 50.17% | 24 | 7.97% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 301 | 139 | 46.18% | 53 | 17.61% |

结论：早期 `PaddleOCR-VL + GLM` 链路准确率偏低，且 invalid 明显偏高。

### 第二轮（2026-04-15）：ERNIE 4.5 + PaddleOCR-VL 统一链路

这一轮的目标是把链路收敛到更可复用的 PaddleOCR-VL + 文本 selector 方案：

1. 将 selector 从 GLM `glm-4-flash` 切换到 ERNIE `ernie-4.5-turbo-128k-preview`
2. 接入统一的远程 PaddleOCR-VL API
3. 强化答案归一化，支持 JSON 和 code fence 解析
4. 强制输出合法选项，减少 invalid
5. 默认不依赖本地 OCR 证据

评测口径：

- 数据：同一 301 条 split
- Selector：`ernie-4.5-turbo-128k-preview`
- Endpoint：`https://aistudio.baidu.com/llm/lmapi/v3`

| 方法 | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| PaddleOCR-VL 1.5 + GLM `glm-4-flash`（旧） | 301 | 139 | 46.18% | 53 |
| **PaddleOCR-VL + ERNIE 4.5** | **301** | **197** | **65.45%** | **0** |
| 提升 | | +58 | **+19.27%** | -53 |

这说明问题的核心不是“再叠更多特化规则”，而是把解析链路和 selector 基础能力切换到更稳的组合。

### 第三轮（2026-04-20）：纯远程 promptLabel 对比

在统一远程链路上，进一步比较不同 `promptLabel` 的单模式表现：

| 模式 | 输出 | Accuracy | Invalid Rate | 结论 |
|---|---|---:|---:|---|
| `spotting` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting.jsonl` | `63.12%` | `0.33%` | 最稳的默认模式 |
| `table` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_table.jsonl` | `62.13%` | `0.66%` | 更适合明显计算型题目 |
| `seal` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_seal.jsonl` | `61.46%` | `7.31%` | rate limit 和 invalid 都更差 |
| `formula` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_formula.jsonl` | `58.14%` | `10.30%` | 不推荐 |

当前结论：

- 默认使用 `promptLabel=spotting`
- 对明显 calculation-heavy 问题手动切到 `promptLabel=table`
- 不默认启用自动 `promptLabel` 路由

完整实验记录见 [`docs/remote_paddleocrvl_ernie45_report.md`](docs/remote_paddleocrvl_ernie45_report.md)。

## 与 PyFi Leaderboard 对照

PyFi README 中给出了 `samples=301` 的模型结果表。下面把本仓库当前推荐的通用远程链路加入对照：

| 模型 | Overall | 备注 |
|---|---:|---|
| GLM-4.5V | 74.75% | 直接 VLM |
| Claude-opus-4-1-20250805 | 64.70% | 直接 VLM |
| **Pure remote PaddleOCR-VL + ERNIE 4.5 (`spotting`)** | **63.12%** | **推荐的通用链路** |
| Hunyuan-Large-Vision | 59.72% | 直接 VLM |
| GPT-4.1 | 52.99% | 直接 VLM |
| PaddleOCR text + GLM `glm-4-flash` | 50.17% | 旧 baseline |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 46.18% | 旧 baseline |
| ERNIE-4.5-turbo-vl | 34.47% | 直接 VLM（PyFi leaderboard） |

如果按当前代码上的最终验证结果统计，推荐命令对应的 `spotting` 结果为 **62.46%**。无论采用实验最佳单模式结果还是最终验证结果，结论都一致：**PaddleOCR-VL 的结构化解析对金融图像问答有稳定收益，但默认推荐应保持为简单、通用、可复用的远程链路**。

完整对照见 [`docs/pyfi_repo_leaderboard_comparison.md`](docs/pyfi_repo_leaderboard_comparison.md)。

## 数据与评测形态

PyFi-600K 是金融图像理解数据集，公开说明中称其包含约 600K 样本。每条样本包含图像、问题、选项、答案 action、图像背景、能力层级和复杂度。

本仓库把原始 CSV 行标准化成以下结构：

```json
{
  "uid": "./images/000001/000010.jpg::fq1::node1",
  "image_path": "./images/000001/000010.jpg",
  "question": "Which color in the chart represents ...?",
  "options": {
    "A": "Light green",
    "B": "Medium green"
  },
  "answer": "A",
  "capability": "Perception",
  "complexity": "1",
  "context": {
    "image_background": "..."
  }
}
```

能力层级包括：

- `Perception`
- `Data_extraction`
- `Calculation_analysis`
- `Pattern_recognition`
- `Logical_reasoning`
- `Decision_support`

`answer` 来自 `actions[*].answer`。当同一行有多个 action 时，代码优先选 `victory_count` 更高的 action，其次选 `visit_count` 更高的 action。

## 模型链路

仓库内保留了几类评测链路，其中当前默认推荐的是纯远程 PaddleOCR-VL + ERNIE 4.5：

| 链路 | 说明 | Accuracy |
|---|---|---:|
| Direct VLM | 图像 + 问题 + 选项直接发给 VLM | 70.10% |
| OCR+LLM | 传统 PaddleOCR 抽文字 + GLM selector | 50.17% |
| PaddleOCR-VL+LLM | PaddleOCR-VL 1.5 解析 Markdown + GLM selector | 46.18% |
| PaddleOCR-VL + ERNIE 4.5 | PaddleOCR-VL Markdown + ERNIE 4.5 selector | 65.45% |
| **Pure remote PaddleOCR-VL + ERNIE 4.5 (`spotting`)** | **当前推荐的通用默认链路** | **63.12%** |

推荐链路的核心流程：

```text
金融图像
  -> remote PaddleOCR-VL 1.5 结构化 Markdown
  -> ERNIE 4.5 selector
  -> 输出 A/B/C/D
```

默认配置说明：

- 默认 `promptLabel=spotting`
- `promptLabel=table` 只作为人工判断后的手动覆盖
- 不默认拼接本地 OCR 作为第二证据通道
- 不默认启用自动 promptLabel 路由

每条样本的 PaddleOCR-VL Markdown 都以 artifact 形式缓存，支持离线复现和错误分析。

## 快速开始

安装本地包：

```powershell
python -m pip install -e .
```

下载 PyFi 元数据。默认建议先下载 README 和 CSV，图像包较大，需要单独下载：

```powershell
python -m finvl_eval.download_pyfi --out data/pyfi --files readme csv
python -m finvl_eval.download_pyfi --out data/pyfi --files images
```

抽样 301 条评测集：

```powershell
python -m finvl_eval.sample_pyfi `
  --csv data/pyfi/PyFi-600K-dataset.csv `
  --out data/pyfi/pyfi_eval_301.jsonl `
  --limit 301 `
  --stratify capability `
  --images-root data/pyfi `
  --require-image
```

跑直接 VLM baseline：

```powershell
$env:OPENAI_API_KEY="your-api-key"
$env:OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model openai-compatible-vlm `
  --openai-model glm-4v-flash `
  --out runs/pyfi_direct_glm4v_flash_301.jsonl
```

跑传统 OCR+LLM baseline：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-text-docqa `
  --selector-model glm-4-flash `
  --artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocr_text_glm301.jsonl
```

跑 PaddleOCR-VL 1.5 + GLM baseline：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-docqa `
  --selector-model glm-4-flash `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_glm301.jsonl
```

跑 PaddleOCR-VL + ERNIE 4.5 链路（默认不使用本地 OCR 证据）：

```powershell
$env:FINVL_SELECTOR_API_KEY="your-aistudio-token"
$env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
$env:PADDLEOCR_VL_API_URL="https://your-remote-host/layout-parsing"
$env:PADDLEOCR_VL_API_TOKEN="your-paddleocr-vl-token"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_hybrid_ernie45.jsonl `
  --no-local-ocr-evidence `
  --require-image `
  --progress-every 25
```

跑纯远程 PaddleOCR-VL markdown + ERNIE 4.5 链路：

```powershell
$env:FINVL_SELECTOR_API_KEY="your-aistudio-token"
$env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
$env:PADDLEOCR_VL_API_URL="https://your-remote-host/layout-parsing"
$env:PADDLEOCR_VL_API_TOKEN="your-paddleocr-vl-token"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model remote-paddleocr-vl-ernie-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_artifacts `
  --out runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45.jsonl `
  --require-image `
  --progress-every 25
```

生成 invalid 审计：

```powershell
python -m finvl_eval.audit_invalid `
  --results runs/pyfi_paddleocrvl15_glm301.jsonl `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --out docs/pyfi_paddleocrvl_invalid_audit.md
```

## 默认推荐

对于纯远程 PaddleOCR-VL + ERNIE 4.5 路径，最终推荐是：

- 默认模式：`promptLabel=spotting`
- 手动覆盖：对明显 calculation-heavy 题目使用 `promptLabel=table`

推荐默认命令：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model remote-paddleocr-vl-ernie-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --selector-base-url https://aistudio.baidu.com/llm/lmapi/v3 `
  --artifacts-dir runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting_artifacts `
  --out runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting.jsonl `
  --require-image `
  --progress-every 25
```

手动 `table` 覆盖：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model remote-paddleocr-vl-ernie-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --selector-base-url https://aistudio.baidu.com/llm/lmapi/v3 `
  --paddle-vl-prompt-label table `
  --artifacts-dir runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_table_artifacts `
  --out runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_table.jsonl `
  --require-image `
  --progress-every 25
```

## 目录结构

| 路径 | 说明 |
|---|---|
| `src/finvl_eval/pyfi.py` | PyFi CSV/JSONL 读取和标准化 |
| `src/finvl_eval/records.py` | 标准样本结构 |
| `src/finvl_eval/prompts.py` | 选择题 prompt |
| `src/finvl_eval/models.py` | 模型适配器（direct VLM、OCR+LLM、PaddleOCR-VL、remote PaddleOCR-VL + ERNIE） |
| `src/finvl_eval/runner.py` | 统一评测入口，支持所有链路与远程参数 |
| `src/finvl_eval/scoring.py` | 评分、答案归一化和聚合指标 |
| `src/finvl_eval/evidence.py` | PaddleOCR-VL 结构化证据抽取 |
| `src/finvl_eval/compare_runs.py` | 跨 run 结果对比工具 |
| `src/finvl_eval/sample_pyfi.py` | 301 条 split 抽样 |
| `src/finvl_eval/audit_invalid.py` | invalid 样本审计 |
| `configs/pyfi_paddleocrvl15_hybrid_ernie45.json` | PaddleOCR-VL + ERNIE 4.5 配置 |
| `configs/pyfi_pure_remote_paddleocrvl_ernie45.json` | 纯远程默认配置 |
| `configs/pyfi_pure_remote_paddleocrvl_ernie45_table.json` | 纯远程 `table` 覆盖配置 |
| `docs/pyfi_baseline_comparison.md` | baseline 对比 |
| `docs/pyfi_repo_leaderboard_comparison.md` | 与 PyFi README 表的对照 |
| `docs/pyfi_paddleocrvl15_hybrid_ernie45_report.md` | PaddleOCR-VL + ERNIE 4.5 评测报告 |
| `docs/remote_paddleocrvl_ernie45_report.md` | 纯远程链路实验报告 |
| `docs/pyfi_reproduction.md` | 复现流程 |
| `docs/pyfi_paddleocrvl_invalid_audit.md` | PaddleOCR-VL invalid 审计 |
| `docs/pyfi_ocr_text_invalid_audit.md` | OCR+LLM invalid 审计 |

## 仓库内容边界

以下内容不会上传到 GitHub：

- `data/`：PyFi CSV、图像包和抽样数据
- `runs/`：模型输出、metrics、Markdown/JSON artifacts
- `PyFi-main/` 和 `PyFi-main.zip`：上游仓库副本
- `*.egg-info/`、`__pycache__/` 等缓存

这样做是为了避免把大文件、第三方数据副本和本地运行产物推到代码仓库里。结果摘要已经固化到 `docs/`，需要复查原始输出时再按复现流程本地生成。

## 验证

本地已通过：

```powershell
python -m compileall -q src tests
python -m unittest discover -s tests -v
python -m pip install -e .
```

## 参考

- PyFi: https://github.com/AgenticFinLab/PyFi
- PyFi-600K: https://huggingface.co/datasets/AgenticFinLab/PyFi-600K
- PaddleOCR-VL 1.5: https://www.paddleocr.ai/main/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html
- PaddleOCR-VL pipeline: https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
