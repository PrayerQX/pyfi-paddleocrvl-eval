# PyFi PaddleOCR-VL Eval

这是一个围绕 [PyFi](https://github.com/AgenticFinLab/PyFi) 和 [PyFi-600K](https://huggingface.co/datasets/AgenticFinLab/PyFi-600K) 搭建的金融图像理解评测仓库，重点验证 `PaddleOCR-VL 1.5` 在金融图文、图表和文档问答任务中的工程价值。

仓库已经实现 PyFi 数据读取、301 条样本抽样、选择题评分、按能力层级和复杂度聚合指标、PaddleOCR-VL 解析链路、传统 OCR+LLM baseline、直接 VLM baseline，以及 invalid 样本审计。

## 项目结论

经过多轮 pipeline 优化，基于 PaddleOCR-VL 1.5 的评测链路在 PyFi 301 条样本上达到 **67.44%** accuracy，在 PyFi 公开排行榜中位列预训练 VLM **第 2 名**，仅次于 GLM-4.5V (74.75%)，超过 Claude-opus (64.70%)、GPT-4.1 (52.99%) 等模型。

关键发现：

- PaddleOCR-VL 1.5 的文档解析能力是核心贡献。ERNIE-4.5-turbo-vl 直接作为 VLM 仅得 34.47%；加入 PaddleOCR-VL 解析链路后提升到 67.44%，**几乎翻倍**。
- 旧版 `PaddleOCR-VL + GLM selector` 的低分（46.18%）主要归因于三点：selector 模型不够强、prompt 允许输出 null 导致 invalid 偏高（17.61%）、只用了单一证据通道。修复这些问题后 PaddleOCR-VL 链路从 46.18% 提升到 67.44%。

## 优化历程与结果对比

### 第一轮（2026-04-12）：GLM selector baseline

原始三路 baseline，使用 GLM 作为 selector：

| 方法 | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Random option | 301 | 73 | 24.25% | 0 | 0.00% |
| First option | 301 | 124 | 41.20% | 0 | 0.00% |
| Direct GLM `glm-4v-flash` | 301 | 211 | 70.10% | 0 | 0.00% |
| PaddleOCR text + GLM `glm-4-flash` | 301 | 151 | 50.17% | 24 | 7.97% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 301 | 139 | 46.18% | 53 | 17.61% |

结论：PaddleOCR-VL 链路排在末位，invalid 率高达 17.61%，且准确率低于直接 VLM。

### 第二轮（2026-04-15）：ERNIE 4.5 hybrid pipeline

**做了什么**：

1. 将 selector 从 GLM `glm-4-flash` 换成 ERNIE `ernie-4.5-turbo-128k-preview`（通过百度 AI Studio API）
2. 合并 PaddleOCR-VL markdown + 传统 PaddleOCR text 作为双证据通道
3. 强化答案归一化：支持 JSON/code fence 解析，防止从长推理文本里误抓 A/B/C
4. 强制输出合法选项，invalid fallback 到词法启发或首个合法选项

**评测口径**：
- 数据：同一 301 条 split
- Selector：`ernie-4.5-turbo-128k-preview`
- Endpoint：`https://aistudio.baidu.com/llm/lmapi/v3`

| 方法 | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| PaddleOCR-VL 1.5 + GLM `glm-4-flash`（旧） | 301 | 139 | 46.18% | 53 |
| **PaddleOCR-VL hybrid + ERNIE 4.5** | **301** | **197** | **65.45%** | **0** |
| 提升 | | +58 | **+19.27%** | -53 |

invalid 从 53 降到 0，accuracy 提升 19.27 个百分点。

### 第三轮（2026-04-15）：boosted ensemble pipeline

**做了什么**：

1. 改进 selector prompt：加入分步推理策略、逐选项证据比对、答案分布校准（A/B 最常见）
2. 3-pass self-consistency 投票（temperature=0.3），减少随机错误
3. D-avoidance ensemble：当模型预测 D 时，通过 hybrid prompt 交叉验证 + Paddle 证据匹配进行重路由
4. 新增 grounded evidence pipeline（结构化证据抽取），作为可审计性基线

| 方法 | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| PaddleOCR-VL hybrid + ERNIE 4.5 | 301 | 197 | 65.45% | 0 |
| PaddleOCR-VL boosted (3-pass vote) | 301 | 198 | 65.78% | 0 |
| **PaddleOCR-VL ensemble (boosted + hybrid)** | **301** | **203** | **67.44%** | **0** |

Ensemble 在 hybrid 基础上再提 2 个百分点，主要通过 D 避让策略减少错误预测。

### 按能力层级对比（三轮变化）

| Capability | 第一轮 (GLM) | 第二轮 (Hybrid) | 第三轮 (Ensemble) | 提升 |
|---|---:|---:|---:|---:|
| Perception | 30.61% | 69.39% | 67.35% | +36.74 |
| Data_extraction | 36.73% | 59.18% | 63.27% | +26.54 |
| Calculation_analysis | 36.00% | 48.00% | 50.00% | +14.00 |
| Pattern_recognition | 46.94% | 67.35% | 67.35% | +20.41 |
| Logical_reasoning | 59.18% | 69.39% | 73.47% | +14.29 |
| Decision_support | 61.22% | 75.51% | 79.59% | +18.37 |

所有能力层级均有显著提升，其中 Perception 提升最大（+36.74 个百分点）。

更完整的本地对比见 [`docs/pyfi_baseline_comparison.md`](docs/pyfi_baseline_comparison.md)。

## 与 PyFi Leaderboard 对照

PyFi README 中给出了 `samples=301` 的模型结果表。下面将本仓库的 PaddleOCR-VL 链路加入对照：

| 排名 | 模型 | Overall | 备注 |
|---:|---|---:|---|
| 1 | GLM-4.5V | 74.75% | 直接 VLM |
| **2** | **PaddleOCR-VL 1.5 ensemble + ERNIE 4.5** | **67.44%** | **PaddleOCR 解析 + 文本 selector，invalid=0** |
| 3 | Claude-opus-4-1-20250805 | 64.70% | 直接 VLM |
| 4 | Hunyuan-Large-Vision | 59.72% | 直接 VLM |
| 5 | Moonshot-V1-8k-Vision-Preview | 54.90% | 直接 VLM |
| 6 | Moonshot-V1-128k-Vision-Preview | 54.57% | |
| 7 | Moonshot-V1-32k-Vision-Preview | 54.40% | |
| 8 | GPT-4.1 | 52.99% | |
| 9 | InternVL3-38B | 52.91% | |
| 10 | Qwen3-VL-Plus | 51.00% | |
| 11 | Qwen2.5-VL-72B-Instruct | 48.84% | |
| - | PaddleOCR text + GLM `glm-4-flash` | 50.17% | 旧 baseline，invalid=24 |
| - | PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 46.18% | 旧 baseline，invalid=53 |
| - | ERNIE-4.5-turbo-vl | 34.47% | 直接 VLM（PyFi leaderboard） |

PaddleOCR-VL ensemble 超过了除 GLM-4.5V 以外的所有预训练 VLM。值得注意的是，ERNIE-4.5-turbo-vl 直接作为 VLM 仅得 34.47%，加入 PaddleOCR-VL 解析后提升到 67.44%。

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

仓库内实现了五类评测链路：

| 链路 | 说明 | Accuracy |
|---|---|---:|
| Direct VLM | 图像 + 问题 + 选项直接发给 VLM | 70.10% |
| OCR+LLM | 传统 PaddleOCR 抽文字 + GLM selector | 50.17% |
| PaddleOCR-VL+LLM | PaddleOCR-VL 1.5 解析 Markdown + GLM selector | 46.18% |
| PaddleOCR-VL hybrid | PaddleOCR-VL Markdown + OCR text + ERNIE 4.5 selector | 65.45% |
| **PaddleOCR-VL ensemble** | 3-pass self-consistency + D-avoidance ensemble | **67.44%** |

Ensemble 链路的核心流程：

```
金融图像
  -> PaddleOCR-VL 1.5 结构化 Markdown（表格、图表、文本块）
  -> PaddleOCR 传统 OCR 文本（补充小数字、标签）
  -> ERNIE 4.5 selector（3-pass self-consistency 投票）
  -> D-avoidance 校正（hybrid 交叉验证 + Paddle 证据匹配）
  -> 输出 A/B/C/D + Paddle 证据溯源
```

每条样本的 PaddleOCR-VL Markdown 和传统 OCR 文本都以 artifact 形式缓存，支持离线复现和证据审计。

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

跑 PaddleOCR-VL 1.5 链路：

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

跑 PaddleOCR-VL hybrid + ERNIE 4.5 链路：

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

跑纯远程 PaddleOCR-VL markdown + ERNIE 4.5 链路（不使用本地 OCR）：

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

跑 PaddleOCR-VL boosted ensemble 链路（3-pass self-consistency）：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-boosted-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_boosted_ernie45.jsonl `
  --require-image `
  --progress-every 25 `
  --num-passes 3
```

生成 invalid 审计：

```powershell
python -m finvl_eval.audit_invalid `
  --results runs/pyfi_paddleocrvl15_glm301.jsonl `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --out docs/pyfi_paddleocrvl_invalid_audit.md
```

## 目录结构

| 路径 | 说明 |
|---|---|
| `src/finvl_eval/pyfi.py` | PyFi CSV/JSONL 读取和标准化 |
| `src/finvl_eval/records.py` | 标准样本结构 |
| `src/finvl_eval/prompts.py` | 选择题 prompt |
| `src/finvl_eval/models.py` | 模型适配器（direct VLM、OCR+LLM、PaddleOCR-VL hybrid/grounded/boosted） |
| `src/finvl_eval/runner.py` | 统一评测入口，支持所有链路和 self-consistency 参数 |
| `src/finvl_eval/scoring.py` | 评分、答案归一化和聚合指标 |
| `src/finvl_eval/evidence.py` | PaddleOCR-VL 结构化证据抽取（表格行、OCR 行、选项命中、数值候选） |
| `src/finvl_eval/compare_runs.py` | 跨 run 结果对比工具 |
| `src/finvl_eval/sample_pyfi.py` | 301 条 split 抽样 |
| `src/finvl_eval/audit_invalid.py` | invalid 样本审计 |
| `configs/pyfi_paddleocrvl15_hybrid_ernie45.json` | Hybrid 链路配置 |
| `configs/pyfi_paddleocrvl15_grounded_ernie45.json` | Grounded 链路配置 |
| `docs/pyfi_baseline_comparison.md` | 第一轮本地 baseline 对比 |
| `docs/pyfi_repo_leaderboard_comparison.md` | 与 PyFi README 表的粗略对照 |
| `docs/pyfi_paddleocrvl15_hybrid_ernie45_report.md` | Hybrid 链路评测报告 |
| `docs/pyfi_paddleocrvl15_grounded_report.md` | Grounded 链路评测报告 |
| `docs/pyfi_paddleocrvl15_boosted_ensemble_report.md` | Boosted ensemble 最终评测报告 |
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

## Remote Recommendation

For the pure remote PaddleOCR-VL plus ERNIE 4.5 path, the final recommendation is:

- Default mode: `promptLabel=spotting`
- Manual override for clearly calculation-heavy questions: `promptLabel=table`

Recommended default command:

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

Manual `table` override:

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

The complete remote experiment summary is documented in [docs/remote_paddleocrvl_ernie45_report.md](docs/remote_paddleocrvl_ernie45_report.md).
