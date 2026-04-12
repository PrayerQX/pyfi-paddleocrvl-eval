# PyFi PaddleOCR-VL Eval

这是一个围绕 [PyFi](https://github.com/AgenticFinLab/PyFi) 和 [PyFi-600K](https://huggingface.co/datasets/AgenticFinLab/PyFi-600K) 搭建的金融图像理解评测仓库，重点验证 `PaddleOCR-VL 1.5` 在金融图文、图表和文档问答任务中的工程价值。

仓库已经实现 PyFi 数据读取、301 条样本抽样、选择题评分、按能力层级和复杂度聚合指标、PaddleOCR-VL 解析链路、传统 OCR+LLM baseline、直接 VLM baseline，以及 invalid 样本审计。

## 项目结论

当前 301 条本地 split 的结果不能支持“PaddleOCR-VL 1.5 是金融文档解析首选模型”这个强结论。

更稳妥的结论是：PaddleOCR-VL 1.5 能生成可审计的 Markdown/JSON 中间解析结果，适合作为金融文档解析链路的一部分；但在 PyFi 这种选择题 VQA 评测中，当前 `PaddleOCR-VL 1.5 + GLM selector` 的准确率和 invalid 率都弱于直接 VLM 和传统 OCR+LLM baseline。

## 本地 Baseline 对比

评测口径：

- 数据：从 PyFi-600K CSV 中抽样 301 条，要求图像存在，并按 capability 轮转采样
- 时间：2026-04-12
- 直接 VLM：GLM `glm-4v-flash`
- 选择器模型：GLM `glm-4-flash`
- PaddleOCR-VL：`paddleocr.PaddleOCRVL(pipeline_version="v1.5")`

| 方法 | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Random option | 301 | 73 | 24.25% | 0 | 0.00% |
| First option | 301 | 124 | 41.20% | 0 | 0.00% |
| Direct GLM `glm-4v-flash` | 301 | 211 | 70.10% | 0 | 0.00% |
| PaddleOCR text + GLM `glm-4-flash` | 301 | 151 | 50.17% | 24 | 7.97% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 301 | 139 | 46.18% | 53 | 17.61% |

按能力层级看，PaddleOCR-VL 链路在 `Logical_reasoning` 和 `Decision_support` 上相对更强，但在 `Perception` 和 `Data_extraction` 上 invalid 率较高。

| 方法 | PP | DE | CA | PR | LR | DS |
|---|---:|---:|---:|---:|---:|---:|
| Direct GLM `glm-4v-flash` | 83.67% | 75.51% | 46.00% | 71.43% | 65.31% | 77.55% |
| PaddleOCR text + GLM `glm-4-flash` | 48.98% | 36.73% | 42.00% | 46.94% | 57.14% | 67.35% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 30.61% | 36.73% | 36.00% | 46.94% | 59.18% | 61.22% |

按复杂度看，PaddleOCR-VL 链路在高复杂度样本上没有崩掉，但整体仍低于直接 VLM。

| 方法 | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| Direct GLM `glm-4v-flash` | 80.85% | 77.78% | 59.09% | 64.10% | 74.42% |
| PaddleOCR text + GLM `glm-4-flash` | 48.94% | 41.27% | 42.42% | 55.13% | 62.79% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` | 34.04% | 38.10% | 43.94% | 52.56% | 58.14% |

更完整的本地对比见 [`docs/pyfi_baseline_comparison.md`](docs/pyfi_baseline_comparison.md)。

## 与 PyFi README 表的粗略对照

PyFi README 中给出了一个 `samples=301` 的模型结果表，但仓库没有提供可直接复用的官方 split 文件。因此下面只能作为粗略横向对照，不能当成正式 leaderboard 排名。

本地 `PaddleOCR-VL 1.5 + GLM selector` 为 46.18%。和 PyFi README 中几家主要模型相比：

| 模型 | Overall | 相对 PaddleOCR-VL |
|---|---:|---:|
| GLM-4.5V | 74.75% | +28.57 |
| Claude-opus-4-1-20250805 | 64.70% | +18.52 |
| Hunyuan-Large-Vision | 59.72% | +13.54 |
| Moonshot-V1-8k-Vision-Preview | 54.90% | +8.72 |
| GPT-4.1 | 52.99% | +6.81 |
| InternVL3-38B | 52.91% | +6.73 |
| Qwen3-VL-Plus | 51.00% | +4.82 |
| Qwen2.5-VL-72B-Instruct | 48.84% | +2.66 |
| PaddleOCR-VL 1.5 + GLM selector | 46.18% | 0.00 |
| DeepSeek-VL2 | 45.18% | -1.00 |
| PyFi-QwenVL-7B-500 | 43.44% | -2.74 |
| Qwen2.5-VL-32B-Instruct | 43.19% | -2.99 |
| PyFi-QwenVL-7B-COT-47K | 42.61% | -3.57 |

粗略位置：PaddleOCR-VL 链路低于强闭源/大模型 VLM 和 Qwen2.5-VL-72B，高于 DeepSeek-VL2、Qwen2.5-VL-32B、较小 Qwen 模型，以及 PyFi README 中列出的 PyFi SFT 模型。但由于本地链路有 17.61% invalid，实际质量不能只看 overall accuracy。

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

仓库内实现了三类主要评测链路：

| 链路 | 说明 |
|---|---|
| Direct VLM | 直接把图像、问题、选项发给 OpenAI-compatible VLM，例如 GLM `glm-4v-flash` |
| OCR+LLM | 使用传统 PaddleOCR 抽取文字，再交给 GLM 选择器答题 |
| PaddleOCR-VL+LLM | 使用 PaddleOCR-VL 1.5 解析图像为 Markdown/JSON，再交给 GLM 选择器答题 |

PaddleOCR-VL 链路的优势是可审计性：每个样本都会保留 Markdown 和 JSON 解析产物。劣势是当前 evidence packing 和 selector 还不够强，导致 invalid 偏高。

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
| `src/finvl_eval/models.py` | 模型适配器，包括 direct VLM、OCR+LLM、PaddleOCR-VL+LLM |
| `src/finvl_eval/runner.py` | 统一评测入口 |
| `src/finvl_eval/scoring.py` | 评分和聚合指标 |
| `src/finvl_eval/sample_pyfi.py` | 301 条 split 抽样 |
| `src/finvl_eval/audit_invalid.py` | invalid 样本审计 |
| `docs/pyfi_baseline_comparison.md` | 本地 baseline 对比 |
| `docs/pyfi_repo_leaderboard_comparison.md` | 与 PyFi README 表的粗略对照 |
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
