# FinVL Eval

这是一个基于 [AgenticFinLab/PyFi](https://github.com/AgenticFinLab/PyFi) 和 [PyFi-600K](https://huggingface.co/datasets/AgenticFinLab/PyFi-600K) 搭出来的金融图文/文档解析评测骨架，目标是把 PyFi 的评测方式提炼成可迁移的领域评测工具，并接入 PaddleOCR-VL 1.5。

当前状态：

- 已确认 PyFi-600K 的核心字段：`question_node_no, options, complexity, visit_count, fq_no, capability, victory_count, image_background, parent_node_no, actions, question, image_path`。
- 已实现 PyFi CSV/JSONL 读取、gold answer 抽取、选择题 prompt、按 `capability`/`complexity` 聚合指标、抽样脚本和评测 runner。
- 已接入 `paddleocr.PaddleOCRVL(pipeline_version="v1.5")` 作为文档解析适配器。PyFi 是选择题 VQA 格式，因此这里采用“PaddleOCR-VL 解析 Markdown + 选择器模型答题”的组合。
- 已在当前机器上跑完 PyFi 301 条 split：direct GLM `glm-4v-flash` 为 70.10% accuracy；传统 PaddleOCR text + GLM `glm-4-flash` 为 50.17%；PaddleOCR-VL 1.5 + GLM `glm-4-flash` selector 为 46.18% accuracy、17.61% invalid rate。结果见 `docs/pyfi_baseline_comparison.md`。

## PyFi 评测形态

PyFi-600K 是金融图像理解数据集，公开说明中称其包含约 600K 样本，每条样本包含金融图像、Q&A、背景信息、能力层级和复杂度。PyFi README 把能力层级分为 6 类：`Perception`、`Data_extraction`、`Calculation_analysis`、`Pattern_recognition`、`Logical_reasoning`、`Decision_support`。

本工具把 PyFi 行标准化成：

```json
{
  "uid": "./images/000001/000010.jpg::fq1::node1",
  "image_path": "./images/000001/000010.jpg",
  "question": "Which color in the chart represents ...?",
  "options": {"A": "Light green", "B": "Medium green"},
  "answer": "A",
  "capability": "Perception",
  "complexity": "1",
  "context": {"image_background": "..."}
}
```

`answer` 来自 `actions[*].answer`。如果有多个 action，代码优先选择 `victory_count` 高、其次 `visit_count` 高的 action。

## 快速复现

安装本地包：

```powershell
python -m pip install -e .
```

下载 PyFi 元数据。默认只下载 README 和 CSV；图像包需要显式指定 `images`：

```powershell
python -m finvl_eval.download_pyfi --out data/pyfi --files readme csv
python -m finvl_eval.download_pyfi --out data/pyfi --files images
```

从完整 CSV 抽一个 301 条、按能力层级轮转的评测集：

```powershell
python -m finvl_eval.sample_pyfi `
  --csv data/pyfi/PyFi-600K-dataset.csv `
  --out data/pyfi/pyfi_eval_301.jsonl `
  --limit 301 `
  --stratify capability `
  --images-root data/pyfi `
  --require-image
```

跑 smoke test 基线：

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model first-option `
  --out runs/pyfi_first_option.jsonl
```

跑 PaddleOCR-VL 1.5 文档解析链路：

```powershell
$env:FINVL_SELECTOR_MODEL="gpt-4.1-mini"
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://api.openai.com/v1"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-docqa `
  --selector-model $env:FINVL_SELECTOR_MODEL `
  --artifacts-dir runs/pyfi_paddleocrvl15_artifacts `
  --out runs/pyfi_paddleocrvl15.jsonl
```

输出包含逐样本 JSONL 和指标文件：`runs/*.jsonl.metrics.json`。

## 结论口径

现在不能写“PaddleOCR-VL 1.5 是金融领域文档解析首选模型”。在这次 301 条 PyFi split 上，direct GLM `glm-4v-flash` 明显领先，传统 PaddleOCR text + GLM 也高于 PaddleOCR-VL 1.5 + GLM。PaddleOCR-VL 的价值主要体现在可审计的 Markdown/JSON 中间产物，但需要改进 evidence packing 和 selector。建议的 claim gate 是：

- 至少跑 PyFi 301 条公开可复现 split，并保存全部原始输出与 Markdown/JSON 解析 artifact。
- 报告 overall、6 个 capability 维度、5 个 complexity 维度、无效输出率、解析失败率、平均延迟和成本。
- 与至少 3 类 baseline 对比：直接 VLM、传统 OCR+LLM、其它文档解析模型。
- 只有当 PaddleOCR-VL 1.5 在总体质量胜出或统计打平，同时在成本、延迟或失败率上有优势时，才写“首选”。

## 目录

- `src/finvl_eval/pyfi.py`：PyFi 解析和 JSONL 序列化。
- `src/finvl_eval/runner.py`：统一评测入口。
- `src/finvl_eval/models.py`：模型适配器，包括 PaddleOCR-VL 1.5 doc-QA 链路。
- `src/finvl_eval/sample_pyfi.py`：评测 split 抽样。
- `configs/pyfi_paddleocrvl15.json`：推荐实验配置。
- `docs/pyfi_reproduction.md`：复现流程。
- `docs/domain_methodology.md`：迁移到医疗等专业领域的方法论。
- `docs/article_draft.md`：文章草稿模板。
- `docs/pyfi_glm_run_report.md`：本机 301 条 PyFi + GLM selector 运行结果。
- `docs/pyfi_baseline_comparison.md`：direct VLM、OCR+LLM、PaddleOCR-VL+LLM 的对比报告。
- `docs/pyfi_paddleocrvl_invalid_audit.md`：PaddleOCR-VL 链路 invalid 样本审计。
- `docs/pyfi_ocr_text_invalid_audit.md`：传统 OCR 链路 invalid 样本审计。
