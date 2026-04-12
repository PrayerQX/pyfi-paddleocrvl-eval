# 文章草稿：用 PyFi 证明 PaddleOCR-VL 1.5 的金融文档解析价值

> 当前是草稿模板，不是最终论文结论。完整结论必须在真实评测跑完后填入结果表。

## 标题

PaddleOCR-VL 1.5 在金融图像文档解析中的可复验证据：基于 PyFi-600K 的分层评测

## 摘要

金融图像文档包含图表、表格、文本和专业符号，普通视觉语言模型在抽取数值、理解图表关系和执行多步推理时容易失误。本文基于 PyFi-600K 构建可复现评测流程，将任务拆分为文档解析、选择题问答和分层指标报告，并接入 PaddleOCR-VL 1.5。实验将报告整体准确率、能力层级准确率、复杂度准确率、解析失败率、延迟和成本，用于判断 PaddleOCR-VL 1.5 是否可以作为金融领域文档解析的首选模型。

## 数据集

PyFi-600K 提供约 600K 金融图像-文本问答样本，样本包含图像路径、问题、选项、答案 action、图像背景、能力层级和复杂度。能力层级包括 Perception、Data_extraction、Calculation_analysis、Pattern_recognition、Logical_reasoning、Decision_support。

本文先采用固定 seed 抽取 301 条样本作为公开 smoke/evidence split，再扩展到更大规模评测。

## 方法

评测链路包括：

1. 从 PyFi CSV 解析样本。
2. 使用 `actions[*].answer` 生成 gold answer。
3. 使用统一 prompt 要求模型只输出选项字母。
4. 对 PaddleOCR-VL 1.5，先把图像解析为 Markdown/JSON，再由选择器模型基于解析结果答题。
5. 保存逐样本输出、解析 artifact 和分桶指标。

## 对比组

建议至少包含：

- 直接 VLM：例如 GLM-4.5V、Qwen-VL、GPT 系列或内部金融 VLM。
- OCR + LLM：传统 OCR 抽取文本后用 LLM 答题。
- PaddleOCR-VL 1.5 + selector：本文目标链路。

## 结果表模板

| Model | Overall | Invalid | PP | DE | CA | PR | LR | DS | Latency/img | Cost/1k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Random option | 24.25 | 0.00 | 24.49 | 18.37 | 28.00 | 18.37 | 26.53 | 30.61 | N/A | N/A |
| First option | 41.20 | 0.00 | 38.78 | 36.73 | 38.00 | 46.94 | 42.86 | 44.90 | N/A | N/A |
| Direct GLM `glm-4v-flash` | 70.10 | 0.00 | 83.67 | 75.51 | 46.00 | 71.43 | 65.31 | 77.55 | TBD | TBD |
| PaddleOCR text + GLM `glm-4-flash` | 50.17 | 7.97 | 48.98 | 36.73 | 42.00 | 46.94 | 57.14 | 67.35 | TBD | TBD |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` selector | 46.18 | 17.61 | 30.61 | 36.73 | 36.00 | 46.94 | 59.18 | 61.22 | TBD | TBD |

## 结论模板

当前 301 条运行结果不支持写“PaddleOCR-VL 1.5 是金融领域文档解析首选模型”。Direct GLM `glm-4v-flash` 以 70.10% overall accuracy 领先，传统 PaddleOCR text + GLM 也达到 50.17%，均高于 PaddleOCR-VL 1.5 + GLM 的 46.18%。PaddleOCR-VL 的优势更像是可审计的结构化中间产物，而不是当前 PyFi 选择题准确率。

如果后续 PaddleOCR-VL 1.5 在总体质量上领先或统计打平，并且解析失败率、成本或延迟更优，可以写：

“在 PyFi 固定评测 split 上，PaddleOCR-VL 1.5 + selector 在金融图像文档解析任务中取得了可复现的质量/效率优势。由于其保留 Markdown/JSON 中间解析结果，错误可审计性优于直接 VLM，因此更适合作为金融文档解析基础模型。”

如果结果没有领先，应写：

“PaddleOCR-VL 1.5 在可审计解析链路上具备工程价值，但当前 PyFi 选择题指标尚不足以支持‘首选模型’结论。下一步应补充表格、图表数据重建和人工审计指标。”

## 复现命令

```powershell
python -m finvl_eval.download_pyfi --out data/pyfi --files readme csv images
Expand-Archive data/pyfi/images.zip -DestinationPath data/pyfi -Force
python -m finvl_eval.sample_pyfi --csv data/pyfi/PyFi-600K-dataset.csv --out data/pyfi/pyfi_eval_301.jsonl --limit 301 --stratify capability --images-root data/pyfi --require-image
python -m finvl_eval.runner --dataset data/pyfi/pyfi_eval_301.jsonl --format jsonl --images-root data/pyfi --model paddleocr-vl-docqa --selector-model $env:FINVL_SELECTOR_MODEL --artifacts-dir runs/pyfi_paddleocrvl15_artifacts --out runs/pyfi_paddleocrvl15.jsonl
```
