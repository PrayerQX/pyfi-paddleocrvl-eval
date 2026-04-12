# PyFi 复现流程

## 1. 数据集格式

PyFi-600K 的 Hugging Face 数据集中包含：

- `PyFi-600K-dataset.csv`
- `PyFi-600K-dataset.json`
- `PyFi-600K-chain-dataset.json`
- `PyFi-600K-chain-CoT-dataset.json`
- `images.zip`

我用 HTTP Range 检查过 CSV/JSON 首段，基础样本字段是：

```text
question_node_no, options, complexity, visit_count, fq_no, capability,
victory_count, image_background, parent_node_no, actions, question, image_path
```

其中：

- `options` 是选项字典。
- `actions` 是 action 列表，`actions[*].answer` 是选择题答案。
- `capability` 是能力层级。
- `complexity` 是复杂度。
- `image_background` 是上下文背景。
- `image_path` 指向 `./images/.../*.jpg`。

## 2. 本地准备

```powershell
python -m pip install -e .
python -m finvl_eval.download_pyfi --out data/pyfi --files readme csv images
Expand-Archive data/pyfi/images.zip -DestinationPath data/pyfi -Force
```

如果只想先检查流程，可以先不下载 `images.zip`，但真实评测必须带图像。

## 3. 抽评测集

PyFi README 中示例评测表使用 `samples=301`。为了可复现，这里固定 seed 并按 capability 轮转采样：

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

## 4. 复现直接 VLM 基线

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model openai-compatible-vlm `
  --openai-model glm-4v-flash `
  --out runs/pyfi_direct_glm4v_flash_301.jsonl
```

## 5. 复现传统 OCR + LLM 基线

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://open.bigmodel.cn/api/paas/v4/"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-text-docqa `
  --selector-model glm-4-flash `
  --artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocr_text_glm301.jsonl `
  --require-image
```

## 6. 复现 PaddleOCR-VL 1.5 解析链路

PaddleOCR-VL 1.5 是文档解析 pipeline，不是纯 VQA 数据集的直接答题器。这里采用：

1. PaddleOCR-VL 1.5 解析图片为 Markdown/JSON。
2. 选择器模型只基于解析结果、问题和选项输出答案字母。
3. 保存解析 artifact，避免只看最终选择题正确率。

```powershell
$env:FINVL_SELECTOR_MODEL="gpt-4.1-mini"
$env:OPENAI_API_KEY="..."

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-docqa `
  --selector-model $env:FINVL_SELECTOR_MODEL `
  --artifacts-dir runs/pyfi_paddleocrvl15_artifacts `
  --out runs/pyfi_paddleocrvl15.jsonl
```

如果你有 PaddleOCR-VL 的服务端推理地址，可以加：

```powershell
  --paddle-vl-backend <backend> `
  --paddle-vl-server-url <server-url>
```

## 7. 结果检查

每次评测会生成：

- `runs/<name>.jsonl`：逐样本输出。
- `runs/<name>.jsonl.metrics.json`：总体准确率、无效输出率、按 capability 和 complexity 的分桶结果。
- `runs/pyfi_paddleocrvl15_artifacts/*.md`：PaddleOCR-VL 解析文本。
- `runs/pyfi_paddleocrvl15_artifacts/*.json`：PaddleOCR-VL 结构化解析结果。

报告至少包含：

- Overall accuracy
- Invalid rate
- Accuracy by capability
- Accuracy by complexity
- Parse failure rate
- Average latency
- Cost per 1000 images

本机 2026-04-12 的 301 条结果见：

- `docs/pyfi_baseline_comparison.md`
- `docs/pyfi_paddleocrvl_invalid_audit.md`
- `docs/pyfi_ocr_text_invalid_audit.md`

## 8. 外部资料

- PyFi code: https://github.com/AgenticFinLab/PyFi
- PyFi-600K dataset: https://huggingface.co/datasets/AgenticFinLab/PyFi-600K
- PyFi paper page: https://arxiv.org/abs/2512.14735
- PaddleOCR-VL 1.5 docs: https://www.paddleocr.ai/main/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.5.html
- PaddleOCR-VL pipeline usage: https://www.paddleocr.ai/main/version3.x/pipeline_usage/PaddleOCR-VL.html
