# PaddleOCR-VL Grounded Chain Report

Date: 2026-04-15

## Goal

This run tests whether PaddleOCR can be made more central to the PyFi answer
chain, instead of only passing raw markdown/text to a selector model.

## Chain

Adapter: `paddleocr-vl-grounded-docqa`

```text
image
-> PaddleOCR-VL 1.5 markdown/table artifacts
-> PaddleOCR text artifacts
-> grounded evidence packet
-> ERNIE 4.5 selector
-> JSON answer with Paddle evidence
```

The evidence packet includes:

- PaddleOCR-VL HTML table rows
- PaddleOCR text lines
- option-aware evidence hits
- exploratory numeric candidates
- raw PaddleOCR-VL markdown appendix
- raw PaddleOCR text appendix

## Local 50-Sample Result

Same first 50 records from `data/pyfi/pyfi_eval_301.jsonl`:

| Method | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| PaddleOCR text + GLM selector | 50 | 27 | 54.00% | 3 |
| PaddleOCR-VL 1.5 + GLM selector | 50 | 29 | 58.00% | 7 |
| PaddleOCR-VL grounded + ERNIE 4.5 | 50 | 30 | 60.00% | 0 |
| PaddleOCR-VL hybrid + ERNIE 4.5 | 50 | 34 | 68.00% | 0 |
| Direct GLM `glm-4v-flash` | 50 | 38 | 76.00% | 0 |

Capability split for grounded + ERNIE 4.5:

| Capability | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| Calculation_analysis | 8 | 3 | 37.50% | 0 |
| Data_extraction | 8 | 4 | 50.00% | 0 |
| Decision_support | 7 | 5 | 71.43% | 0 |
| Logical_reasoning | 7 | 7 | 100.00% | 0 |
| Pattern_recognition | 7 | 3 | 42.86% | 0 |
| Perception | 7 | 2 | 28.57% | 0 |
| None | 6 | 6 | 100.00% | 0 |

## Ranking Read

This is not a formal PyFi leaderboard result because the official reusable
301-sample split is not available in this repo, and this run used only the
first 50 local records.

On the local 50-record slice:

1. Direct GLM `glm-4v-flash`: 76.00%
2. PaddleOCR-VL hybrid + ERNIE 4.5: 68.00%
3. PaddleOCR-VL grounded + ERNIE 4.5: 60.00%
4. PaddleOCR-VL + GLM selector: 58.00%
5. PaddleOCR text + GLM selector: 54.00%

Naively comparing 60.00% to the PyFi README model table would place the
grounded chain around Hunyuan-Large-Vision's 59.72% and below Claude's 64.70%.
That comparison is only directional. The local first-50 slice appears easier
than the full local 301 split: the old PaddleOCR-VL + GLM chain scores 58.00%
on these 50 records but 46.18% on the local 301 records.

## Takeaway

The grounded chain better exposes PaddleOCR's contribution because the selector
must work from Paddle table rows, OCR lines, option hits, and numeric candidates.
It also removes invalid answers. However, it underperforms the less constrained
hybrid ERNIE chain on accuracy.

For ranking, use `paddleocr-vl-hybrid-docqa`.
For proving PaddleOCR contribution and producing auditable evidence, use
`paddleocr-vl-grounded-docqa`.

## Reproduction

```powershell
$env:FINVL_SELECTOR_API_KEY="your-aistudio-token"
$env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"

python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-grounded-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_grounded_ernie45_50.jsonl `
  --limit 50 `
  --require-image `
  --progress-every 10
```

Full local 301 run:

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
