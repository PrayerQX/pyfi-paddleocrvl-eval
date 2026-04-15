# PaddleOCR-VL 1.5 + ERNIE 4.5 Hybrid PyFi Run

Date: 2026-04-15

## Chain

- Parser evidence 1: PaddleOCR-VL 1.5 markdown artifacts
- Parser evidence 2: traditional PaddleOCR text artifacts
- Selector: `ernie-4.5-turbo-128k-preview`
- Endpoint: `https://aistudio.baidu.com/llm/lmapi/v3`
- Adapter: `paddleocr-vl-hybrid-docqa`

The selector prompt now requires one valid MCQ option and rejects `null`.
If a selector still returns an invalid answer, the runner falls back to a
lexical evidence heuristic or the first valid option, which keeps invalid rate
at zero for forced-choice ranking.

## Local Results

Same first 50 records from `data/pyfi/pyfi_eval_301.jsonl`:

| Method | Total | Correct | Accuracy | Invalid |
|---|---:|---:|---:|---:|
| PaddleOCR-VL 1.5 + GLM selector | 50 | 29 | 58.00% | 7 |
| PaddleOCR text + GLM selector | 50 | 27 | 54.00% | 3 |
| PaddleOCR-VL 1.5 + OCR text + ERNIE 4.5 selector | 50 | 34 | 68.00% | 0 |
| Direct GLM `glm-4v-flash` | 50 | 38 | 76.00% | 0 |

The new Paddle chain improves the local PaddleOCR-VL score by +10.00 points on
this 50-sample slice and removes invalid outputs. It still trails the direct
VLM baseline on the same slice.

## Reproduction

Set the selector key in the shell only:

```powershell
$env:FINVL_SELECTOR_API_KEY="your-aistudio-token"
$env:FINVL_SELECTOR_BASE_URL="https://aistudio.baidu.com/llm/lmapi/v3"
```

Run the 50-sample check:

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_hybrid_ernie45_50.jsonl `
  --limit 50 `
  --require-image `
  --progress-every 10
```

Run the full 301-sample eval:

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_hybrid_ernie45_301.jsonl `
  --require-image `
  --progress-every 25
```

## ERNIE 5 Thinking Note

`ernie-5.0-thinking-preview` is wired through the same OpenAI-compatible path.
For thinking models, the adapter can use `max_completion_tokens` and streaming:

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model paddleocr-vl-hybrid-docqa `
  --selector-model ernie-5.0-thinking-preview `
  --selector-stream `
  --selector-use-max-completion-tokens `
  --selector-max-tokens 8192 `
  --artifacts-dir runs/pyfi_paddleocrvl15_glm301_artifacts `
  --ocr-artifacts-dir runs/pyfi_paddleocr_text_glm301_artifacts `
  --out runs/pyfi_paddleocrvl15_hybrid_ernie5.jsonl `
  --limit 10 `
  --require-image
```

In local testing, the thinking model was much slower than ERNIE 4.5 for batch
evaluation, so ERNIE 4.5 is the practical selector for the current ranking run.
