# PyFi Baseline Comparison

Run date: 2026-04-12

## Summary

| Model | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Random option | 301 | 73 | 24.25% | 0 | 0.00% |
| First option | 301 | 124 | 41.20% | 0 | 0.00% |
| Direct GLM glm-4v-flash | 301 | 211 | 70.10% | 0 | 0.00% |
| PaddleOCR text + GLM glm-4-flash | 301 | 151 | 50.17% | 24 | 7.97% |
| PaddleOCR-VL 1.5 + GLM glm-4-flash | 301 | 139 | 46.18% | 53 | 17.61% |

## Accuracy By Capability

| Model | PP | DE | CA | PR | LR | DS |
|---|---:|---:|---:|---:|---:|---:|
| Random option | 24.49% | 18.37% | 28.00% | 18.37% | 26.53% | 30.61% |
| First option | 38.78% | 36.73% | 38.00% | 46.94% | 42.86% | 44.90% |
| Direct GLM glm-4v-flash | 83.67% | 75.51% | 46.00% | 71.43% | 65.31% | 77.55% |
| PaddleOCR text + GLM glm-4-flash | 48.98% | 36.73% | 42.00% | 46.94% | 57.14% | 67.35% |
| PaddleOCR-VL 1.5 + GLM glm-4-flash | 30.61% | 36.73% | 36.00% | 46.94% | 59.18% | 61.22% |

## Invalid Rate By Capability

| Model | PP | DE | CA | PR | LR | DS |
|---|---:|---:|---:|---:|---:|---:|
| Random option | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| First option | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| Direct GLM glm-4v-flash | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| PaddleOCR text + GLM glm-4-flash | 10.20% | 10.20% | 14.00% | 8.16% | 0.00% | 6.12% |
| PaddleOCR-VL 1.5 + GLM glm-4-flash | 32.65% | 22.45% | 14.00% | 16.33% | 10.20% | 12.24% |

## Accuracy By Complexity

| Model | C1 | C2 | C3 | C4 | C5 |
|---|---:|---:|---:|---:|---:|
| Random option | 23.40% | 19.05% | 27.27% | 28.21% | 23.26% |
| First option | 38.30% | 39.68% | 48.48% | 42.31% | 37.21% |
| Direct GLM glm-4v-flash | 80.85% | 77.78% | 59.09% | 64.10% | 74.42% |
| PaddleOCR text + GLM glm-4-flash | 48.94% | 41.27% | 42.42% | 55.13% | 62.79% |
| PaddleOCR-VL 1.5 + GLM glm-4-flash | 34.04% | 38.10% | 43.94% | 52.56% | 58.14% |

## Interpretation

- Direct GLM `glm-4v-flash` is the strongest baseline on this split: 70.10% accuracy and 0% invalid rate.
- Traditional OCR text + GLM reaches 50.17%, outperforming PaddleOCR-VL 1.5 + GLM on this specific multiple-choice QA setup.
- PaddleOCR-VL 1.5 + GLM reaches 46.18% and has a 17.61% invalid rate. Its Markdown/JSON artifacts are useful for auditability, but this result does not support calling it the first-choice model for PyFi-style financial VQA.
- The next productive step is improving the selector prompt or evidence packing for parsed Markdown, then re-running the same split.

## Files

- Direct VLM: `runs/pyfi_direct_glm4v_flash_301.jsonl`
- OCR + LLM: `runs/pyfi_paddleocr_text_glm301.jsonl`
- PaddleOCR-VL + LLM: `runs/pyfi_paddleocrvl15_glm301_merged.jsonl`
- PaddleOCR-VL invalid audit: `docs/pyfi_paddleocrvl_invalid_audit.md`
- OCR invalid audit: `docs/pyfi_ocr_text_invalid_audit.md`
