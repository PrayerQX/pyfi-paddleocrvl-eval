# PyFi 301 Run Report

Run date: 2026-04-12

## Setup

- Dataset: PyFi-600K CSV + `images.zip`
- Split: `data/pyfi/pyfi_eval_301.jsonl`
- Samples: 301 lines, 300 unique UIDs
- Parser: PaddleOCR-VL 1.5 via `paddleocr.PaddleOCRVL(pipeline_version="v1.5")`
- Selector: GLM OpenAI-compatible API, model `glm-4-flash`
- GLM base URL: `https://open.bigmodel.cn/api/paas/v4/`
- API key: not stored in repo

## Output Files

- Merged predictions: `runs/pyfi_paddleocrvl15_glm301_merged.jsonl`
- Merged metrics: `runs/pyfi_paddleocrvl15_glm301_merged.jsonl.metrics.json`
- Parser artifacts: `runs/pyfi_paddleocrvl15_glm301_artifacts/`
- Parser artifact count: 300 Markdown files + 300 JSON files

The run was resumed after a 4-hour shell timeout. Part files were merged by source dataset order and UID. The source split has one duplicate UID; the merged output intentionally keeps 301 rows to match the sampled split.

## Results

| Model | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Random option | 301 | 73 | 24.25% | 0 | 0.00% |
| First option | 301 | 124 | 41.20% | 0 | 0.00% |
| PaddleOCR-VL 1.5 + GLM `glm-4-flash` selector | 301 | 139 | 46.18% | 53 | 17.61% |

## By Capability

| Capability | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| Calculation_analysis | 50 | 18 | 36.00% | 7 | 14.00% |
| Data_extraction | 49 | 18 | 36.73% | 11 | 22.45% |
| Decision_support | 49 | 30 | 61.22% | 6 | 12.24% |
| Logical_reasoning | 49 | 29 | 59.18% | 5 | 10.20% |
| None | 6 | 6 | 100.00% | 0 | 0.00% |
| Pattern_recognition | 49 | 23 | 46.94% | 8 | 16.33% |
| Perception | 49 | 15 | 30.61% | 16 | 32.65% |

## By Complexity

| Complexity | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| 1 | 47 | 16 | 34.04% | 16 | 34.04% |
| 2 | 63 | 24 | 38.10% | 12 | 19.05% |
| 3 | 66 | 29 | 43.94% | 8 | 12.12% |
| 4 | 78 | 41 | 52.56% | 10 | 12.82% |
| 5 | 43 | 25 | 58.14% | 7 | 16.28% |
| None | 4 | 4 | 100.00% | 0 | 0.00% |

## Interpretation

The PaddleOCR-VL 1.5 + GLM selector chain beat the two trivial baselines on this 301-sample split, but the invalid rate is high. This result supports that the pipeline is usable and auditable, not that PaddleOCR-VL 1.5 is already the financial-document first choice.

After this run, direct GLM and OCR+LLM baselines were added. See `docs/pyfi_baseline_comparison.md`. The direct GLM `glm-4v-flash` baseline is substantially stronger on this split, so the stronger claim should not be made from these results.
