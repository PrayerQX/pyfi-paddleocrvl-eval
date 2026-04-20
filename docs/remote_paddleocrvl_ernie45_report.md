# Remote PaddleOCR-VL + ERNIE 4.5 Report

This document summarizes the pure remote PaddleOCR-VL 1.5 plus ERNIE 4.5 experiments on `data/pyfi/pyfi_eval_301.jsonl`.

## Experiment Summary

All runs below use:

- Remote PaddleOCR-VL endpoint: `layout-parsing`
- Selector model: `ernie-4.5-turbo-128k-preview`
- No local OCR evidence

### Canonical Single-Mode Runs

| Mode | Output | Accuracy | Invalid Rate | Notes |
|---|---|---:|---:|---|
| `spotting` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting.jsonl` | `63.12%` | `0.33%` | Best stable single-mode run |
| `table` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_table.jsonl` | `62.13%` | `0.66%` | Best for calculation-heavy questions |
| `seal` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_seal.jsonl` | `61.46%` | `7.31%` | Accuracy is acceptable, but rate limiting is much worse |
| `formula` | `runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_formula.jsonl` | `58.14%` | `10.30%` | Not recommended |

### PromptLabel Routing Experiments

| Mode | Output | Accuracy | Invalid Rate | Conclusion |
|---|---|---:|---:|---|
| Capability oracle over all four labels | `runs/pyfi_eval_301_pure_remote_paddleocrvl_capability_route.jsonl` | `66.78%` | `1.66%` | Useful as an upper-bound experiment only |
| Capability oracle over `table + spotting` | `runs/pyfi_eval_301_promptlabel_route_report_table_spotting_only.json` | `65.12%` | `1.00%` | Shows there is real routing headroom |
| Real rule router v3 | `runs/pyfi_eval_301_pure_remote_paddleocrvl_table_spotting_router_v3.jsonl` | `62.46%` | `1.00%` | Better than `table`, still worse than pure `spotting` |
| Real rule router v4 | `runs/pyfi_eval_301_pure_remote_paddleocrvl_table_spotting_router_v4.jsonl` | `61.46%` | `0.00%` | Stable but not better |
| Minimal real router | `runs/pyfi_eval_301_pure_remote_paddleocrvl_table_spotting_router_minimal.jsonl` | `59.14%` | `1.33%` | Not recommended |

## Final Conclusion

The best practical recommendation is:

1. Default to remote PaddleOCR-VL with `promptLabel="spotting"`.
2. Keep `promptLabel="table"` as a manual override for clearly calculation-heavy questions.
3. Do not enable automatic promptLabel routing by default.

Why:

- `spotting` is the strongest stable single-mode run on the 301-set.
- `table` helps most on `Calculation_analysis`, but hurts enough other categories that it should not be the default.
- Rule-based routing became more complex without surpassing pure `spotting`.
- `seal` and `formula` are not good production defaults because the gain does not justify the rate-limit cost.

## Final Validation On Current Code

After the recommendation was baked into the codebase, the final full 301 validation runs were:

| Mode | Output | Accuracy | Invalid Rate |
|---|---|---:|---:|
| Recommended default `spotting` | `runs/pyfi_eval_301_recommended_spotting.jsonl` | `62.46%` | `0.66%` |
| Manual `table` override | `runs/pyfi_eval_301_recommended_table.jsonl` | `61.13%` | `2.33%` |

The recommendation remains the same after validation:

- Use `spotting` by default.
- Keep `table` as a manual override only.

## Practical Recommendation

### Recommended default

```powershell
python -m finvl_eval.runner `
  --dataset data/pyfi/pyfi_eval_301.jsonl `
  --format jsonl `
  --images-root data/pyfi `
  --model remote-paddleocr-vl-ernie-docqa `
  --selector-model ernie-4.5-turbo-128k-preview `
  --selector-base-url https://aistudio.baidu.com/llm/lmapi/v3 `
  --artifacts-dir runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting_artifacts `
  --out runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_spotting.jsonl
```

### Manual calculation-heavy override

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
  --out runs/pyfi_eval_301_pure_remote_paddleocrvl_ernie45_table.jsonl
```
