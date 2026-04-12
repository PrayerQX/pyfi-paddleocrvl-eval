# PyFi Repo Leaderboard Comparison

Run date: 2026-04-12

## Scope

This compares the local PaddleOCR-VL 1.5 chain with the model table published in the PyFi README.

Important caveat: the PyFi README reports a 301-sample evaluation with image groups, but the repository does not include a reusable official split file. The PaddleOCR-VL 1.5 row below is from our local stratified 301-row split, so this is a rough cross-check, not a formal leaderboard submission.

Local PaddleOCR-VL 1.5 setting:

| Model | Total | Correct | Accuracy | Invalid | Invalid Rate |
|---|---:|---:|---:|---:|---:|
| PaddleOCR-VL 1.5 + GLM glm-4-flash selector | 301 | 139 | 46.18% | 53 | 17.61% |

Capability accuracy for the local row:

| Model | PP | DE | CA | PR | LR | DS |
|---|---:|---:|---:|---:|---:|---:|
| PaddleOCR-VL 1.5 + GLM glm-4-flash selector | 30.61% | 36.73% | 36.00% | 46.94% | 59.18% | 61.22% |

## Overall Ranking Check

`Delta vs PaddleOCR-VL` is `repo/model overall - 46.18`. Positive values mean the repo model is ahead of the local PaddleOCR-VL chain.

| Rank | Group | Model | Overall | Delta vs PaddleOCR-VL |
|---:|---|---|---:|---:|
| 1 | Pre-trained | GLM-4.5V | 74.75% | +28.57 |
| 2 | Pre-trained | Claude-opus-4-1-20250805 | 64.70% | +18.52 |
| 3 | Pre-trained | Hunyuan-Large-Vision | 59.72% | +13.54 |
| 4 | Pre-trained | Moonshot-V1-8k-Vision-Preview | 54.90% | +8.72 |
| 5 | Pre-trained | Moonshot-V1-128k-Vision-Preview | 54.57% | +8.39 |
| 6 | Pre-trained | Moonshot-V1-32k-Vision-Preview | 54.40% | +8.22 |
| 7 | Pre-trained | GPT-4.1 | 52.99% | +6.81 |
| 8 | Pre-trained | InternVL3-38B | 52.91% | +6.73 |
| 9 | Pre-trained | Qwen3-VL-Plus | 51.00% | +4.82 |
| 10 | Pre-trained | Qwen2.5-VL-72B-Instruct | 48.84% | +2.66 |
| 11 | Local | PaddleOCR-VL 1.5 + GLM glm-4-flash selector | 46.18% | 0.00 |
| 12 | Pre-trained | DeepSeek-VL2 | 45.18% | -1.00 |
| 13 | SFT 500 | PyFi-QwenVL-7B-500 | 43.44% | -2.74 |
| 14 | Pre-trained | Qwen2.5-VL-32B-Instruct | 43.19% | -2.99 |
| 15 | SFT 47K | PyFi-QwenVL-7B-COT-47K | 42.61% | -3.57 |
| 16 | SFT 47K | PyFi-QwenVL-3B-COT-47K | 40.37% | -5.81 |
| 17 | SFT 500 | PyFi-QwenVL-7B-COT-500 | 39.53% | -6.65 |
| 18 | Pre-trained | Qwen2.5-VL-7B-Instruct | 37.87% | -8.31 |
| 19 | SFT 47K | Qwen2.5-VL-7B-Instruct-L | 34.55% | -11.63 |
| 20 | SFT 500 | Qwen2.5-VL-7B-Instruct-L | 34.55% | -11.63 |
| 21 | Pre-trained | ERNIE-4.5-turbo-vl | 34.47% | -11.71 |
| 22 | SFT 500 | PyFi-QwenVL-3B-COT-500 | 27.66% | -18.52 |
| 23 | SFT 47K | PyFi-QwenVL-7B-47K | 27.08% | -19.10 |
| 24 | SFT 500 | PyFi-QwenVL-3B-500 | 26.74% | -19.44 |
| 25 | SFT 47K | PyFi-QwenVL-3B-47K | 25.25% | -20.93 |
| 26 | SFT 47K | Qwen2.5-VL-3B-Instruct-L | 20.85% | -25.33 |
| 27 | SFT 500 | Qwen2.5-VL-3B-Instruct-L | 20.85% | -25.33 |
| 28 | Pre-trained | Qwen2.5-VL-3B-Instruct | 20.51% | -25.67 |

## Nearest Comparisons

| Comparison | Overall Difference | Notes |
|---|---:|---|
| vs GLM-4.5V | -28.57 | Far behind the repo's strongest pre-trained VLM row. |
| vs Claude-opus-4-1-20250805 | -18.52 | Clearly behind. |
| vs GPT-4.1 | -6.81 | Behind, but closer than the top proprietary VLMs. |
| vs Qwen3-VL-Plus | -4.82 | Behind. |
| vs Qwen2.5-VL-72B-Instruct | -2.66 | Slightly behind. |
| vs DeepSeek-VL2 | +1.00 | Slightly ahead on overall accuracy. |
| vs PyFi-QwenVL-7B-500 | +2.74 | Ahead of the best 500-chain SFT row in the README. |
| vs PyFi-QwenVL-7B-COT-47K | +3.57 | Ahead of the best 47K-chain PyFi SFT row in the README. |

## Takeaway

Under rough numeric comparison, PaddleOCR-VL 1.5 + GLM selector lands in the middle: below the stronger pre-trained VLMs and Qwen2.5-VL-72B, but above DeepSeek-VL2, Qwen2.5-VL-32B, smaller Qwen rows, and all PyFi SFT rows published in the README.

The weak point remains invalid rate: the repo table reports only accuracy, while our local PaddleOCR-VL chain has 17.61% invalid answers. That makes the comparison less favorable than overall accuracy alone suggests.

## Sources

- PyFi README: https://github.com/AgenticFinLab/PyFi
- Local PaddleOCR-VL metrics: `runs/pyfi_paddleocrvl15_glm301_merged.jsonl.metrics.json`
