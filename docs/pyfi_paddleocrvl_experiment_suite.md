# PyFi PaddleOCR-VL 四组实验执行方案

本文档把四组实验落成仓库里的可执行入口。目标不是一次性混合所有技巧，而是按成本和风险逐层判断 PaddleOCR-VL 在 PyFi 上的能力边界。

## 总原则

- 不使用 gold answer 参与推理、prompt、veto 或 evidence builder。
- 每个实验内对所有样本使用同一条链路，不按 capability 切换 prompt。
- 不做 option-letter prior、first-option fallback、stacking 或测试集校准。
- `data/pyfi/pyfi_eval_301.jsonl` 只作为本地可复用评测 split，不作为训练数据。
- 报告必须写清楚 dataset path、hash、model、prompt style、artifact path、output path 和 metrics。

## 实验 1：直接零样本跑 PyFi

目的：最快判断 PaddleOCR-VL 作为直接图像问答模型的原生底子。

实现：使用 `openai-compatible-vlm` adapter，把图像、题目、选项直接发给一个 PaddleOCR-VL 兼容的 VLM endpoint。prompt 使用 `--prompt-style direct`，只要求输出最终选项。

配置：`configs/pyfi_exp1_direct_zeroshot_paddleocrvl.json`

适合观察：Overall、Data_extraction、Calculation_analysis、Logical_reasoning。

失败点：如果 endpoint 实际只支持文档解析而不支持 VQA，这组不能代表 PaddleOCR-VL 的直接问答能力，需要改为实验 3 的解析底座路线。

## 实验 2：金字塔式分步提示

目的：验证 PaddleOCR-VL 的短板是否来自 PyFi 风格的推理节奏没有被激活。

实现：仍使用直接 VLM endpoint，但 prompt 切换为 `--prompt-style pyramid`。固定流程是：先识别图表元素，再抽取相关证据，再比较/计算/归纳，最后输出答案。

配置：`configs/pyfi_exp2_pyramid_prompt_paddleocrvl.json`

成功信号：相比实验 1，Calculation_analysis、Pattern_recognition、Logical_reasoning 有明显提升，同时 invalid 不上升。

失败点：模型可能把分步提示当成解释任务，导致输出格式变差；评分时仍按严格最终选项归一化。

## 实验 3：PaddleOCR-VL 结构底座 + 两阶段推理

目的：最大化 PaddleOCR-VL 的文档解析、表格、图表和结构感知优势。

实现：新增 `paddleocr-vl-structured-docqa` adapter。

阶段 A：PaddleOCR-VL 解析图像为 markdown，再统一构造结构化中间表示：

- question keywords / numbers
- option evidence
- numeric candidates
- relevant PaddleOCR-VL rows
- raw PaddleOCR-VL markdown excerpt

阶段 B：把题目、选项、图像背景和结构化中间表示交给 selector，输出严格 JSON 答案。

配置：`configs/pyfi_exp3_structured_two_stage_paddleocrvl.json`

适合观察：Calculation_analysis、Pattern_recognition、Logical_reasoning 是否相对实验 2 同步改善。

失败点：如果 PaddleOCR-VL markdown 没有恢复颜色、线型、图例对应关系，Perception 仍可能受限。这时需要另开“图像 VLM selector”实验口径，不应和纯 OCR 证据链路混报。

## 实验 4：PyFi chain LoRA / SFT 数据准备

目的：为后续 PaddleOCR-VL 版 PyFi-CoT 训练准备数据格式。

实现：新增命令 `finvl-prepare-pyfi-sft` / `python -m finvl_eval.prepare_pyfi_sft`。

支持两种数据：

- `final-only`：只训练最终问答。
- `cot-from-metadata`：从记录 metadata 中读取 chain/COT 字段，生成分步推理 + final answer。

配置：`configs/pyfi_exp4_lora_sft_paddleocrvl.json`

安全限制：脚本默认拒绝把名字包含 `eval` 或 `301` 的 split 转成训练数据；只有 smoke test 才允许显式传 `--allow-eval-split`。

失败点：当前本地 301 split 不包含完整 PyFi 47K chain，不能凭空合成 CoT 训练集。真正的 CoT 训练需要准备带 chain metadata 的训练 JSONL。

## 推荐执行顺序

1. 实验 1：确认直接 VQA 底线。
2. 实验 2：确认 prompt 是否能激活 PyFi 风格推理。
3. 实验 3：确认 PaddleOCR-VL 作为结构底座是否优于直接问答。
4. 实验 4：只有前三步证明值得继续时，再投入 LoRA/SFT。

如果只做两组，优先做实验 2 和实验 3，因为它们最能判断 PaddleOCR-VL 是更适合“直接答题”，还是更适合“先解析再推理”。
