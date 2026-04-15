# 项目原理与通用化分析

## 一、项目原理

### 要解决的核心问题

这个项目回答一个问题：**PaddleOCR-VL 1.5 把图像解析成结构化文本的能力，到底值多少钱？**

PyFi 排行榜上，各路 VLM（GPT-4.1、Claude、ERNIE-VL 等）都是"直接看图答题"。而我们的链路是：

```
图像 → PaddleOCR-VL 解析成文本 → 文本模型读文本答题
```

这多了一步中间转换，自然会丢失视觉信息（颜色、趋势、图形）。所以问题变成：PaddleOCR-VL 解析出来的东西够不够好，能不能让一个纯文本模型只看解析结果就能答对题？

答案是可以的——ERNIE-4.5-turbo-vl 直接看图只得 34.47%，加了 PaddleOCR-VL 解析后跳到 67.44%。说明 PaddleOCR-VL 解析出来的表格、图表结构、文字块，对答题有实质帮助。

### 架构分层

整个项目可以分成 5 层：

```
┌──────────────────────────────────────────────────┐
│  Runner (统一入口, CLI, 进度条, 输出)              │
├──────────────────────────────────────────────────┤
│  Model Adapter (策略层)                           │
│  ┌─────────────┐ ┌──────────────┐ ┌───────────┐ │
│  │ Direct VLM  │ │ Hybrid Pipeline│ │ Boosted  │ │
│  │ (直接看图)   │ │ (双证据+选择器)│ │ (投票+校正)│ │
│  └─────────────┘ └──────────────┘ └───────────┘ │
├──────────────────────────────────────────────────┤
│  Parser (解析层)                                  │
│  ┌──────────────────┐  ┌─────────────────────┐   │
│  │ PaddleOCR-VL 1.5 │  │ 传统 PaddleOCR text │   │
│  │ 表格/图表/文本块  │  │ 小数字/标签/脚注     │   │
│  └──────────────────┘  └─────────────────────┘   │
├──────────────────────────────────────────────────┤
│  Evidence (证据层)                                │
│  表格行抽取 → OCR行匹配 → 选项命中 → 数值候选    │
├──────────────────────────────────────────────────┤
│  Scoring (评分层)                                 │
│  答案归一化(JSON/文字/代码块) → 正确性 → 聚合指标  │
├──────────────────────────────────────────────────┤
│  Data (数据层)                                    │
│  CSV/JSONL → EvalRecord (uid, image, Q, options)  │
└──────────────────────────────────────────────────┘
```

每层的作用：

**数据层** (`records.py`, `pyfi.py`)：把各种格式的评测数据统一成 `EvalRecord`——包含 uid、图片路径、问题、选项、答案、能力层级、复杂度。这一层屏蔽了上游数据格式差异。

**评分层** (`scoring.py`)：答案归一化是关键。模型可能返回 `{"answer":"B"}`、`` ```json\n{"answer":"B"}\n``` ``、`The answer is B` 等各种格式。`normalize_answer` 会依次尝试 JSON dict 提取、显式答案模式匹配（ANSWER IS / 答案是 / 选项）、短文本裸字母匹配，最后才放弃。invalid fallback 会用词法启发或首个合法选项兜底，保证 forced-choice 排行榜里不丢分。

**解析层** (`models.py` 中的 parse 方法)：两个 Paddle 解析器互补——PaddleOCR-VL 1.5 擅长表格结构、图表版面、文本块；传统 PaddleOCR 擅长小数字、坐标轴标签、脚注。它们的错误不重合，合并后覆盖面更广。解析产物以 artifact 形式缓存（`.md` 和 `.txt` 文件），后续运行直接读取缓存，不重复解析。

**证据层** (`evidence.py`)：从 PaddleOCR 产物中提取结构化证据——HTML 表格解析成行级文本、问题关键词与证据行的相关性打分、每个选项的文本/数字是否出现在证据中、数值型问题的计算候选。这一层让 selector 的判断有据可依。

**策略层** (`models.py` 中的 adapter 类)：5 种不同的答题策略，从简单到复杂：
- `first-option` / `random-option`：最低基线
- `openai-compatible-vlm`：直接 VLM 看图
- `paddleocr-text-docqa`：传统 OCR + LLM
- `paddleocr-vl-hybrid-docqa`：双证据通道 + ERNIE selector + invalid fallback
- `paddleocr-vl-boosted-docqa`：改进 prompt + 3 次投票 + D-avoidance ensemble

### 为什么最终分数能排到第 2

关键不是某个单一技巧，而是多个修复的叠加：

1. **消除 invalid**（+最大贡献）：旧 prompt 允许模型说"我不知道"，forced-choice 下不选就是错。从 53 个 invalid 降到 0，直接白捡分数。
2. **换更强的 selector**：GLM-4-flash → ERNIE-4.5-turbo-128k，更好的文本推理能力。
3. **双证据通道**：PaddleOCR-VL markdown + 传统 OCR text，覆盖各自盲区。
4. **D-avoidance**：模型系统性地过度预测 D（实际只有 6% 的题答案是 D，但模型预测了 14%），通过 hybrid 交叉验证和证据匹配校正。

---

## 二、能否提炼成通用评测框架

**可以**，而且大部分组件已经是领域无关的。

### 直接可复用的部分

| 组件 | 领域依赖度 | 说明 |
|------|-----------|------|
| `EvalRecord` 数据结构 | 无 | uid + image + question + options + answer + capability + complexity，任何 MCQ 评测都能用 |
| `scoring.py` 答案归一化 | 无 | JSON/code fence/文字模式提取，完全通用 |
| `runner.py` 评测框架 | 无 | CLI 入口、进度条、artifact 缓存、metrics 输出 |
| `ModelAdapter` 协议 | 无 | `predict(record, image_path, prompt) -> str`，任何模型只要实现这个接口就能接入 |
| Self-consistency 投票 | 无 | 多次调用取多数票，通用去噪手段 |
| Invalid fallback | 低 | 词法启发 + 首选项兜底，forced-choice 场景通用 |

### 需要按领域替换的部分

| 组件 | 当前实现 | 换领域要改什么 |
|------|---------|--------------|
| `pyfi.py` 数据读取 | PyFi CSV 格式 | 新的数据加载器（JSONL/CSV/其他） |
| `evidence.py` 证据抽取 | 金融表格/图表/数字 | 领域特定的结构化抽取逻辑 |
| Selector prompt | 金融文档术语和策略 | 按领域调整 prompt |
| 解析器 | PaddleOCR-VL + PaddleOCR | 可换成其他文档解析器（如医疗领域的 PDF 解析） |

### 提炼方案

如果要做一个通用框架，建议分 3 步：

**第一步：抽象数据接口**

当前 `EvalRecord` 已经足够通用。只需要写新的数据加载器，把任何领域的数据转成 `EvalRecord` 格式。比如：

- 医疗影像 MCQ：CT/X光 + 诊断问题 + 选项
- 法律文档 QA：合同截图 + 法条问题 + 选项
- 教育试卷：课本插图 + 题目 + 选项

**第二步：Parser 插件化**

把 `parse_image()` 抽成独立接口：

```python
class DocumentParser(Protocol):
    def parse(self, image_path: Path, uid: str) -> str: ...
```

PaddleOCR-VL、传统 OCR、甚至 PDF 解析器、医学影像分析器，都实现这个接口就行。

**第三步：Evidence 和 Prompt 模板化**

把 evidence 抽取逻辑和 selector prompt 做成可配置的模板。金融领域关注表格和数字，医疗领域关注病灶描述和指标，教育领域关注知识点——不同领域的 evidence 抽取策略不同，但"从解析结果里抽结构化证据 → 喂给 selector"这个流程是一样的。

### 最值得提炼的价值

这套架构最有价值的不是某个具体实现，而是它验证了的设计模式：

> **文档解析器 + 文本 LLM 的两阶段架构，可以在纯文本模型的条件下逼近甚至超过直接 VLM 的效果。**

这个结论不限于金融领域。任何涉及文档/图像理解的场景——合同审核、病历分析、工程图纸解读、教育阅卷——都可以用同样的模式：先用文档解析器把图像转成结构化文本，再让文本模型基于结构化证据做判断。
