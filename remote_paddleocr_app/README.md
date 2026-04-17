# Remote PaddleOCR App

这是一个独立的远程 PaddleOCR + 文心一言 API 小项目。它不在本地加载 PaddleOCR 模型，所有文档解析都走远程 PaddleOCR API；文档问答和总结走文心一言 OpenAI-compatible API。

## 环境要求

本项目必须使用 `uv` 管理 Python 环境。

```powershell
cd remote_paddleocr_app
uv sync
```

复制环境变量模板：

```powershell
Copy-Item .env.example .env
```

然后在 `.env` 中填写本地 token。不要提交 `.env`。

必需变量：

```text
PADDLEOCR_API_URL=https://i0u1edb895ael4d6.aistudio-app.com/layout-parsing
PADDLEOCR_API_TOKEN=...
ERNIE_API_KEY=...
ERNIE_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
ERNIE_MODEL=ernie-5.0-thinking-preview
```

## 解析文档

图片：

```powershell
uv run remote-paddleocr parse path\to\image.jpg --file-type image --out output
```

图表类图片建议打开远程图表识别：

```powershell
uv run remote-paddleocr parse path\to\chart.jpg `
  --file-type image `
  --use-chart-recognition `
  --out output
```

PDF：

```powershell
uv run remote-paddleocr parse path\to\doc.pdf --file-type pdf --out output
```

输出内容：

- `result.json`：PaddleOCR API 原始响应中的 `result`
- `doc_0.md`、`doc_1.md`：每页或每段 layout parsing 的 Markdown
- `markdown_images/` 和 `output_images/`：API 返回的图片资源

## 文档问答

```powershell
uv run remote-paddleocr ask path\to\image.jpg `
  --file-type image `
  --use-chart-recognition `
  --question "这张图表说明了什么？" `
  --out output
```

多选题可以传选项：

```powershell
uv run remote-paddleocr ask path\to\image.jpg `
  --file-type image `
  --question "哪一个选项最符合图表？" `
  --option A="收入上升" `
  --option B="收入下降" `
  --option C="没有变化" `
  --out output
```

`ask` 会先调用 PaddleOCR 远程解析，把 Markdown 证据交给文心一言，再输出答案。

默认只输出文心最终答案。调试 thinking 模型时可以追加 `--include-reasoning`，把 `reasoning_content` 一起写入输出。

## 只调用文心

```powershell
uv run remote-paddleocr chat "解释一下资产负债表的核心指标"
```

## 运行测试

```powershell
uv run python -m pytest
```

## 安全说明

真实 token 只允许放在本地环境变量或 `.env`。代码和文档不会提交真实 token。
