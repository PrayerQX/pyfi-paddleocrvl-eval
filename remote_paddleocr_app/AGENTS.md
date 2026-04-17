# AGENTS.md

本目录是独立的远程 PaddleOCR + 文心一言 API 项目。Claude Code 在这里工作时必须遵守以下规则。

## Python 环境

- 必须使用 `uv` 管理本地 Python 环境。
- 安装依赖使用：
  ```powershell
  uv sync
  ```
- 运行命令使用：
  ```powershell
  uv run remote-paddleocr --help
  uv run python -m pytest
  ```
- 不要使用 `pip install ...` 直接改全局环境。
- 不要提交 `.venv/`、`.env`、输出目录或真实 API token。

## API 使用

- PaddleOCR 远程解析只调用 `PADDLEOCR_API_URL` 指向的 HTTP API。
- 文心一言只通过 OpenAI-compatible `chat.completions` API 调用。
- 真实密钥必须从环境变量或本地 `.env` 读取：
  - `PADDLEOCR_API_TOKEN`
  - `ERNIE_API_KEY`
  - `ERNIE_BASE_URL`
  - `ERNIE_MODEL`
- 代码、测试、文档示例中只能出现占位符，不能硬编码真实 token。

## 项目边界

- 本目录应保持为可独立运行的小项目，不依赖仓库根目录的 `finvl_eval` 包。
- 默认输出写入 `output/` 或 CLI 指定目录。
- 修改功能时优先保持 CLI 向后兼容。
- 外部网络调用需要有清晰错误信息，不能静默失败。

## 常用命令

```powershell
cd remote_paddleocr_app
uv sync

uv run remote-paddleocr parse path\to\image.jpg --file-type image --out output

uv run remote-paddleocr ask path\to\image.jpg `
  --file-type image `
  --question "这张图表的核心结论是什么？" `
  --out output
```
