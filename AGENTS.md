# Repository Guidelines

## 项目结构与模块组织

- `hqqsvd/`：核心 Python 包，包含量化、解量化与 Triton 加速相关实现（例如 `quantize.py`、`triton_mm.py`）。
- `scripts/`：开发/性能脚本，例如 `bench_dequantize.py`。
- `pyproject.toml`：项目元数据与依赖（当前依赖 `torch`）。
- 目前仓库内未提供测试目录与静态资源目录。

## 构建、测试与本地运行命令

- 本仓库使用 conda 环境：`DiffSynth-Studio`。
- PowerShell 7 中请通过 `cmd` 激活环境（不要直接 `conda activate`）：

```powershell
cmd.exe /d /c "chcp 65001>nul & call "%USERPROFILE%\anaconda3\Scripts\activate.bat" DiffSynth-Studio ^
& pwsh -NoLogo" ^
& set "PYTHONUTF8=1" ^
& set "PYTHONIOENCODING=utf-8"
```

- 可选安装（可编辑）：`python -m pip install -e .`
- 示例运行脚本：`python .\scripts\bench_dequantize.py`

## 编码风格与命名约定

- Python 4 空格缩进，类名 `PascalCase`，函数/变量 `snake_case`。
- 模块名小写、下划线分隔；保持 import 顺序简洁。
- 若新增 Triton/torch 相关代码，优先与现有模块保持一致风格。

## 测试指南

- 当前仓库未包含测试框架与用例。
- 若新增测试，建议使用 `pytest`，文件命名 `test_*.py`，并在文档中补充运行方式。

## 提交与拉取请求指南

- 现有提交多为简短祈使句（例如 `add ...`、`fix ...`、`enable ...`）。建议保持短小明确、聚焦变更点。
- PR 建议包含：变更摘要、复现/运行命令、性能或数值影响（如有）。

## 代理与环境注意事项

- 运行环境为 Windows 11 + PowerShell 7，只使用 PowerShell 命令（如 `Get-ChildItem`、`Select-String`）。
- 避免使用 Bash 语法与命令（如 `grep`、`sed`、`find -name`）。
