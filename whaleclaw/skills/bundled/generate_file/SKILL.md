---
triggers:
  - PDF
  - pdf
max_tokens: 2200
lock_session: false
---

# 生成文件（通用）

## 流程

1. 先回复用户：说明你打算做什么
2. 如需图片：`browser` → 搜图
3. `file_write` → 写完整 Python 脚本到 `~/.whaleclaw/workspace/tmp/gen_xxx.py`
4. `bash` → 执行（Windows: `.\python\python.exe`，Linux/Mac: `./python/bin/python3.12`）
5. 告诉用户文件路径

**严禁：** 不用 `python -c`；不分多次 file_write；图片路径硬编码绝对路径

## PDF (reportlab)

- 注册中文字体（SimSun 或 Microsoft YaHei）
- 边距：左右 2.5cm，上下 2cm
- 标题 22pt 加粗，小标题 14pt，正文 11pt
- 插图用 PIL 预裁剪到目标比例后 `drawImage()`
- 表格有表头背景色 + 边框

## 关键提醒

- 脚本必须完整可运行
- 不需要安装依赖（已预装 python-pptx / openpyxl / reportlab / python-docx / Pillow）
- 保存到 `~/.whaleclaw/workspace/tmp/<有意义的文件名>.<后缀>`
- 文件名仅保留中文/英文/数字/下划线
- 先定版式 token 预算，再写内容：标题/要点/表格都先压缩，避免靠临时缩字号硬塞
- 同一文件只允许 1 套主题字体 + 1 套字号层级，禁止每页/每段随意漂移
- 小改已有 Office 文件时优先走对应编辑工具，避免整份重生成导致样式跑偏
