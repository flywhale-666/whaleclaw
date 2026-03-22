---
triggers:
  - Word
  - docx
  - 文档
  - 报告
  - 简历
  - resume
  - CV
  - 方案
  - 手册
max_tokens: 2200
lock_session: false
---

# 生成 Word (python-docx)

## 流程

1. 先回复用户：说明文档结构
2. 如需图片：`browser` → 搜图
3. `file_write` → 完整脚本到 `~/.whaleclaw/workspace/tmp/gen_doc_xxx.py`
4. `bash` → 执行（Win: `.\python\python.exe`）
5. 告诉用户路径

严禁：不用 `python -c`；不分多次 file_write；图片路径硬编码绝对路径

## 基础设置

```python
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from lxml import etree

doc = Document()
sec = doc.sections[0]
sec.page_width=Cm(21); sec.page_height=Cm(29.7)
sec.left_margin=Cm(2.5); sec.right_margin=Cm(2.5)
sec.top_margin=Cm(2); sec.bottom_margin=Cm(2)

style = doc.styles['Normal']
style.font.name='Microsoft YaHei'; style.font.size=Pt(11)
style.font.color.rgb=RGBColor(0x33,0x33,0x33)
style.paragraph_format.line_spacing=1.5
style._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

for lv, (sz, clr, sp) in enumerate([
    (Pt(22), RGBColor(0x0F,0x27,0x47), Pt(24)),
    (Pt(16), RGBColor(0x1A,0x73,0xE8), Pt(18)),
    (Pt(13), RGBColor(0x2D,0x3A,0x3A), Pt(12)),
], start=1):
    h=doc.styles[f'Heading {lv}']
    h.font.name='Microsoft YaHei'; h.font.size=sz; h.font.color.rgb=clr; h.font.bold=True
    h.paragraph_format.space_before=sp
    h._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')

def set_run_font(run, size=None, bold=None, color=None, name='Microsoft YaHei'):
    run.font.name = name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), name)
    if size is not None: run.font.size = size
    if bold is not None: run.font.bold = bold
    if color is not None: run.font.color.rgb = color
```

## 版式守恒

- 正文11pt/1.5倍行距，最小10.5pt；H1/H2/H3固定22/16/13pt
- 段前后距统一，不用空段落撑排版；单段40-120字，超长拆标题/列表/表格
- 小改优先`docx_edit`，不整份重生成

## 封面页

```python
def add_cover(doc, title, subtitle="", author="", date_text=""):
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_before=Pt(120)
    p.add_run("━"*30).font.color.rgb=RGBColor(0x1A,0x73,0xE8)
    pt=doc.add_paragraph(); pt.alignment=WD_ALIGN_PARAGRAPH.CENTER
    r=pt.add_run(title); r.font.size=Pt(28); r.font.bold=True
    r.font.color.rgb=RGBColor(0x0F,0x27,0x47); r.font.name='Microsoft YaHei'
    r._element.rPr.rFonts.set(qn('w:eastAsia'), 'Microsoft YaHei')
    if subtitle:
        ps=doc.add_paragraph(); ps.alignment=WD_ALIGN_PARAGRAPH.CENTER
        ps.add_run(subtitle).font.color.rgb=RGBColor(0x66,0x66,0x66)
    p2=doc.add_paragraph(); p2.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p2.add_run("━"*30).font.color.rgb=RGBColor(0x1A,0x73,0xE8)
    if author or date_text:
        pi=doc.add_paragraph(); pi.alignment=WD_ALIGN_PARAGRAPH.CENTER; pi.paragraph_format.space_before=Pt(48)
        pi.add_run("  |  ".join(filter(None,[f"作者：{author}" if author else "",date_text]))).font.color.rgb=RGBColor(0x99,0x99,0x99)
    doc.add_page_break()
```

## 目录

TOC域代码：fldChar begin → instrText `TOC \o "1-3" \h \z \u` → fldChar separate → 提示 → fldChar end → 分页。

## 页眉页脚

页眉右对齐文档标题(Pt9灰色)，页脚居中"第 PAGE 页"域代码。

## 表格

```python
def add_table(doc, headers, rows):
    t=doc.add_table(rows=1+len(rows), cols=len(headers))
    t.alignment=WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    for i,h in enumerate(headers):
        c=t.rows[0].cells[i]; c.text=h
        r=c.paragraphs[0].runs[0]; r.font.bold=True; r.font.color.rgb=RGBColor(0xFF,0xFF,0xFF)
        etree.SubElement(c._tc.get_or_add_tcPr(), qn('w:shd')).set(qn('w:fill'), '0F2747')
    for ri,rd in enumerate(rows):
        for ci,v in enumerate(rd):
            c=t.rows[ri+1].cells[ci]; c.text=str(v)
            if ri%2==0:
                etree.SubElement(c._tc.get_or_add_tcPr(), qn('w:shd')).set(qn('w:fill'), 'F4F7FB')
```

表格必须固定列宽，不依赖Word自动缩放。

## 图片（不变形）

```python
def add_doc_image(doc, img_path, width=Inches(6), ratio=16/9):
    from whaleclaw.utils.image_crop import detect_face_info, smart_crop_box
    with PILImage.open(img_path) as im:
        iw, ih = im.size
        tw, th = iw, int(iw/ratio)
        if th > ih: tw, th = int(ih*ratio), ih
        fi = detect_face_info(img_path)
        x0,y0,x1,y1 = smart_crop_box(iw, ih, tw, th, face_info=fi)
        cropped = im.crop((x0,y0,x1,y1))
        tmp = img_path.rsplit('.',1)[0]+'_crop.jpg'
        cropped.save(tmp, quality=92)
    p = doc.add_picture(tmp, width=width)
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
```

人像自动人脸感知裁剪，非人像居中裁剪。只指定width，python-docx自动保持比例。

## 引用块

左缩进1.5cm，pBdr/w:left蓝色(1A73E8)色条(sz=24)。

---

## 简历模板（"简历"/"resume"/"CV"触发）

不需要封面/目录/页眉页脚，窄边距(1.8cm)，紧凑1-2页。

```python
doc = Document()
sec=doc.sections[0]
sec.left_margin=Cm(1.8); sec.right_margin=Cm(1.8)
sec.top_margin=Cm(1.5); sec.bottom_margin=Cm(1.5)

C1=RGBColor(0x1A,0x56,0x8E)  # 深蓝
C2=RGBColor(0x2E,0x86,0xC1)  # 亮蓝
CT=RGBColor(0x2C,0x2C,0x2C)  # 正文
CG=RGBColor(0x77,0x77,0x77)  # 灰

def add_sec(doc, title):
    # 分隔线: pBdr/w:bottom single sz=6 color=D5DBE1
    p=doc.add_paragraph()
    p.add_run("■ ").font.color.rgb=C2
    r=p.add_run(title); r.font.size=Pt(13); r.font.bold=True; r.font.color.rgb=C1

def add_exp(doc, org, role, period, details):
    # 无边框2列表格: 左=公司(bold), 右=时间(右对齐灰色)
    # 下方: 职位(蓝色斜体)
    # 要点: List Bullet, 每条Pt10, 左缩进0.5cm
```

**简历顺序**：姓名(Pt26)+联系方式→求职意向→经历(倒序)→项目→教育→技能→证书
**要求**：经历"动词+量化结果"；技能具体到工具名；1-2页

---

## 普通文档结构

封面→目录→页眉页脚→正文(H1/H2/H3)→表格→图片(居中带图注)→总结

- 图片统一宽度(5.8in或6.2in)，图注10pt灰色居中
- 列表用Word原生样式；中文统一微软雅黑不混字体
