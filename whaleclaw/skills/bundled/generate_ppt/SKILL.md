---
triggers:
  - PPT
  - pptx
  - 幻灯片
  - 演示文稿
max_tokens: 2200
lock_session: false
---

# 生成 PPT (python-pptx)

## 流程

1. 回复用户：几页、什么内容
2. `browser` → 搜图（风景搜"[地名] 风景 高清"，人物搜"[人名] 写真"），封面选横版
3. `file_write` → 完整脚本到 `~/.whaleclaw/workspace/tmp/gen_ppt_xxx.py`
4. `bash` → 执行
5. 告诉用户路径

复刻：截图→vision提取配色→确认→搜图→写脚本；.pptx→bash提取颜色字体→写脚本。
严禁：`python -c`；分多次file_write；图片路径硬编码

## 基础

```python
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image as PILImage
prs = Presentation()
prs.slide_width = Inches(13.333); prs.slide_height = Inches(7.5)
SW, SH = prs.slide_width, prs.slide_height
```

配色7变量全文统一：PRIMARY/SECONDARY/ACCENT/BG_LIGHT/TEXT_DARK/TEXT_LIGHT/TEXT_GRAY
字体：标题"Microsoft YaHei"，英文"Arial Black"/"Arial"
字号(Pt)：HERO=44, H1=32, H2=24, BODY=16, CAPTION=12, NUMBER=56

## 版式守恒

- 1页1点，超量拆页不缩字号；标题≤18字，要点≤32字/条，3-5条/页
- 最小字号：HERO≥36, H1≥28, H2≥22, BODY≥15, CAPTION≥11
- 同类页1套字号；安全区左右≥0.6in上下≥0.5in；小改优先`ppt_edit`

## 辅助函数

```python
def add_picture_cropped(slide, img_path, left, top, tw, th):
    from whaleclaw.utils.image_crop import detect_face_info, smart_crop_box
    with PILImage.open(img_path) as im: iw, ih = im.size
    img_r, box_r = iw/ih, tw/th
    if img_r > box_r: sw,sh = int(th*img_r), int(th)
    else: sw,sh = int(tw), int(tw/img_r)
    pic = slide.shapes.add_picture(img_path, int(left), int(top), sw, sh)
    fi = detect_face_info(img_path)
    x0,y0,x1,y1 = smart_crop_box(iw, ih, tw, th, face_info=fi)
    pic.crop_left=x0/iw; pic.crop_right=1-x1/iw
    pic.crop_top=y0/ih; pic.crop_bottom=1-y1/ih

def add_rect(slide, l, t, w, h, color, alpha=1.0):
    s = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, int(l), int(t), int(w), int(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    if alpha < 1.0:
        from pptx.oxml.ns import qn; from lxml import etree
        sf = s.fill._fill.find(qn("a:solidFill"))
        if sf is not None and len(sf):
            etree.SubElement(sf[0], qn("a:alpha")).set("val", str(int(alpha*100000)))
    return s

def add_tb(slide, l, t, w, h, text, sz, color, bold=False, fn="Microsoft YaHei", align=PP_ALIGN.LEFT):
    from pptx.enum.text import MSO_AUTO_SIZE
    tb = slide.shapes.add_textbox(int(l), int(t), int(w), int(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left=tf.margin_right=Pt(2); tf.margin_top=tf.margin_bottom=0
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    r = p.runs[0]; r.font.size=sz; r.font.color.rgb=color; r.font.bold=bold; r.font.name=fn
    p.space_before=p.space_after=0; p.line_spacing=1.15
    return tb
```

add_tb的h须留余量：单行≥字号×1.5，多行≥行数×字号×1.3。放不下减条或拆页。

## 模板

### 1A 全图封面（禁止全屏纯色矩形）

```python
def make_cover(prs, title, subtitle, img):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_picture_cropped(s, img, 0, 0, SW, SH)
    add_rect(s, 0, int(SH*0.55), SW, SH-int(SH*0.55), PRIMARY, 0.75)
    add_tb(s, Inches(1), int(SH*0.60), Inches(11), Inches(1.2), title, SIZE_HERO, TEXT_LIGHT, True)
    add_tb(s, Inches(1), int(SH*0.78), Inches(9), Inches(0.6), subtitle, SIZE_BODY, RGBColor(0xCC,0xCC,0xCC))
```

### 1B 左右分栏封面：左42%纯色+标题，右58%图片。图片最后添加。

### 2 左图右文（图片最后添加）

```python
def make_left_img(prs, title, bullets, img):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG_LIGHT
    add_tb(s, Inches(6.8), Inches(0.8), Inches(5.8), Inches(1), title, SIZE_H1, PRIMARY, True)
    ln = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(6.8), Inches(1.7), Inches(1.5), Pt(4))
    ln.fill.solid(); ln.fill.fore_color.rgb = ACCENT; ln.line.fill.background()
    ys, ye = Inches(2.1), Inches(6.8)
    sp = min((ye-ys)/max(len(bullets),1), Inches(1.15))
    for i, b in enumerate(bullets):
        y = int(ys + sp*i)
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(6.8), int(y+Pt(4)), Pt(10), Pt(10))
        d.fill.solid(); d.fill.fore_color.rgb = SECONDARY; d.line.fill.background()
        add_tb(s, Inches(7.15), y, Inches(5.3), int(sp-Pt(4)), b, SIZE_BODY, TEXT_DARK)
    add_picture_cropped(s, img, Inches(0.6), Inches(0.6), Inches(5.6), Inches(6.3))
```

### 3 右图左文：模板2镜像，文字x=0.8，图片x=7.1。

### 4 数据页（禁止图片）

```python
def make_data(prs, title, items):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    s.background.fill.solid(); s.background.fill.fore_color.rgb = BG_LIGHT
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SW, Inches(1.4))
    bar.fill.solid(); bar.fill.fore_color.rgb = PRIMARY; bar.line.fill.background()
    add_tb(s, Inches(0.8), Inches(0.25), Inches(11), Inches(0.9), title, SIZE_H1, TEXT_LIGHT, True)
    n = min(len(items), 4); cw, gap = Inches(2.8), Inches(0.5)
    sx = int((SW - n*cw - (n-1)*gap)/2)
    for i, (num, lbl) in enumerate(items[:4]):
        cx = int(sx + i*(cw+gap))
        add_rect(s, cx, Inches(2), cw, Inches(3.2), RGBColor(0xFF,0xFF,0xFF))
        add_tb(s, cx+Inches(0.2), int(Inches(2.5)), int(cw-Inches(0.4)), Inches(1.2), num, SIZE_NUMBER, ACCENT, True, align=PP_ALIGN.CENTER)
        add_tb(s, cx+Inches(0.2), int(Inches(3.7)), int(cw-Inches(0.4)), Inches(0.6), lbl, SIZE_BODY, TEXT_DARK, align=PP_ALIGN.CENTER)
```

### 5 结尾页：全屏图+遮罩或纯色，居中标题。
### 6 过渡页（10页+必须有）：纯色，左大编号Pt120，右标题。
### 7 上图下文：上55%图，下标题+≤3列要点。图片最后添加。
### 8 对比页：两栏+中间竖线。

## 组合

- **5页**：封面→左→右→左→结尾
- **7页**：封面→左→右→数据→左→右→结尾
- **10页**：封面→左→过渡→右→上图下文→过渡→数据→左→对比→结尾
- **12+**：封面→[过渡→2~3内容]×N→数据→结尾

## 铁律

- **Z-order**：图片最后添加
- **所有图片用add_picture_cropped**，严禁add_picture同时指定宽高
- **add_picture_cropped后禁止设pic.width/pic.height**
- **封面遮罩≤45%半透明；数据页禁止图片**
- **要点≥3条自适应间距；内容具体带数字；配色全文统一**
- **同模板页尺寸一致；复改优先局部改**
