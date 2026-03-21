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

1. 先回复用户：说明几页、什么内容
2. `browser` → 搜图（关键词具体：风景搜"[地名] 风景 高清"，人物搜"[人名] 写真"，商务搜"business professional"），封面图选横版
3. `file_write` → 完整脚本到 `~/.whaleclaw/workspace/tmp/gen_ppt_xxx.py`
4. `bash` → 执行（Win: `.\python\python.exe`）
5. 告诉用户路径

复刻模式：用户提供截图→vision提取配色布局→确认→搜图→写脚本；提供.pptx→bash提取颜色字体→自定义变量→写脚本。

严禁：不用 `python -c`；不分多次 file_write；图片路径硬编码绝对路径

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

## 配色（7变量全文统一）

定义 PRIMARY/SECONDARY/ACCENT/BG_LIGHT/TEXT_DARK/TEXT_LIGHT/TEXT_GRAY，按主题自动选色。
字体：标题 "Microsoft YaHei"，英文 "Arial Black"/"Arial"
字号(Pt)：HERO=44, H1=32, H2=24, BODY=16, CAPTION=12, NUMBER=56

## 版式守恒

- 先定模板再写文案；1页只表达1个核心点，超量内容拆新页，不靠缩字号补救
- 标题最多 18 字；副标题最多 28 字；单条要点最多 32 字；每页要点 3-5 条
- HERO 最小 36，H1 最小 28，H2 最小 22，BODY 最小 15，CAPTION 最小 11；触底仍放不下就删减文案或增页
- 同类页只允许 1 套字号，不因某一页内容变多就单独缩到更小
- 文本框高度先固定，再按内容反推页数；禁止写完后任意拉伸文本框破坏对齐
- 页边安全区：左右至少 0.6in，上下至少 0.5in；装饰线/编号也不能贴边
- 每页元素按网格对齐：左右页镜像时只换 x，不改单个元素尺寸和间距
- 修改已有 PPT：小改文案/颜色/图片优先 `ppt_edit`，不要整份重生成

## 辅助函数

```python
def add_picture_cropped(slide, img_path, left, top, tw, th):
    with PILImage.open(img_path) as im: iw, ih = im.size
    r1, r2 = iw/ih, tw/th
    if r1 > r2: sh = th; sw = th*r1
    else: sw = tw; sh = tw/r1
    pic = slide.shapes.add_picture(img_path, int(left-(sw-tw)/2), int(top-(sh-th)/2), int(sw), int(sh))
    clr, ctb = (sw-tw)/2/sw, (sh-th)/2/sh
    pic.crop_left=clr; pic.crop_right=clr; pic.crop_top=ctb; pic.crop_bottom=ctb
    pic.left=int(left); pic.top=int(top); pic.width=int(tw); pic.height=int(th)

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
    tb = slide.shapes.add_textbox(int(l), int(t), int(w), int(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.margin_left=tf.margin_right=Pt(2); tf.margin_top=tf.margin_bottom=0
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    r = p.runs[0]; r.font.size = sz; r.font.color.rgb = color; r.font.bold = bold; r.font.name = fn
    p.space_before = p.space_after = 0; p.line_spacing = 1.15
    return tb
```

多条要点不要每条单独估字号；统一用固定行高（BODY 对应 1.15-1.25 倍），放不下就减少要点或拆页。

## 模板

### 1A 全图封面
全屏图+底部半透明条(≤45%,alpha0.75)+标题。禁止全屏纯色矩形。

```python
def make_cover(prs, title, subtitle, img):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_picture_cropped(s, img, 0, 0, SW, SH)
    add_rect(s, 0, int(SH*0.55), SW, SH-int(SH*0.55), PRIMARY, 0.75)
    add_tb(s, Inches(1), int(SH*0.60), Inches(11), Inches(1.2), title, SIZE_HERO, TEXT_LIGHT, True)
    add_tb(s, Inches(1), int(SH*0.78), Inches(9), Inches(0.6), subtitle, SIZE_BODY, RGBColor(0xCC,0xCC,0xCC))
```

### 1B 左右分栏封面（商务/正式）
左42%纯色+标题装饰线，右58%图片。图片最后添加。

### 2 左图右文
左45%图，右55%标题+装饰线+自适应要点。**图片最后添加（Z-order铁律）。**

```python
def make_left_img(prs, title, bullets, img, pn=""):
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

文字区固定为：标题 1 行、副标题/导语 0-2 行、要点区从 `y=2.1` 到 `y=6.8`。要点间距先算总可用高，再均分，不允许最后一条贴底。

### 3 右图左文
模板2的镜像：左侧文字x=0.8，右侧图片x=7.1。其余结构相同。图片最后添加。

### 4 数据页（⚠️禁止图片）
顶部色条+标题，横排白色卡片(最多4个，含大数字+标签)。data_items: [(number, label), ...]

```python
def make_data(prs, title, items, pn=""):
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

### 5 结尾页
全屏图+半透明遮罩 或 纯色背景，居中标题+副标题。可选contact_info字典显示联系方式。

### 6 章节过渡页（10页以上必须有）
纯色背景，左侧大编号(Pt120)，右侧章节标题+描述。

### 7 上图下文
上55%大图，下方标题+多列要点(最多3列)。图片最后添加。

### 8 对比页
两栏对比，顶部色块标题+下方要点，中间竖线分隔。适合优缺点/方案对比。

## 组合

- **5页**：封面→左图右文→右图左文→左图右文→结尾
- **7页**：封面→左→右→数据→左→右→结尾
- **10页**：封面→左→过渡→右→上图下文→过渡→数据→左→对比→结尾
- **12+页**：封面→[过渡→2~3内容页]×N章节→数据→结尾
- 7页以上至少3种模板；10页以上必须有过渡页；内容页交替变化

## 铁律

- **Z-order**：图片最后添加，确保在最上层
- **数据页禁止图片**
- **所有图片用add_picture_cropped**，严禁add_picture同时指定宽高
- **封面遮罩≤45%，必须半透明**
- **要点≥3条，自适应间距，不溢出**
- **内容具体**（带价格/时间/数字）
- **配色全文统一**
- **长文案先改写再排版**：不要为了塞内容把 BODY 降到 14 以下
- **同模板页尺寸一致**：左右分栏、卡片宽高、标题位置在全文保持不变
- **复改优先局部改**：已有成品只改 1-2 页时，优先编辑，不重做全 deck
