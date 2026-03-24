---
triggers:
  - PPT
  - pptx
  - 幻灯片
  - 演示文稿
max_tokens: 2300
lock_session: false
---

# 生成 PPT (python-pptx)

## 流程

1. 回复用户：几页、什么内容
2. `browser`搜图（根据主题搜"[关键词] 高清"），封面优先横版高清大图
3. `file_write`完整脚本→`~/.whaleclaw/workspace/tmp/gen_ppt_xxx.py`
4. `bash`执行 → 告诉用户路径

复刻：截图→vision提取配色→确认→搜图→脚本；.pptx→bash提取→脚本
严禁：`python -c`；分多次file_write；硬编码路径；重写辅助函数

## 基础（脚本开头必须包含）

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

配色7变量：PRIMARY/SECONDARY/ACCENT/BG_LIGHT/TEXT_DARK/TEXT_LIGHT/TEXT_GRAY
字体：中文"Microsoft YaHei"，英文"Arial Black"/"Arial"
字号：HERO=44 H1=32 H2=24 BODY=16 CAPTION=12 NUMBER=56

## 辅助函数（原样复制到脚本，禁止重写）

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

def add_shape_block(slide, l, t, w, h, color, shape=MSO_SHAPE.RECTANGLE, alpha=1.0):
    s = slide.shapes.add_shape(shape, int(l), int(t), int(w), int(h))
    s.fill.solid(); s.fill.fore_color.rgb = color; s.line.fill.background()
    if alpha < 1.0:
        from pptx.oxml.ns import qn; from lxml import etree
        sf = s._element.spPr.find(qn("a:solidFill"))
        if sf is not None:
            clr = sf.find(qn("a:srgbClr"))
            if clr is not None:
                etree.SubElement(clr, qn("a:alpha")).set("val", str(int(alpha*100000)))
    return s

def add_tb(slide, l, t, w, h, text, sz, color, bold=False, align=PP_ALIGN.LEFT):
    from pptx.enum.text import MSO_AUTO_SIZE
    tb = slide.shapes.add_textbox(int(l), int(t), int(w), int(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left=tf.margin_right=Pt(2); tf.margin_top=tf.margin_bottom=0
    p = tf.paragraphs[0]; p.text = text; p.alignment = align
    r = p.runs[0]; r.font.size=sz; r.font.color.rgb=color; r.font.bold=bold; r.font.name="Microsoft YaHei"
    p.space_before=p.space_after=0; p.line_spacing=1.15
    return tb
```

add_tb的h：单行>=字号*1.5，多行>=行数*字号*1.3。放不下则减条或拆页。

可用形状：RECTANGLE/ROUNDED_RECTANGLE/OVAL/ISOSCELES_TRIANGLE/RIGHT_TRIANGLE/PARALLELOGRAM/TRAPEZOID/DIAMOND/CHEVRON，禁止使用不在此列表中的MSO_SHAPE常量。

## 风格指引

| 主题 | 配色 | 封面 | 装饰 |
|-----|-----|-----|-----|
| 人物 | 紫/玫红+金 | 左右分栏 | 金竖线、圆角标签 |
| 旅行 | 蓝/青绿+橙 | 全图+底栏 | 色块 |
| 商务 | 深蓝+金 | 上下分区/分栏 | 横线、标签 |
| 科技 | 暗灰+荧光 | 斜切/几何 | 线条框、色块 |
| 教育 | 绿/暖橙+白 | 居中卡片 | 圆形编号、色带 |
| 美食 | 暖红/橙+白 | 全图+底栏 | 圆角色块 |

同主题多次生成应变换布局。

## 排版通则

- **安全区**：上下>=0.5in，左右>=0.6in
- **对齐**：同区域元素左对齐；标题、装饰线、要点左边缘一致
- **间距**：标题→装饰线约0.1in→首条要点约0.3in→要点等距
- **文字区宽度**：图文页占50-55%，不低于40%
- **装饰**：用add_shape_block，根据主题选形状（直角/圆角/圆/三角/菱形等）
- **版式守恒**：1页1点超量拆页；标题<=18字；要点<=32字/条 3-5条/页
- **最小字号**：HERO>=36 H1>=28 H2>=22 BODY>=15 CAPTION>=11；小改用`ppt_edit`

## 版式规则

### 封面
- >=6 shapes，标题+副标题+>=2 ACCENT装饰
- 布局：左右分栏/全图+底栏/上下分区/斜切/居中卡片
- 全图遮罩<=45%面积，alpha<=0.75
- 标题与副标题间用装饰横线分隔

### 内容页
- 图片占42-48%宽，BG_LIGHT背景，图距边>=0.5in
- 文字区：标题→装饰线→要点(3-5条等距，SECONDARY标记)
- 左图右文/右图左文交替；图片最后添加(Z-order)

### 数据页
- 禁图片，顶部PRIMARY标题栏(h约1.3in)
- 下方2-4白色卡片等距(w约2.5-3in)，卡片内NUMBER+BODY居中

### 结尾页
- >=5 shapes，呼应封面配色，标题+副标题+>=1 ACCENT装饰
- 标题垂直居中(y 40-55%)居中对齐
- 禁全屏不透明遮罩；可用封面镜像/全图居中/纯色大字

### 过渡页（10页+必须有）
- PRIMARY纯色背景，左侧大编号Pt120(ACCENT)，右侧标题

## 组合

- **3页**：封面→内容→结尾 / **5页**：封面→左→右→左→结尾
- **7页**：封面→左→右→数据→左→右→结尾
- **10+**：封面→[过渡→2~3内容]*N→数据→结尾

## 铁律

- **辅助函数原样复制到脚本，禁止重写/简化/替代**；禁用shape.fill.transparency
- 图片最后添加(Z-order)；**只用add_picture_cropped**，之后禁设width/height
- 封面>=6 shapes、结尾>=5 shapes，含ACCENT装饰；3页也要独立结尾
- 遮罩<=45%面积+alpha<=0.75；禁全屏不透明遮罩；数据页禁图片
- 内容具体带数字；配色统一；每次变换封面/结尾布局
