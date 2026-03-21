---
triggers:
  - Excel
  - xlsx
  - 表格
  - spreadsheet
  - 数据表
  - 报表
max_tokens: 2200
lock_session: false
---

# 生成 Excel (openpyxl)

## 流程

1. 先回复用户：说明表格结构和数据规划
2. `file_write` → 写完整 Python 脚本到 `~/.whaleclaw/workspace/tmp/gen_xlsx_xxx.py`
3. `bash` → 执行（Windows: `.\python\python.exe`，Linux/Mac: `./python/bin/python3.12`）
4. 告诉用户文件路径

**严禁：** 不用 `python -c`；不分多次 file_write

---

## 基础设置

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import unicodedata

wb = Workbook()
ws = wb.active
ws.title = "数据报表"

HEADER_FILL = PatternFill(start_color="0F2747", end_color="0F2747", fill_type="solid")
HEADER_FONT = Font(name="Microsoft YaHei", size=11, bold=True, color="FFFFFF")
BODY_FONT = Font(name="Microsoft YaHei", size=10, color="333333")
ACCENT_FONT = Font(name="Microsoft YaHei", size=10, bold=True, color="1A73E8")
ZEBRA_FILL = PatternFill(start_color="F4F7FB", end_color="F4F7FB", fill_type="solid")
THIN_BORDER = Border(
    left=Side(style='thin', color='D0D5DD'), right=Side(style='thin', color='D0D5DD'),
    top=Side(style='thin', color='D0D5DD'), bottom=Side(style='thin', color='D0D5DD'))
CENTER_ALIGN = Alignment(horizontal='center', vertical='center', wrap_text=True)
LEFT_ALIGN = Alignment(horizontal='left', vertical='center', wrap_text=True)

def text_units(value):
    total = 0
    for ch in str(value or ""):
        total += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return total
```

## 表头与数据

```python
headers = ["序号", "项目名称", "负责人", "进度", "截止日期", "备注"]
for col_idx, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col_idx, value=header)
    cell.font = HEADER_FONT; cell.fill = HEADER_FILL
    cell.alignment = CENTER_ALIGN; cell.border = THIN_BORDER

for row_idx, row_data in enumerate(data, 2):
    for col_idx, value in enumerate(row_data, 1):
        cell = ws.cell(row=row_idx, column=col_idx, value=value)
        cell.font = BODY_FONT
        cell.alignment = CENTER_ALIGN if col_idx in (1, 4, 5) else LEFT_ALIGN
        cell.border = THIN_BORDER
        if row_idx % 2 == 0:
            cell.fill = ZEBRA_FILL

# 列宽自适应
for col_idx in range(1, len(headers) + 1):
    max_len = max(text_units(ws.cell(row=r, column=col_idx).value) for r in range(1, ws.max_row + 1))
    ws.column_dimensions[get_column_letter(col_idx)].width = min(max(max_len * 0.9 + 4, 10), 28)

ws.row_dimensions[1].height = 24
for row_idx in range(2, ws.max_row + 1):
    ws.row_dimensions[row_idx].height = 20

ws.freeze_panes = "A2"
ws.auto_filter.ref = ws.dimensions
```

## 汇总行

```python
SUMMARY_FILL = PatternFill(start_color="E8EDF2", end_color="E8EDF2", fill_type="solid")
SUMMARY_FONT = Font(name="Microsoft YaHei", size=10, bold=True, color="0F2747")

summary_row = ws.max_row + 1
ws.cell(row=summary_row, column=1, value="合计").font = SUMMARY_FONT
for col_idx in range(1, len(headers) + 1):
    cell = ws.cell(row=summary_row, column=col_idx)
    cell.fill = SUMMARY_FILL; cell.border = THIN_BORDER
# 对数字列求和（按实际列调整）
for num_col in [4]:
    cell = ws.cell(row=summary_row, column=num_col)
    cell.value = f"=SUM({get_column_letter(num_col)}2:{get_column_letter(num_col)}{summary_row-1})"
    cell.font = SUMMARY_FONT; cell.number_format = '#,##0'
```

## 条件格式

```python
from openpyxl.formatting.rule import DataBarRule, CellIsRule

# 数据条（进度可视化）
ws.conditional_formatting.add(f"D2:D{ws.max_row}",
    DataBarRule(start_type='min', end_type='max', color="1A73E8", showValue=True))

# 异常值高亮
ws.conditional_formatting.add(f"E2:E{ws.max_row}",
    CellIsRule(operator='greaterThan', formula=['10000'],
              fill=PatternFill(start_color="FDE8E8", end_color="FDE8E8", fill_type="solid"),
              font=Font(color="CC0000", bold=True)))
```

## 图表

```python
from openpyxl.chart import BarChart, Reference

def add_chart(ws, title, data_col, label_col=1):
    chart = BarChart(); chart.style = 10; chart.title = title
    chart.width = 18; chart.height = 12
    data_end = ws.max_row
    if ws.cell(row=data_end, column=1).value == "合计":
        data_end -= 1
    data = Reference(ws, min_col=data_col, min_row=1, max_row=data_end)
    labels = Reference(ws, min_col=label_col, min_row=2, max_row=data_end)
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(labels)
    if chart.series:
        chart.series[0].graphicalProperties.solidFill = "1A73E8"
    col = get_column_letter(len(list(ws.columns)) + 2)
    ws.add_chart(chart, f"{col}2")
```

## 内容要求

- 数据必须真实具体，不要"示例1"
- 首行冻结 + 深色背景 + 加粗白字
- 斑马纹交替行 + 全边框
- 列宽自适应
- 有数字列时加汇总行
- 数据量>5行时加数据条/条件格式
- 报表类配套至少1个图表
- 多维度数据拆分多Sheet：`wb.create_sheet(title="xxx")`
- 表头固定 11pt、正文固定 10pt；不要因列宽紧张去改字号
- 先按字段类型排版：编号/日期/百分比居中，名称/备注左对齐，金额加 number_format
- 单元格内容过长时优先换行和调列宽，不要合并大量单元格制造错位
- 修改已有 Excel 优先 `xlsx_edit`，不要重建整表导致筛选、冻结、公式丢失
