---
triggers: ["浏览器", "打开网页", "截图", "browser", "screenshot", "webpage", "小红书", "公众号", "登录后操作"]
max_tokens: 800
---
# 浏览器控制

## 触发条件
当用户需要打开网页、截图、控制浏览器，或操作需要登录态的网站时使用此技能。

## 两种模式

### launch 模式（默认）
启动全新 Chrome，适用于搜图、爬取公开页面。无登录态。

### cdp 模式（接管辅助浏览器）
连接到用户已启动的 Claw 辅助浏览器，复用登录态。适用于小红书发布、公众号管理等需要登录的场景。
用户需先双击「启动Claw辅助浏览器.command」并手动登录目标网站。

## 指令
使用 browser 工具执行导航、截图、点击、输入、JS 执行等操作。

## 工具
- browser (action: navigate / screenshot / click / type / get_text / evaluate / back / close)

## 示例
- 打开百度并截图
- 帮我在小红书发布一篇笔记（需 cdp 模式）
- 访问 https://example.com 并提取页面文字
