#!/bin/bash
# 启动带远程调试端口的 Chrome（WhaleClaw 浏览器接管模式）
# 双击此文件即可运行，WhaleClaw 将通过 CDP 连接到此 Chrome 实例

PORT=9222
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
USER_DATA="$HOME/.whaleclaw/chrome-profile"

if [ ! -f "$CHROME" ]; then
    echo "❌ 未找到 Chrome，请确认已安装 Google Chrome"
    echo "   期望路径: $CHROME"
    read -n 1 -s -r -p "按任意键退出..."
    exit 1
fi

if lsof -i :"$PORT" >/dev/null 2>&1; then
    echo "✅ Claw 辅助浏览器已在运行，可以直接使用"
    echo ""
    echo "   WhaleClaw 可通过 localhost:$PORT 接管操作"
    echo "   如需重启，请先手动关闭辅助浏览器窗口，再双击本文件"
    echo ""
    read -n 1 -s -r -p "按任意键关闭此窗口..."
    exit 0
fi

mkdir -p "$USER_DATA"

echo "🚀 启动 Claw 辅助浏览器（端口: $PORT）"
echo ""
echo "   此浏览器使用独立数据目录，不影响你日常使用的 Chrome"
echo "   数据目录: $USER_DATA"
echo ""
echo "   👉 请在此浏览器中登录你需要操作的网站（如小红书）"
echo "   👉 登录完成后，WhaleClaw 即可接管操作"
echo ""
echo "   验证: curl http://localhost:$PORT/json"
echo ""

"$CHROME" \
    --remote-debugging-port="$PORT" \
    --user-data-dir="$USER_DATA" \
    --no-first-run \
    --no-default-browser-check \
    --disable-background-timer-throttling \
    --disable-backgrounding-occluded-windows \
    --disable-renderer-backgrounding \
    --disable-blink-features=AutomationControlled \
    --enable-features=SharedArrayBuffer \
    "about:blank"
