@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo.
echo ========================================
echo   买手 Agent 网页服务
echo   启动后浏览器打开: http://127.0.0.1:8001
echo ========================================
echo.
where py >nul 2>&1
if %errorlevel% neq 0 (
  echo [错误] 未找到 py 命令，请安装 Python 3.9+
  pause
  exit /b 1
)
if not exist ".env" (
  echo [提示] 未找到 .env，请复制 .env.example 并填写 DEEPSEEK_API_KEY
  echo        或在网页里保存操作员 Key
  echo.
)
start "" "http://127.0.0.1:8001"
py -3.9 -m uvicorn web_server:app --reload --host 127.0.0.1 --port 8001
