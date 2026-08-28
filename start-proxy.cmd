@echo off
setlocal EnableExtensions
chcp 65001 >nul

rem ===== XUnity + DeepSeek proxy settings =====
set "BIND_HOST=127.0.0.1"
set "PROXY_PORT=8765"
set "DEEPSEEK_BASE_URL=https://api.deepseek.com"
set "DEEPSEEK_MODEL=deepseek-v4-flash"
set "DEEPSEEK_API_KEY_ENV=DEEPSEEK_API_KEY"
set "REQUEST_TIMEOUT_SECONDS=60"
set "MAX_INPUT_CHARS=200"
set "MAX_OUTPUT_TOKENS=1000"
set "TEMPERATURE=0.1"
set "MAX_RETRIES=1"
set "LOG_LEVEL=INFO"
set "LOG_TEXT=False"
set "LOG_RETENTION_DAYS=2"

rem Make Python and child processes use UTF-8.
set "PYTHONUTF8=1"

py -3.12 "%~dp0proxy_server.py"
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Proxy exited with code: %EXIT_CODE%
  pause
)

endlocal & exit /b %EXIT_CODE%
