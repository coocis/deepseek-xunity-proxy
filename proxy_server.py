"""XUnity Auto Translator 的 DeepSeek 本机转发器。"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
LOG_DIR = APP_DIR / "logs"
FAILED_TRANSLATIONS_FILE = APP_DIR / "failed_translations.jsonl"


class ConfigurationError(ValueError):
    """启动配置无效。"""


@dataclass(frozen=True)
class Settings:
    bind_host: str
    port: int
    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float
    max_input_chars: int
    max_output_tokens: int
    temperature: float
    max_retries: int
    log_level: str
    log_text: bool
    log_retention_days: int
    system_prompt: str
    user_prompt: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class TranslationResult:
    text: str
    usage: TokenUsage | None


class SessionUsage:
    """线程安全地统计当前服务进程的 API token 用量。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    def add(self, usage: TokenUsage) -> TokenUsage:
        with self._lock:
            self._prompt_tokens += usage.prompt_tokens
            self._completion_tokens += usage.completion_tokens
            self._total_tokens += usage.total_tokens
            return TokenUsage(
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._total_tokens,
            )


def env_text(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def env_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(env_text(name, str(default)))
    except ValueError as error:
        raise ConfigurationError(f"{name} 必须是整数。") from error
    if value < minimum:
        raise ConfigurationError(f"{name} 不能小于 {minimum}。")
    return value


def env_float(name: str, default: float, minimum: float = 0) -> float:
    try:
        value = float(env_text(name, str(default)))
    except ValueError as error:
        raise ConfigurationError(f"{name} 必须是数字。") from error
    if value < minimum:
        raise ConfigurationError(f"{name} 不能小于 {minimum}。")
    return value


def env_bool(name: str, default: bool) -> bool:
    value = env_text(name, str(default)).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} 必须是 True 或 False。")


def read_prompt(filename: str) -> str:
    path = APP_DIR / filename
    try:
        text = path.read_text(encoding="utf-8-sig").strip()
    except FileNotFoundError as error:
        raise ConfigurationError(f"缺少提示词文件：{path}") from error
    if not text:
        raise ConfigurationError(f"提示词文件不能为空：{path}")
    return text


def load_settings() -> Settings:
    log_level = env_text("LOG_LEVEL", "INFO").upper()
    if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigurationError("LOG_LEVEL 必须是 DEBUG、INFO、WARNING、ERROR 或 CRITICAL。")

    return Settings(
        bind_host=env_text("BIND_HOST", "127.0.0.1"),
        port=env_int("PROXY_PORT", 8765, 1),
        base_url=env_text("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        model=env_text("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        api_key_env=env_text("DEEPSEEK_API_KEY_ENV", "DEEPSEEK_API_KEY"),
        timeout_seconds=env_float("REQUEST_TIMEOUT_SECONDS", 60.0, 1.0),
        max_input_chars=env_int("MAX_INPUT_CHARS", 200, 1),
        max_output_tokens=env_int("MAX_OUTPUT_TOKENS", 1000, 1),
        temperature=env_float("TEMPERATURE", 0.1, 0.0),
        max_retries=env_int("MAX_RETRIES", 1, 0),
        log_level=log_level,
        log_text=env_bool("LOG_TEXT", False),
        log_retention_days=env_int("LOG_RETENTION_DAYS", 2, 0),
        system_prompt=read_prompt("system_prompt.txt"),
        user_prompt=read_prompt("user_prompt.txt"),
    )


def clean_old_logs(retention_days: int) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    cutoff = time.time() - timedelta(days=retention_days).total_seconds()
    for log_path in LOG_DIR.glob("proxy-*.log"):
        try:
            if log_path.stat().st_mtime < cutoff:
                log_path.unlink()
        except OSError:
            # 日志清理失败不应阻止转发器启动。
            pass


def configure_logging(settings: Settings) -> logging.Logger:
    clean_old_logs(settings.log_retention_days)
    logger = logging.getLogger("deepseek_xunity_proxy")
    logger.setLevel(getattr(logging, settings.log_level))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log_path = LOG_DIR / f"proxy-{datetime.now():%Y-%m-%d}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def append_failed_translation(source: str, source_lang: str, target_lang: str, reason: str) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_language": source_lang,
        "target_language": target_lang,
        "source_text": source,
        "reason": reason,
    }
    with FAILED_TRANSLATIONS_FILE.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_token_usage(payload: dict[str, Any]) -> TokenUsage | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    try:
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        total_tokens = int(usage.get("total_tokens", prompt_tokens + completion_tokens))
    except (KeyError, TypeError, ValueError):
        return None
    return TokenUsage(prompt_tokens, completion_tokens, total_tokens)


def extract_translation(payload: dict[str, Any]) -> TranslationResult:
    try:
        result = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("DeepSeek 响应中没有可用译文。") from error
    if not isinstance(result, str) or not result.strip():
        raise ValueError("DeepSeek 返回了空译文。")
    return TranslationResult(text=result.strip(), usage=read_token_usage(payload))


def translate(settings: Settings, source: str, source_lang: str, target_lang: str) -> TranslationResult:
    api_key = os.environ.get(settings.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(f"未找到 API Key 环境变量：{settings.api_key_env}")

    messages = [
        {"role": "system", "content": settings.system_prompt},
        {"role": "user", "content": settings.user_prompt},
        {
            "role": "user",
            "content": (
                f"源语言：{source_lang}\n目标语言：{target_lang}\n"
                "以下是待翻译文本。只输出译文：\n"
                f"{source}"
            ),
        },
    ]
    body = json.dumps(
        {
            "model": settings.model,
            "messages": messages,
            "temperature": settings.temperature,
            "max_tokens": settings.max_output_tokens,
            "stream": False,
            "thinking": {"type": "disabled"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        f"{settings.base_url}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(settings.max_retries + 1):
        try:
            with urlopen(request, timeout=settings.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
            return extract_translation(json.loads(response_body))
        except (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as error:
            last_error = error
            if attempt < settings.max_retries:
                time.sleep(1 + attempt)

    raise RuntimeError(str(last_error) if last_error else "未知翻译错误")


def make_handler(
    settings: Settings, logger: logging.Logger, session_usage: SessionUsage
) -> type[BaseHTTPRequestHandler]:
    class TranslationHandler(BaseHTTPRequestHandler):
        server_version = "DeepSeekXUnityProxy/1.0"

        def log_message(self, format: str, *args: object) -> None:
            logger.info("HTTP %s", format % args)

        def send_plain(self, status: int, text: str) -> None:
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                self.send_plain(200, "ok")
                return
            if parsed.path != "/translate":
                self.send_plain(404, "Not found")
                return

            query = parse_qs(parsed.query, keep_blank_values=True)
            source = query.get("text", [""])[0]
            source_lang = query.get("from", ["ja"])[0]
            target_lang = query.get("to", ["zh-CN"])[0]
            if not source:
                self.send_plain(400, "Missing text")
                return
            if len(source) > settings.max_input_chars:
                self.send_plain(400, "Text exceeds MAX_INPUT_CHARS")
                return

            try:
                result = translate(settings, source, source_lang, target_lang)
            except Exception as error:  # 必须以非 2xx 响应让 XUnity 不缓存失败结果。
                reason = f"{type(error).__name__}: {error}"
                try:
                    append_failed_translation(source, source_lang, target_lang, reason)
                except OSError as write_error:
                    logger.error("写入失败清单失败：%s", write_error)
                logger.error("翻译失败，已保留原文且记录到失败清单：%s", reason)
                self.send_plain(502, "Translation service failed")
                return

            if settings.log_text:
                logger.info("翻译成功：%r -> %r", source, result.text)
            else:
                logger.info("翻译成功：%s 字符", len(source))
            if result.usage is None:
                logger.warning("DeepSeek 响应未提供 token 用量，无法计入本次启动累计。")
            else:
                total = session_usage.add(result.usage)
                logger.info(
                    "Token 用量：本次 输入=%s 输出=%s 合计=%s；本次启动累计 输入=%s 输出=%s 合计=%s",
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    result.usage.total_tokens,
                    total.prompt_tokens,
                    total.completion_tokens,
                    total.total_tokens,
                )
            self.send_plain(200, result.text)

    return TranslationHandler


def main() -> int:
    try:
        settings = load_settings()
    except ConfigurationError as error:
        print(f"配置错误：{error}", file=sys.stderr)
        return 2

    logger = configure_logging(settings)
    logger.info("启动 DeepSeek XUnity 转发器；模型=%s，监听 http://%s:%s", settings.model, settings.bind_host, settings.port)
    logger.info("提示词从 system_prompt.txt 与 user_prompt.txt 读取；旧日志保留 %s 天", settings.log_retention_days)

    try:
        server = ThreadingHTTPServer(
            (settings.bind_host, settings.port), make_handler(settings, logger, SessionUsage())
        )
    except OSError as error:
        logger.error("无法监听端口：%s", error)
        return 3

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("收到停止请求，正在退出。")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
