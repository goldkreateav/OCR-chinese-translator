from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class TranslateError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v)


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str], timeout_s: float) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url=url, data=body, method="POST")
    req.add_header("Content-Type", "application/json; charset=utf-8")
    for k, v in headers.items():
        if v is None:
            continue
        req.add_header(k, v)
    try:
        with urlopen(req, timeout=float(timeout_s)) as resp:
            raw = resp.read()
    except HTTPError as e:
        msg = ""
        try:
            msg = (e.read() or b"").decode("utf-8", errors="replace")
        except Exception:
            msg = str(e)
        raise TranslateError(f"HTTP {getattr(e, 'code', '?')}: {msg}") from e
    except URLError as e:
        raise TranslateError(f"Network error: {e}") from e
    except Exception as e:
        raise TranslateError(f"Request failed: {e}") from e
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as e:
        raise TranslateError(f"Bad JSON response: {e}") from e


@dataclass
class OpenAICompatConfig:
    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.1
    timeout_s: float = 60.0
    max_retries: int = 2
    prompt_version: str = "translate_v1"


def load_openai_compat_config() -> OpenAICompatConfig:
    base_url = _env("TRANSLATE_API_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    base_url = base_url.rstrip("/")
    api_key = _env("TRANSLATE_API_KEY", None)
    model = _env("TRANSLATE_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
    temperature = float(_env("TRANSLATE_TEMPERATURE", "0.1") or "0.1")
    timeout_s = float(_env("TRANSLATE_TIMEOUT_S", "60") or "60")
    max_retries = int(float(_env("TRANSLATE_MAX_RETRIES", "2") or "2"))
    prompt_version = _env("TRANSLATE_PROMPT_VERSION", "translate_v1") or "translate_v1"
    return OpenAICompatConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_retries=max_retries,
        prompt_version=prompt_version,
    )


def _chat_completions(cfg: OpenAICompatConfig, messages: list[dict[str, str]]) -> str:
    url = f"{cfg.base_url}/chat/completions"
    headers: dict[str, str] = {}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": float(cfg.temperature),
    }

    last_err: Exception | None = None
    for attempt in range(max(1, int(cfg.max_retries) + 1)):
        try:
            data = _post_json(url, payload, headers=headers, timeout_s=cfg.timeout_s)
            choices = data.get("choices") or []
            if not choices:
                raise TranslateError("Empty choices in response.")
            msg = (choices[0] or {}).get("message") or {}
            content = str(msg.get("content") or "").strip()
            if not content:
                raise TranslateError("Empty message.content in response.")
            return content
        except Exception as e:
            last_err = e
            if attempt >= max(1, int(cfg.max_retries) + 1):
                break
            time.sleep(min(2.5, 0.6 * (attempt + 1)))
    raise TranslateError(str(last_err) if last_err else "Translate failed.")


def translate_region_draft(cfg: OpenAICompatConfig, region_text_zh: str, *, target_lang: str = "ru") -> str:
    text = (region_text_zh or "").strip()
    if not text:
        return ""
    system = (
        f"You are a professional technical translator. Translate from Chinese to {target_lang}. "
        "Return ONLY the translation text, no explanations."
    )
    user = f"Chinese:\n{text}\n\n{target_lang.upper()} translation:"
    return _chat_completions(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def translate_page_context(
    cfg: OpenAICompatConfig,
    page_text_zh: str,
    *,
    target_lang: str = "ru",
    delimiter: str = "\n---\n",
) -> str:
    text = (page_text_zh or "").strip()
    if not text:
        return ""
    system = (
        f"You are a professional technical translator. Translate from Chinese to {target_lang}. "
        f"Preserve the exact block separation delimiter {repr(delimiter)} between blocks. "
        "Return ONLY the translation text."
    )
    user = f"Chinese page text (blocks separated by delimiter):\n{text}\n\n{target_lang.upper()} translation:"
    return _chat_completions(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])


def translate_region_refine(
    cfg: OpenAICompatConfig,
    region_text_zh: str,
    page_context_translation: str,
    *,
    target_lang: str = "ru",
) -> str:
    text = (region_text_zh or "").strip()
    if not text:
        return ""
    context = (page_context_translation or "").strip()
    system = (
        f"You are a professional technical translator. Translate from Chinese to {target_lang}. "
        "You will be given a full-page translation for context. "
        "Return ONLY the best translation for the provided block."
    )
    user = (
        f"Full-page {target_lang.upper()} translation (context):\n{context}\n\n"
        f"Chinese block:\n{text}\n\n"
        f"{target_lang.upper()} block translation:"
    )
    return _chat_completions(cfg, [{"role": "system", "content": system}, {"role": "user", "content": user}])

