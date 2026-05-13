from __future__ import annotations

from dataclasses import dataclass
import os
import time

from openai import APIError, BadRequestError, OpenAI  # type: ignore


class TranslateError(RuntimeError):
    pass


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        return default
    return str(v)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return bool(default)
    v = str(raw).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off", ""}:
        return False
    return bool(default)


@dataclass
class OpenAICompatConfig:
    base_url: str
    api_key: str | None
    model: str
    temperature: float = 0.1
    timeout_s: float = 60.0
    max_retries: int = 2
    prompt_version: str = "translate_v1"
    # Chat-completions output limit. Keeping this set avoids requests that "hang"
    # behind gateways and makes job completion deterministic.
    region_max_tokens: int = 256
    page_max_tokens: int = 4096
    # "Thinking" / reasoning effort knobs for OpenAI-compatible gateways.
    # For per-region (single) calls we default to explicitly disabled ("none"),
    # because some gateways may otherwise return reasoning-only payloads with empty content.
    # For page-context calls we default to "high" to maximize quality.
    region_reasoning_effort: str | None = "none"
    page_reasoning_effort: str | None = "high"
    # llama.cpp OpenAI-compat uses chat template kwargs to toggle "thinking" mode
    # for Qwen3.5-style templates. When enabled:
    # - page-context: enable_thinking=true
    # - per-region: enable_thinking=false
    llamacpp_chat_template_thinking: bool = False


def load_openai_compat_config() -> OpenAICompatConfig:
    base_url = _env("TRANSLATE_API_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
    base_url = base_url.rstrip("/")
    api_key = _env("TRANSLATE_API_KEY", None)
    model = _env("TRANSLATE_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini"
    temperature = float(_env("TRANSLATE_TEMPERATURE", "0.1") or "0.1")
    timeout_s = float(_env("TRANSLATE_TIMEOUT_S", "60") or "60")
    max_retries = int(float(_env("TRANSLATE_MAX_RETRIES", "2") or "2"))
    prompt_version = _env("TRANSLATE_PROMPT_VERSION", "translate_v1") or "translate_v1"
    region_max_tokens = int(float(_env("TRANSLATE_REGION_MAX_TOKENS", "256") or "256"))
    page_max_tokens = int(float(_env("TRANSLATE_PAGE_MAX_TOKENS", "4096") or "4096"))
    region_reasoning_effort = _env("TRANSLATE_REGION_REASONING_EFFORT", "none") or "none"
    page_reasoning_effort = _env("TRANSLATE_PAGE_REASONING_EFFORT", "high") or "high"
    llamacpp_chat_template_thinking = _env_bool("TRANSLATE_LLAMACPP_CHAT_TEMPLATE_THINKING", False)
    return OpenAICompatConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout_s=timeout_s,
        max_retries=max_retries,
        prompt_version=prompt_version,
        region_max_tokens=max(1, region_max_tokens),
        page_max_tokens=max(1, page_max_tokens),
        region_reasoning_effort=region_reasoning_effort,
        page_reasoning_effort=page_reasoning_effort,
        llamacpp_chat_template_thinking=bool(llamacpp_chat_template_thinking),
    )


def _extract_chat_text(resp: object) -> str:
    """
    Be tolerant to slightly-nonstandard OpenAI-compatible responses.

    Expected (OpenAI SDK): resp.choices[0].message.content -> str
    Some gateways may return message as a plain string/dict; sometimes text is in choices[0].text.
    """
    try:
        choices = getattr(resp, "choices", None) or []
    except Exception:
        choices = []
    if not choices:
        return ""

    ch0 = choices[0]

    msg = getattr(ch0, "message", None)
    if msg is not None:
        # Common SDK shape
        try:
            content = getattr(msg, "content", None)
        except Exception:
            content = None
        if content is not None:
            try:
                s = str(content).strip()
                if s:
                    return s
            except Exception:
                pass

        # Gateway oddities
        if isinstance(msg, str):
            s = msg.strip()
            if s:
                return s
        if isinstance(msg, dict):
            try:
                s = str(msg.get("content") or "").strip()
                if s:
                    return s
            except Exception:
                pass

        # Last resort: stringify message object
        try:
            s = str(msg).strip()
            if s:
                return s
        except Exception:
            pass

        # Some gateways put output into message.reasoning_content and leave content empty.
        # We do NOT treat it as translation output, but it's useful for diagnostics upstream.

    # Some compat servers return "text" per choice.
    try:
        s = str(getattr(ch0, "text", "") or "").strip()
        if s:
            return s
    except Exception:
        pass

    return ""


def _chat_completions(
    cfg: OpenAICompatConfig,
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    reasoning_effort: str | None,
    enable_thinking: bool | None,
) -> str:
    """
    Use the official OpenAI SDK against an OpenAI-compatible gateway.

    Important behavior:
    - Always sets max_tokens (to avoid very long/hanging requests behind gateways).
    - Uses `reasoning_effort` only when requested. For per-region calls we default to None.
      For page-context calls we default to "high".
    - Falls back automatically if the gateway rejects `reasoning_effort`.
    """
    client = OpenAI(
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        timeout=float(cfg.timeout_s),
    )

    last_err: Exception | None = None
    for attempt in range(max(1, int(cfg.max_retries) + 1)):
        try:
            kwargs: dict = {
                "model": cfg.model,
                "temperature": float(cfg.temperature),
                "stream": False,
                "messages": messages,
            }

            # OpenAI platform prefers max_completion_tokens; llama.cpp OpenAI-compat typically expects max_tokens.
            if cfg.llamacpp_chat_template_thinking:
                kwargs["max_tokens"] = int(max_tokens)
            else:
                # max_tokens is deprecated and not compatible with some reasoning models.
                # Use max_completion_tokens (includes visible + reasoning tokens).
                kwargs["max_completion_tokens"] = int(max_tokens)

            if cfg.llamacpp_chat_template_thinking and enable_thinking is not None:
                kwargs["extra_body"] = {
                    "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                }

            # Only pass this knob for the "full text" path unless explicitly configured.
            if reasoning_effort is not None:
                kwargs["reasoning_effort"] = reasoning_effort
            try:
                resp = client.chat.completions.create(**kwargs)
            except TypeError:
                # Client doesn't accept reasoning_effort (or other kwarg); retry without it.
                kwargs.pop("reasoning_effort", None)
                resp = client.chat.completions.create(**kwargs)
            except (BadRequestError, APIError):
                # Some gateways accept the param name but reject value (e.g. "none")
                # or only support a subset. Retry without reasoning_effort.
                if "reasoning_effort" in kwargs:
                    kwargs.pop("reasoning_effort", None)
                    resp = client.chat.completions.create(**kwargs)
                else:
                    raise

            content = _extract_chat_text(resp)
            if not content:
                try:
                    choice0 = getattr(resp, "choices", [None])[0]
                    msg0 = getattr(choice0, "message", None)
                    reasoning0 = getattr(msg0, "reasoning_content", None)
                except Exception:
                    msg0 = None
                    reasoning0 = None
                if reasoning0:
                    # Truncate to keep status/error payloads small.
                    r = str(reasoning0)
                    if len(r) > 800:
                        r = r[:800] + "…"
                    raise TranslateError(
                        f"Empty message content in response (message={msg0!r}, reasoning_content={r!r})."
                    )
                raise TranslateError(f"Empty message content in response (message={msg0!r}).")
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
    return _chat_completions(
        cfg,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=int(cfg.region_max_tokens),
        reasoning_effort=cfg.region_reasoning_effort,
        enable_thinking=False,
    )


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
    return _chat_completions(
        cfg,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=int(cfg.page_max_tokens),
        reasoning_effort=cfg.page_reasoning_effort,
        enable_thinking=True,
    )


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
    return _chat_completions(
        cfg,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=int(cfg.region_max_tokens),
        reasoning_effort=cfg.region_reasoning_effort,
        enable_thinking=False,
    )

