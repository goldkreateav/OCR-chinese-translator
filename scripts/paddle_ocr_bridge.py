from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


def _preview_obj(obj, *, max_items: int = 24, max_str: int = 400, depth: int = 0, max_depth: int = 3):
    if depth > max_depth:
        return "<max_depth>"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        s = obj
        if len(s) > max_str:
            s = s[:max_str] + "…"
        return s
    if isinstance(obj, dict):
        keys = list(obj.keys())
        sample = {}
        for k in keys[:max_items]:
            try:
                sample[str(k)] = _preview_obj(obj.get(k), max_items=max_items, max_str=max_str, depth=depth + 1)
            except Exception:
                sample[str(k)] = "<error>"
        return {
            "__type__": "dict",
            "keys": [str(k) for k in keys[:max_items]],
            "keys_total": len(keys),
            "keys_truncated": len(keys) > max_items,
            "sample": sample,
        }
    if isinstance(obj, (list, tuple)):
        items = []
        for x in list(obj)[:max_items]:
            items.append(_preview_obj(x, max_items=max_items, max_str=max_str, depth=depth + 1))
        return {"__type__": "list" if isinstance(obj, list) else "tuple", "len": len(obj), "items": items}
    # objects with attrs
    try:
        attrs = {}
        for name in ("text", "score", "rec_text", "rec_score", "dt_polys", "dt_scores"):
            if hasattr(obj, name):
                attrs[name] = _preview_obj(getattr(obj, name), depth=depth + 1)
        if attrs:
            return {"__type__": type(obj).__name__, "attrs": attrs}
    except Exception:
        pass
    try:
        r = repr(obj)
    except Exception:
        r = "<unrepr>"
    return {"__type__": type(obj).__name__, "repr": _preview_obj(r, depth=depth + 1)}


def _best_text_from_any(result) -> tuple[str, float]:
    """
    PaddleOCR/PaddleX outputs vary wildly across versions.
    Try to find the best (text, score) candidate anywhere inside nested dict/list structures.
    """
    best_text = ""
    best_score = 0.0

    def consider(text, score) -> None:
        nonlocal best_text, best_score
        try:
            t = str(text or "").strip()
        except Exception:
            t = ""
        if not t:
            return
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s > best_score:
            best_text, best_score = t, s

    def walk(x) -> None:
        if x is None:
            return
        # Common pair formats: ["TEXT", 0.98] or ("TEXT", 0.98)
        if isinstance(x, (list, tuple)):
            if len(x) >= 2 and isinstance(x[0], str):
                consider(x[0], x[1])
            for it in x:
                walk(it)
            return
        if isinstance(x, dict):
            # Direct fields
            if "rec_text" in x or "rec_score" in x:
                consider(x.get("rec_text"), x.get("rec_score"))
            if "text" in x and ("score" in x or "prob" in x or "confidence" in x):
                consider(x.get("text"), x.get("score", x.get("prob", x.get("confidence"))))

            # Vectorized fields (common in PaddleX)
            if isinstance(x.get("rec_texts"), list) and isinstance(x.get("rec_scores"), list):
                texts = x.get("rec_texts") or []
                scores = x.get("rec_scores") or []
                for i in range(min(len(texts), len(scores))):
                    consider(texts[i], scores[i])
            if isinstance(x.get("rec_res"), list):
                for it in x.get("rec_res") or []:
                    walk(it)

            # Sometimes recognition outputs are under "result"/"results"/"res"
            for k in ("result", "results", "res", "rec", "rec_result", "rec_results"):
                if k in x:
                    walk(x.get(k))

            # Fallback: walk all values (but avoid very deep recursion)
            for v in x.values():
                walk(v)
            return
        # objects: try common attrs
        for name_text, name_score in (("rec_text", "rec_score"), ("text", "score")):
            if hasattr(x, name_text) and hasattr(x, name_score):
                consider(getattr(x, name_text), getattr(x, name_score))
        return

    walk(result)
    return best_text, float(best_score)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--lang", type=str, default="ch")
    args = parser.parse_args()

    from paddleocr import PaddleOCR  # type: ignore

    # Newer PaddleX-backed pipelines expect 3-channel images.
    img = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if img is None:
        print(json.dumps({"text": "", "confidence": 0.0, "raw_preview": {"error": "imread_failed"}}, ensure_ascii=True))
        return

    ctor_options = [
        dict(
            lang=args.lang,
            det=False,
            rec=True,
            show_log=False,
            use_angle_cls=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            lang=args.lang,
            show_log=False,
            use_angle_cls=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(
            lang=args.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        ),
        dict(use_angle_cls=False, lang=args.lang, det=False, rec=True, show_log=False),
        dict(use_angle_cls=False, lang=args.lang, show_log=False),
        dict(use_angle_cls=False, lang=args.lang),
        dict(lang=args.lang),
    ]
    last_exc = None
    ocr = None
    for kwargs in ctor_options:
        try:
            ocr = PaddleOCR(**kwargs)
            last_exc = None
            break
        except Exception as exc:
            last_exc = exc
    if ocr is None:
        raise RuntimeError(str(last_exc) if last_exc is not None else "Failed to initialize PaddleOCR bridge.")

    # PaddleOCR API varies across versions:
    # - older: ocr(img, det=..., rec=..., cls=...)
    # - newer: ocr(img) delegates to predict(img) with different kwargs
    result = None
    try:
        result = ocr.ocr(img, det=False, rec=True, cls=False)
    except Exception:
        try:
            result = ocr.ocr(img, cls=False)
        except Exception:
            if hasattr(ocr, "predict"):
                result = ocr.predict(img)
            else:
                result = ocr.ocr(img)

    raw_preview = None
    try:
        text, conf = _best_text_from_any(result)
    except Exception:
        text, conf = "", 0.0
    try:
        raw_preview = _preview_obj(result)
    except Exception:
        raw_preview = {"error": "preview_failed"}

    # Some Windows shells default to cp1251/cp866. Escaping avoids encoding errors.
    sys.stdout.write(json.dumps({"text": text, "confidence": conf, "raw_preview": raw_preview}, ensure_ascii=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

