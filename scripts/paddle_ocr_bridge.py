from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2


def _preview_obj(obj, *, max_items: int = 6, max_str: int = 400, depth: int = 0, max_depth: int = 3):
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
        return {"__type__": "dict", "keys": [str(k) for k in keys[:max_items]], "sample": sample}
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

    text = ""
    conf = 0.0
    raw_preview = None
    try:
        # PaddleOCR rec-only formats vary; normalize to best candidate.
        # Common: [[("text", 0.98)]] or [("text", 0.98)] or [["text", 0.98]]
        if isinstance(result, list) and result:
            r0 = result[0]
            if isinstance(r0, list) and r0:
                r0 = r0[0]
            if isinstance(r0, (list, tuple)) and len(r0) >= 2:
                text = str(r0[0] or "")
                conf = float(r0[1] or 0.0)
    except Exception:
        text = ""
        conf = 0.0
    try:
        raw_preview = _preview_obj(result)
    except Exception:
        raw_preview = {"error": "preview_failed"}

    # Some Windows shells default to cp1251/cp866. Escaping avoids encoding errors.
    sys.stdout.write(json.dumps({"text": text, "confidence": conf, "raw_preview": raw_preview}, ensure_ascii=True))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

