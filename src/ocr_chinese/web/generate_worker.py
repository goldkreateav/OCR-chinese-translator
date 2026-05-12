from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

from .service import GenerateOptions, ProjectService


def _read_payload_from_stdin() -> dict:
    raw = sys.stdin.read()
    raw = (raw or "").strip()
    if not raw:
        raise RuntimeError("Missing worker payload on stdin.")
    return json.loads(raw)


def main() -> int:
    payload = _read_payload_from_stdin()
    project_id = str(payload.get("project_id") or "").strip()
    root_dir = str(payload.get("root_dir") or "").strip()
    opts_raw = payload.get("options") or {}
    if not project_id or not root_dir:
        raise RuntimeError("Invalid payload: project_id/root_dir is required.")

    tw_raw = payload.get("translate_workers")
    tw_opt = None
    if tw_raw is not None:
        try:
            tw_opt = int(tw_raw)
        except (TypeError, ValueError):
            tw_opt = None
    try:
        base = int(tw_opt if tw_opt is not None else (os.getenv("TRANSLATE_WORKERS", "5") or "5"))
    except ValueError:
        base = 5
    base = max(1, min(5, base))
    # While OCR runs in this process, uncapped translation threads compete with GPU/memory.
    # Cap concurrent translation workers unless TRANSLATE_WORKERS_IN_GENERATION overrides (1..5).
    try:
        gen_cap = int(os.getenv("TRANSLATE_WORKERS_IN_GENERATION", "2") or "2")
    except ValueError:
        gen_cap = 2
    gen_cap = max(1, min(5, gen_cap))
    tw_final = max(1, min(base, gen_cap))
    service = ProjectService(Path(root_dir), translate_workers=int(tw_final))
    options = GenerateOptions(**opts_raw)

    # Cooperative cancellation between processes:
    # parent writes output/cancel.flag; we poll it and set a local event.
    cancel_event = threading.Event()
    cancel_flag = service._cancel_flag_path(project_id)

    def _cancel_poller() -> None:
        while not cancel_event.is_set():
            try:
                if cancel_flag.exists():
                    cancel_event.set()
                    return
            except Exception:
                pass
            time.sleep(0.5)

    t = threading.Thread(target=_cancel_poller, name=f"cancel-poller-{project_id}", daemon=True)
    t.start()

    # Run generation in this worker process (no backgrounding here).
    service.generate(project_id, options, cancel_event=cancel_event)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        # Print error for parent to collect in stderr log.
        print(str(exc), file=sys.stderr)
        raise

