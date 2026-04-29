from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import sys
import subprocess
import threading
import time
import uuid
import zipfile
import io
from typing import Any

from fastapi import HTTPException, UploadFile

from ..pipeline import PipelineConfig, run_mask_pipeline_with_regions
from ..render import PdfRenderOptions, render_pdf_to_images
from ..detect import OrientedTextDetector
from ..recognize import RecognitionConfig, RegionTextRecognizer, extract_region_roi
from ..translate import (
    OpenAICompatConfig,
    load_openai_compat_config,
    translate_region_draft,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json_safe(path: Path, default: dict | list | None = None) -> Any:
    """
    Best-effort JSON loader.

    Prevents transient empty/partial writes from crashing polling endpoints.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return default if default is not None else {}
    if not str(raw).strip():
        return default if default is not None else {}
    try:
        return json.loads(raw)
    except Exception:
        return default if default is not None else {}


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """
    Atomically write text to disk (best-effort on Windows).

    Avoids readers observing an empty/half-written JSON file during polling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex}")
    tmp.write_text(text, encoding=encoding)
    try:
        tmp.replace(path)
    except Exception:
        # Fallback if replace() fails across FS boundaries / permission quirks.
        try:
            path.write_text(text, encoding=encoding)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass


@dataclass
class GenerateOptions:
    dpi: int = 400
    render_backend: str = "auto"
    poppler_path: str | None = None
    ocr_mode: str = "eco"
    ocr_workers: int | None = None
    ocr_device: str = "cpu"
    allow_fallback: bool = False


class ProjectService:
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._translate_cfg: OpenAICompatConfig | None = None
        self._translate_queue: "queue.Queue[dict]" = queue.Queue()
        self._translate_threads: list[threading.Thread] = []
        self._translate_started = False
        self._translate_metrics_lock = threading.Lock()
        # project_id -> {"queued": int, "active": int, "done": int, "error": int}
        self._translate_metrics: dict[str, dict[str, int]] = {}
        self._translate_dedupe_lock = threading.Lock()
        self._translate_dedupe_keys: set[str] = set()
        self._generate_threads: dict[str, threading.Thread] = {}
        self._generate_lock = threading.Lock()
        self._regions_lock = threading.Lock()
        self._regions_page_locks: dict[str, threading.Lock] = {}

    def _page_regions_lock(self, project_id: str, page_id: str) -> threading.Lock:
        key = f"{project_id}:{page_id}"
        with self._regions_lock:
            lock = self._regions_page_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._regions_page_locks[key] = lock
            return lock

    def _upsert_page_region(self, project_id: str, page_id: str, region: dict) -> None:
        """
        Incrementally persist OCR regions while the pipeline is running.
        Stored in output/regions/{page_id}_regions.json so /assets can surface partial results.
        """
        region_id = str(region.get("region_id") or "").strip()
        if not region_id:
            return
        path = self._page_regions_path(project_id, page_id)
        lock = self._page_regions_lock(project_id, page_id)
        with lock:
            payload = {"page_id": page_id, "regions": []}
            if path.exists():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    payload = {"page_id": page_id, "regions": []}
            regions = payload.get("regions") or []
            if not isinstance(regions, list):
                regions = []
            by_id: dict[str, dict] = {}
            for r in regions:
                try:
                    rid = str((r or {}).get("region_id") or "").strip()
                except Exception:
                    rid = ""
                if rid:
                    by_id[rid] = dict(r or {})
            by_id[region_id] = dict(region)
            merged = list(by_id.values())
            # Keep stable ordering for UI: sort by region_id.
            try:
                merged.sort(key=lambda x: str(x.get("region_id") or ""))
            except Exception:
                pass
            payload["page_id"] = page_id
            payload["regions"] = merged
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _translate_metrics_snapshot(self, project_id: str) -> dict[str, int]:
        with self._translate_metrics_lock:
            cur = self._translate_metrics.get(project_id) or {}
            return {
                "queued": int(cur.get("queued", 0) or 0),
                "active": int(cur.get("active", 0) or 0),
                "done": int(cur.get("done", 0) or 0),
                "error": int(cur.get("error", 0) or 0),
            }

    def _translate_metrics_bump(self, project_id: str, *, queued: int = 0, active: int = 0, done: int = 0, error: int = 0) -> None:
        with self._translate_metrics_lock:
            cur = self._translate_metrics.setdefault(project_id, {"queued": 0, "active": 0, "done": 0, "error": 0})
            cur["queued"] = max(0, int(cur.get("queued", 0) or 0) + int(queued))
            cur["active"] = max(0, int(cur.get("active", 0) or 0) + int(active))
            cur["done"] = max(0, int(cur.get("done", 0) or 0) + int(done))
            cur["error"] = max(0, int(cur.get("error", 0) or 0) + int(error))

    def _refresh_translation_overview(self, project_id: str, *, lang: str = "ru", max_pages: int = 250) -> dict:
        """
        Lightweight best-effort aggregate used for /status polling.
        We intentionally avoid deep per-region inspection here.
        """
        try:
            pages = self.list_pages(project_id)
        except Exception:
            pages = []

        totals = {
            "pages_total": int(len(pages)),
            "pages_with_data": 0,
            "regions_total": 0,
            "draft_done": 0,
            "draft_error": 0,
        }
        for page_id in pages[: max(0, int(max_pages))]:
            try:
                st = self.load_translation_status(project_id, page_id, lang=lang)
            except Exception:
                continue
            totals["pages_with_data"] += 1
            totals["regions_total"] += int(st.get("regions_total", 0) or 0)
            totals["draft_done"] += int(st.get("draft_done", 0) or 0)
            totals["draft_error"] += int(st.get("draft_error", 0) or 0)

        q = self._translate_metrics_snapshot(project_id)
        in_flight = int(q.get("queued", 0) + q.get("active", 0))
        state = "idle"
        if totals["regions_total"] > 0 and (in_flight > 0 or totals["draft_done"] + totals["draft_error"] < totals["regions_total"]):
            state = "running"
        if totals["regions_total"] > 0 and totals["draft_done"] + totals["draft_error"] >= totals["regions_total"] and in_flight == 0:
            state = "done"
        return {"state": state, "queue": q, **totals, "updated_at": _utc_now()}

    @staticmethod
    def _probe_paddle_runtime() -> dict:
        import_ok = False
        init_ok = False
        probe_error: str | None = None
        try:
            import paddleocr  # type: ignore

            import_ok = True
            version = str(getattr(paddleocr, "__version__", "unknown"))
        except Exception as exc:
            return {
                "paddle_import_ok": False,
                "paddle_init_ok": False,
                "paddle_version": None,
                "paddle_error": str(exc),
            }

        cmd = [
            sys.executable,
            "-c",
            (
                "from paddleocr import PaddleOCR\n"
                "ok = False\n"
                "errs = []\n"
                "ctors = [\n"
                "    dict(lang='ch', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, show_log=False),\n"
                "    dict(lang='ch', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, use_angle_cls=False),\n"
                "    dict(lang='ch', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False),\n"
                "    dict(lang='ch', show_log=False),\n"
                "    dict(lang='ch', use_angle_cls=False),\n"
                "    dict(lang='ch'),\n"
                "]\n"
                "for kw in ctors:\n"
                "    try:\n"
                "        PaddleOCR(**kw)\n"
                "        ok = True\n"
                "        break\n"
                "    except Exception as e:\n"
                "        errs.append(str(e))\n"
                "if not ok:\n"
                "    raise RuntimeError(' ; '.join(errs[-2:]) or 'Paddle init failed')\n"
                "print('ok')\n"
            ),
        ]
        env = dict(os.environ)
        env.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(os.getenv("OCR_PADDLE_PROBE_TIMEOUT_S", "45")),
                env=env,
            )
            init_ok = proc.returncode == 0
            if not init_ok:
                probe_error = (proc.stderr or proc.stdout or "").strip() or f"exit_code={proc.returncode}"
        except Exception as exc:
            init_ok = False
            probe_error = str(exc)

        return {
            "paddle_import_ok": import_ok,
            "paddle_init_ok": init_ok,
            "paddle_version": version,
            "paddle_error": probe_error,
        }

    def _ensure_translate_workers(self) -> None:
        if self._translate_started:
            return
        self._translate_started = True
        workers = int(os.getenv("TRANSLATE_WORKERS", "2") or "2")
        workers = max(1, min(6, workers))
        for i in range(workers):
            t = threading.Thread(target=self._translate_worker_loop, name=f"translate-worker-{i}", daemon=True)
            t.start()
            self._translate_threads.append(t)

    @staticmethod
    def _probe_paddle_device_runtime() -> dict:
        """
        Lightweight Paddle runtime probe:
        - is Paddle compiled with CUDA
        - what device Paddle reports (cpu / gpu:0)
        """
        paddle_error: str | None = None
        compiled = False
        device = "cpu"
        version: str | None = None
        try:
            import paddle  # type: ignore

            version = str(getattr(paddle, "__version__", None) or "unknown")
            compiled = bool(paddle.is_compiled_with_cuda())
            try:
                device = str(paddle.get_device() or "cpu")
            except Exception:
                device = "cpu"
        except Exception as exc:
            paddle_error = str(exc)
            compiled = False
            device = "cpu"

        cuda_usable = bool(compiled and str(device).lower().startswith("gpu"))
        return {
            "python_executable": sys.executable,
            "paddle_version": version,
            "paddle_is_compiled_with_cuda": bool(compiled),
            "paddle_device": device,
            "paddle_cuda_available": bool(cuda_usable),
            "paddle_error": paddle_error,
        }

    def _translate_worker_loop(self) -> None:
        while True:
            job = self._translate_queue.get()
            try:
                self._run_translate_job(job)
            except Exception:
                # Errors are persisted per-job into translation JSON; avoid killing the worker.
                pass
            finally:
                try:
                    self._translate_queue.task_done()
                except Exception:
                    pass

    def _translate_config(self) -> OpenAICompatConfig:
        if self._translate_cfg is None:
            self._translate_cfg = load_openai_compat_config()
        return self._translate_cfg

    def _translations_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "output" / "translations"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _page_context_path(self, project_id: str, page_id: str, lang: str) -> Path:
        return self._translations_dir(project_id) / f"{page_id}_page_context_{lang}.json"

    def _page_regions_translation_path(self, project_id: str, page_id: str, lang: str) -> Path:
        return self._translations_dir(project_id) / f"{page_id}_regions_{lang}.json"

    def _regions_dir(self, project_id: str) -> Path:
        d = self._project_dir(project_id) / "output" / "regions"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _page_regions_path(self, project_id: str, page_id: str) -> Path:
        return self._regions_dir(project_id) / f"{page_id}_regions.json"

    def enqueue_missing_translations_from_report(self, project_id: str, report: dict, *, lang: str = "ru") -> dict:
        """
        Import-mode helper: persist regions from exported report.json into the project and
        enqueue translation jobs for anything not yet translated.
        """
        pages = report.get("pages") or []
        regions_by_page = report.get("regionsByPage") or {}
        if not isinstance(regions_by_page, dict):
            raise HTTPException(status_code=400, detail="Invalid report: regionsByPage must be object.")

        # Persist regions so translation totals/status work (load_translation_status reads output/regions).
        persisted_pages = 0
        persisted_regions = 0
        for p in pages:
            page_id = str((p or {}).get("page_id") or "").strip()
            if not page_id:
                continue
            regions = regions_by_page.get(page_id) or []
            if not isinstance(regions, list):
                continue
            out_path = self._page_regions_path(project_id, page_id)
            try:
                out_path.write_text(json.dumps({"page_id": page_id, "regions": regions}, ensure_ascii=False, indent=2), encoding="utf-8")
                persisted_pages += 1
                persisted_regions += int(len(regions))
            except Exception:
                # best-effort
                pass

        # Enqueue missing jobs (dedupe against existing translation payload).
        queued_region_draft = 0

        for p in pages:
            page_id = str((p or {}).get("page_id") or "").strip()
            if not page_id:
                continue
            regions = regions_by_page.get(page_id) or []
            if not isinstance(regions, list):
                continue

            payload = self._load_regions_translation_payload(project_id, page_id, lang)
            items = payload.get("items") or {}

            for r in regions:
                region_id = str((r or {}).get("region_id") or "").strip()
                text = str((r or {}).get("text") or "").strip()
                if not region_id or not text or text == "Текст не найден":
                    continue

                existing = items.get(region_id) or {}
                if str(existing.get("status_draft") or "") != "done" and not str(existing.get("draft_translation") or "").strip():
                    self.enqueue_region_draft(project_id, page_id, {"region_id": region_id, "text": text}, lang=lang)
                    queued_region_draft += 1

        return {
            "project_id": project_id,
            "lang": lang,
            "persisted_pages": int(persisted_pages),
            "persisted_regions": int(persisted_regions),
            "queued": {
                "region_draft": int(queued_region_draft),
            },
        }

    @staticmethod
    def _hash_text(*parts: str) -> str:
        h = hashlib.sha1()
        for p in parts:
            h.update((p or "").encode("utf-8"))
            h.update(b"\n")
        return h.hexdigest()[:20]

    def _load_regions_translation_payload(self, project_id: str, page_id: str, lang: str) -> dict:
        path = self._page_regions_translation_path(project_id, page_id, lang)
        if not path.exists():
            return {"page_id": page_id, "target_lang": lang, "items": {}}
        payload = _read_json_safe(path, default=None)
        if isinstance(payload, dict):
            payload.setdefault("page_id", page_id)
            payload.setdefault("target_lang", lang)
            payload.setdefault("items", {})
            return payload
        return {"page_id": page_id, "target_lang": lang, "items": {}}

    def _write_regions_translation_payload(self, project_id: str, page_id: str, lang: str, payload: dict) -> None:
        path = self._page_regions_translation_path(project_id, page_id, lang)
        _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _upsert_region_translation(
        self,
        project_id: str,
        page_id: str,
        region_id: str,
        *,
        lang: str,
        source_text: str,
        patch: dict,
    ) -> None:
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        items = payload.get("items") or {}
        entry = items.get(region_id) or {}
        entry.setdefault("source_text", source_text)
        entry.update(patch or {})
        entry["updated_at"] = _utc_now()
        items[region_id] = entry
        payload["items"] = items
        self._write_regions_translation_payload(project_id, page_id, lang, payload)

    def enqueue_region_draft(self, project_id: str, page_id: str, region: dict, *, lang: str = "ru") -> None:
        text = str(region.get("text") or "").strip()
        if not text or text == "Текст не найден":
            return
        cfg = self._translate_config()
        src_hash = self._hash_text(text, cfg.model, lang, cfg.prompt_version, "region_draft")

        # Skip enqueue if already translated with same source hash.
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        existing = ((payload.get("items") or {}).get(str(region.get("region_id") or "")) or {})
        if (
            str(existing.get("draft_source_hash") or "") == src_hash
            and str(existing.get("status_draft") or "") == "done"
            and str(existing.get("draft_translation") or "").strip()
        ):
            return

        # In-memory dedupe to avoid queue storms before worker marks "running".
        rid = str(region.get("region_id") or "").strip()
        if not rid:
            return
        dedupe_key = f"{project_id}:{page_id}:{rid}:{src_hash}"
        with self._translate_dedupe_lock:
            if dedupe_key in self._translate_dedupe_keys:
                return
            self._translate_dedupe_keys.add(dedupe_key)

        self._ensure_translate_workers()
        self._translate_metrics_bump(project_id, queued=1)
        self._translate_queue.put(
            {
                "type": "region_draft",
                "project_id": project_id,
                "page_id": page_id,
                "region_id": str(region.get("region_id") or ""),
                "lang": lang,
                "source_text": text,
                "dedupe_key": dedupe_key,
            }
        )

    def _run_translate_job(self, job: dict) -> None:
        cfg = self._translate_config()
        job_type = str(job.get("type") or "")
        project_id = str(job.get("project_id") or "")
        page_id = str(job.get("page_id") or "")
        lang = str(job.get("lang") or "ru")
        if not project_id or not page_id or not job_type:
            return

        # Metrics: the worker took one job from the queue.
        # IMPORTANT: we MUST always decrement "active" before returning,
        # including cache-hits/skip paths, otherwise /status will be stuck in "running".
        self._translate_metrics_bump(project_id, queued=-1, active=1)
        finished = False

        def _finish(*, done: int = 0, error: int = 0) -> None:
            nonlocal finished
            if finished:
                return
            finished = True
            self._translate_metrics_bump(project_id, active=-1, done=done, error=error)
            try:
                dk = str(job.get("dedupe_key") or "").strip()
                if dk:
                    with self._translate_dedupe_lock:
                        self._translate_dedupe_keys.discard(dk)
            except Exception:
                pass

        if job_type == "region_draft":
            region_id = str(job.get("region_id") or "")
            source_text = str(job.get("source_text") or "").strip()
            if not region_id or not source_text:
                _finish(done=1)
                return
            src_hash = self._hash_text(source_text, cfg.model, lang, cfg.prompt_version, "region_draft")
            payload = self._load_regions_translation_payload(project_id, page_id, lang)
            existing = ((payload.get("items") or {}).get(region_id) or {})
            if str(existing.get("draft_source_hash") or "") == src_hash and str(existing.get("status_draft") or "") == "done":
                _finish(done=1)
                return
            self._upsert_region_translation(
                project_id,
                page_id,
                region_id,
                lang=lang,
                source_text=source_text,
                patch={"status_draft": "running", "draft_source_hash": src_hash, "error_draft": None},
            )
            try:
                translation = translate_region_draft(cfg, source_text, target_lang=lang)
                self._upsert_region_translation(
                    project_id,
                    page_id,
                    region_id,
                    lang=lang,
                    source_text=source_text,
                    patch={"draft_translation": translation, "status_draft": "done"},
                )
                _finish(done=1)
            except Exception as e:
                self._upsert_region_translation(
                    project_id,
                    page_id,
                    region_id,
                    lang=lang,
                    source_text=source_text,
                    patch={"status_draft": "error", "error_draft": str(e)},
                )
                _finish(error=1)
            return

    def create_project(self, upload: UploadFile) -> dict:
        project_id = uuid.uuid4().hex[:12]
        project_dir = self.root_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=False)

        filename = upload.filename or "input.pdf"
        pdf_path = project_dir / "input.pdf"
        with pdf_path.open("wb") as file_handle:
            file_handle.write(upload.file.read())

        status = {
            "project_id": project_id,
            "filename": filename,
            "status": "uploaded",
            "error": None,
            "pages": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_status(project_id, status)
        return status

    @staticmethod
    def _safe_stem(filename: str | None) -> str:
        name = str(filename or "").strip() or "document"
        # Remove path components and illegal characters for Windows.
        name = name.replace("\\", "/").split("/")[-1]
        for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*']:
            name = name.replace(ch, "_")
        if name.lower().endswith(".pdf"):
            name = name[:-4]
        name = name.strip().strip(".") or "document"
        return name[:120]

    def export_ocpkg_bytes(self, project_id: str) -> tuple[bytes, str]:
        """
        Create a single-file bundle (.ocpkg) containing:
        - input.pdf
        - report.json
        """
        project_dir = self._project_dir(project_id)
        status = self.get_status(project_id)
        pdf_path = project_dir / "input.pdf"
        report_path = project_dir / "output" / "report.json"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Input PDF not found.")
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="report.json not found (nothing to export yet).")

        stem = self._safe_stem(str(status.get("filename") or "document.pdf"))
        download_name = f"{stem}_ocr.ocpkg"

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps({"schema": "ocpkg_v1", "project_id": project_id}, ensure_ascii=False))
            zf.write(str(pdf_path), arcname="input.pdf")
            zf.write(str(report_path), arcname="report.json")
        return buf.getvalue(), download_name

    def import_ocpkg_and_create_project(self, upload: UploadFile) -> dict:
        """
        Import a .ocpkg (zip) and create a new project with input.pdf restored.
        Returns: {project_id, filename}
        """
        raw = upload.file.read()
        try:
            zf = zipfile.ZipFile(io.BytesIO(raw))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid ocpkg: {exc}") from exc

        names = set(zf.namelist())
        if "input.pdf" not in names or "report.json" not in names:
            raise HTTPException(status_code=400, detail="Invalid ocpkg: expected input.pdf and report.json inside.")

        # Create project directory and write PDF
        project_id = uuid.uuid4().hex[:12]
        project_dir = self.root_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=False)
        pdf_path = project_dir / "input.pdf"
        with pdf_path.open("wb") as f:
            f.write(zf.read("input.pdf"))

        filename = upload.filename or "result.ocpkg"
        status = {
            "project_id": project_id,
            "filename": filename,
            "status": "uploaded",
            "error": None,
            "pages": 0,
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }
        self._write_status(project_id, status)

        # Return report content for enqueue/render path (caller will use it)
        report_raw = zf.read("report.json")
        try:
            report = json.loads(report_raw.decode("utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid report.json in ocpkg: {exc}") from exc
        return {"project_id": project_id, "filename": filename, "report": report}

    def create_import_project_and_render_pages(
        self,
        upload: UploadFile,
        *,
        dpi: int = 400,
        render_backend: str = "auto",
        poppler_path: str | None = None,
    ) -> dict:
        """
        Create a project from an uploaded PDF and render pages only (no OCR/mask).
        Intended for fast offline report review (import JSON + PDF).
        """
        created = self.create_project(upload)
        project_id = str(created["project_id"])
        project_dir = self._project_dir(project_id)
        pdf_path = project_dir / "input.pdf"
        output_dir = project_dir / "output"
        rendered_dir = output_dir / "rendered_pages"
        output_dir.mkdir(parents=True, exist_ok=True)

        status = self.get_status(project_id)
        status["status"] = "running"
        status["error"] = None
        status["stage"] = "render"
        status["progress"] = {
            "status": "running",
            "indeterminate": True,
            "current_page": 0,
            "total_pages": 0,
            "page_id": None,
            "current_region": 0,
            "total_regions": 0,
        }
        status["updated_at"] = _utc_now()
        self._write_status(project_id, status)

        try:
            opts = PdfRenderOptions()
            opts.dpi = int(dpi)
            opts.backend = render_backend or "auto"
            opts.poppler_path = poppler_path
            page_paths = render_pdf_to_images(pdf_path, rendered_dir, opts)
            status["status"] = "done"
            status["stage"] = "render"
            status["pages"] = int(len(page_paths))
            status["progress"] = {
                "status": "done",
                "indeterminate": False,
                "current_page": int(len(page_paths)),
                "total_pages": int(len(page_paths)),
                "page_id": str(page_paths[0].stem) if page_paths else None,
                "current_region": 0,
                "total_regions": 0,
            }
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            return {"project_id": project_id, "pages": [p.stem for p in page_paths]}
        except Exception as exc:
            status["status"] = "error"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            raise

    def render_pages_for_existing_project(
        self,
        project_id: str,
        *,
        dpi: int = 400,
        render_backend: str = "auto",
        poppler_path: str | None = None,
    ) -> dict:
        """
        Render pages for an already-created project (expects input.pdf present).
        Used by .ocpkg import flow.
        """
        project_dir = self._project_dir(project_id)
        pdf_path = project_dir / "input.pdf"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Input PDF not found.")
        output_dir = project_dir / "output"
        rendered_dir = output_dir / "rendered_pages"
        output_dir.mkdir(parents=True, exist_ok=True)

        status = self.get_status(project_id)
        status["status"] = "running"
        status["error"] = None
        status["stage"] = "render"
        status["progress"] = {
            "status": "running",
            "indeterminate": True,
            "current_page": 0,
            "total_pages": 0,
            "page_id": None,
            "current_region": 0,
            "total_regions": 0,
        }
        status["updated_at"] = _utc_now()
        self._write_status(project_id, status)

        try:
            opts = PdfRenderOptions()
            opts.dpi = int(dpi)
            opts.backend = render_backend or "auto"
            opts.poppler_path = poppler_path
            page_paths = render_pdf_to_images(pdf_path, rendered_dir, opts)
            status["status"] = "done"
            status["stage"] = "render"
            status["pages"] = int(len(page_paths))
            status["progress"] = {
                "status": "done",
                "indeterminate": False,
                "current_page": int(len(page_paths)),
                "total_pages": int(len(page_paths)),
                "page_id": str(page_paths[0].stem) if page_paths else None,
                "current_region": 0,
                "total_regions": 0,
            }
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            return {"project_id": project_id, "pages": [p.stem for p in page_paths]}
        except Exception as exc:
            status["status"] = "error"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            raise

    def generate(self, project_id: str, options: GenerateOptions) -> dict:
        status = self.get_status(project_id)
        project_dir = self._project_dir(project_id)
        pdf_path = project_dir / "input.pdf"
        output_dir = project_dir / "output"
        if not pdf_path.exists():
            raise HTTPException(status_code=404, detail="Input PDF not found.")

        status["status"] = "running"
        status["error"] = None
        status["stage"] = "mask"
        status["progress"] = {
            "status": "running",
            "indeterminate": True,
            "current_page": 0,
            "total_pages": 0,
            "page_id": None,
            "current_region": 0,
            "total_regions": 0,
        }
        requested_device = str(options.ocr_device or "cpu").strip().lower()
        if requested_device not in {"cpu", "cuda"}:
            requested_device = "cpu"
        paddle_probe = self._probe_paddle_device_runtime()
        cuda_ok = bool(paddle_probe.get("paddle_cuda_available"))
        effective_device = "cuda" if requested_device == "cuda" and cuda_ok else "cpu"
        status["paddle_runtime"] = {
            "requested_device": requested_device,
            "effective_device": effective_device,
            **paddle_probe,
        }
        status["updated_at"] = _utc_now()
        self._write_status(project_id, status)

        try:
            config = PipelineConfig(dpi=options.dpi)
            config.render.backend = options.render_backend
            config.render.poppler_path = options.poppler_path
            config.detector.ocr_device = (options.ocr_device or "cpu").strip().lower()
            config.detector.allow_fallback = bool(options.allow_fallback)
            self._tune_pipeline_config(config, options)
            rec_cfg = self._build_recognition_config(options)

            # Strict Paddle-only mode validates detector/recognizer init upfront so
            # failures are reported immediately in web status.
            OrientedTextDetector(config.detector)
            RegionTextRecognizer(rec_cfg)

            def on_progress(update: dict) -> None:
                status_inner = self.get_status(project_id)
                status_inner["stage"] = update.get("stage", "ocr")
                status_inner["progress"] = {
                    "current_page": int(update.get("current_page", 0)),
                    "total_pages": int(update.get("total_pages", 0)),
                    "page_id": update.get("page_id"),
                    "status": update.get("status"),
                    "current_region": int(update.get("current_region", 0) or 0),
                    "total_regions": int(update.get("total_regions", 0) or 0),
                }
                # Per-page progress map for UI (page_id -> {percent, current_region, total_regions, status})
                page_id = update.get("page_id")
                if page_id:
                    pages_map = status_inner.get("progress_pages") or {}
                    try:
                        cur_r = int(update.get("current_region", 0) or 0)
                        tot_r = int(update.get("total_regions", 0) or 0)
                    except Exception:
                        cur_r, tot_r = 0, 0
                    percent = None
                    if tot_r > 0:
                        raw_pct = int(round((float(cur_r) * 100.0) / float(max(1, tot_r))))
                        percent = max(0, min(100, raw_pct))
                    pages_map[str(page_id)] = {
                        "percent": percent,
                        "current_region": cur_r,
                        "total_regions": tot_r,
                        "status": str(update.get("status") or ""),
                        "current_page": int(update.get("current_page", 0) or 0),
                        "total_pages": int(update.get("total_pages", 0) or 0),
                    }
                    status_inner["progress_pages"] = pages_map
                status_inner["updated_at"] = _utc_now()
                self._write_status(project_id, status_inner)

            report = run_mask_pipeline_with_regions(
                pdf_path=pdf_path,
                output_dir=output_dir,
                config=config,
                recognition_config=rec_cfg,
                progress_callback=on_progress,
                region_ready_callback=lambda page_id, rec: (
                    self._upsert_page_region(project_id, str(page_id), rec),
                    self.enqueue_region_draft(project_id, str(page_id), rec, lang="ru"),
                ),
                page_done_callback=None,
            )
            # Persist full report to disk. status.json remains a lightweight snapshot for polling.
            self._write_report(project_id, report)
            status["pipeline_status"] = "done"
            # Translation continues asynchronously (already enqueued during OCR). Keep stage at OCR until fully done.
            status["status"] = "running"
            status["stage"] = "ocr"
            status["pages"] = len(report.get("pages", []))
            status["report_path"] = str((output_dir / "report.json").as_posix())
            status["ocr_profile_path"] = str((output_dir / "ocr_profile.json").as_posix())
            status["updated_at"] = _utc_now()
            status["translation"] = self._refresh_translation_overview(project_id, lang="ru")
            self._write_status(project_id, status)
            return {"status": status, "report": report}
        except Exception as exc:
            status["status"] = "error"
            status["error"] = str(exc)
            status["updated_at"] = _utc_now()
            self._write_status(project_id, status)
            raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    def start_generate_background(self, project_id: str, options: GenerateOptions) -> dict:
        """
        Start generation in a background thread so the web server can keep
        answering /status polls while the pipeline runs.
        """
        # Validate existence early (also warms up status file).
        status = self.get_status(project_id)

        with self._generate_lock:
            existing = self._generate_threads.get(project_id)
            if existing is not None and existing.is_alive():
                return {"started": False, "status": "already_running", "project_id": project_id}

            def _runner() -> None:
                try:
                    self.generate(project_id, options)
                except Exception:
                    # generate() persists error status; keep thread alive-safe.
                    return

            t = threading.Thread(target=_runner, name=f"generate-{project_id}", daemon=True)
            self._generate_threads[project_id] = t
            t.start()

        # Return the latest status snapshot immediately.
        try:
            status = self.get_status(project_id)
        except Exception:
            pass
        return {"started": True, "status": "running", "project_id": project_id, "snapshot": status}

    def load_translation_status(self, project_id: str, page_id: str, *, lang: str = "ru") -> dict:
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        items = payload.get("items") or {}

        # Compute totals from actual OCR regions, not from translation payload (which can be sparse).
        try:
            regions = self.load_page_regions(project_id, page_id) or []
        except Exception:
            regions = []
        region_ids = [str(r.get("region_id") or "") for r in regions if str(r.get("region_id") or "").strip()]

        draft_done = draft_err = draft_running = draft_pending = draft_unknown = 0

        for rid in region_ids:
            entry = items.get(rid) or {}
            sd = str(entry.get("status_draft") or "").strip().lower()

            if not sd:
                draft_pending += 1
            elif sd == "done":
                draft_done += 1
            elif sd == "error":
                draft_err += 1
            elif sd == "running":
                draft_running += 1
            else:
                draft_unknown += 1
        return {
            "page_id": page_id,
            "lang": lang,
            "regions_total": int(len(region_ids)),
            "draft_done": int(draft_done),
            "draft_error": int(draft_err),
            "draft_running": int(draft_running),
            "draft_pending": int(draft_pending),
            "draft_unknown": int(draft_unknown),
        }

    def load_region_translation(self, project_id: str, page_id: str, region_id: str, *, lang: str = "ru") -> dict:
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        entry = (payload.get("items") or {}).get(region_id) or {}
        return {
            "page_id": page_id,
            "region_id": region_id,
            "lang": lang,
            "draft_translation": entry.get("draft_translation"),
            "status_draft": entry.get("status_draft") or ("pending" if not entry else "unknown"),
            "error_draft": entry.get("error_draft"),
            "updated_at": entry.get("updated_at"),
        }

    def load_page_translations(self, project_id: str, page_id: str, *, lang: str = "ru") -> dict:
        """
        Return all region translations for a page in one request to avoid N+1 polling.
        """
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        items = payload.get("items") or {}
        out: dict[str, dict] = {}
        for region_id, entry in (items or {}).items():
            e = entry or {}
            out[str(region_id)] = {
                "region_id": str(region_id),
                "draft_translation": e.get("draft_translation"),
                "status_draft": e.get("status_draft") or ("pending" if not e else "unknown"),
                "error_draft": e.get("error_draft"),
                "updated_at": e.get("updated_at"),
            }
        return {"page_id": page_id, "lang": lang, "items": out}

    def _build_recognition_config(self, options: GenerateOptions) -> RecognitionConfig:
        mode = (options.ocr_mode or "eco").lower()
        ocr_device = (options.ocr_device or "cpu").strip().lower()
        if ocr_device not in {"cpu", "cuda"}:
            ocr_device = "cpu"
        is_cuda = ocr_device == "cuda"
        cpu = int(os.cpu_count() or 4)
        debug_raw = str(os.getenv("OCR_DEBUG_RAW_RESULTS", "") or "").strip().lower() in {"1", "true", "yes", "on"}
        # On Windows, the latest PaddleOCR/PaddleX pipeline can fail at runtime with
        # oneDNN/PIR executor issues in detection, while RapidOCR works reliably for ROI recognition.
        # Allow forcing Paddle recognition via env var if needed.
        force_paddle = str(os.getenv("OCR_FORCE_PADDLE_RECOGNITION", "") or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        backend = "paddleocr"
        if sys.platform.startswith("win") and not force_paddle:
            backend = "rapidocr"
        auto_workers = max(2, min(8, cpu // 2))
        workers = options.ocr_workers if options.ocr_workers is not None else auto_workers
        workers = max(1, int(workers))
        if mode == "max":
            max_workers = min(8, workers)
            if is_cuda:
                max_workers = min(8, max(2, workers))
            return RecognitionConfig(
                backend=backend,
                mode="hybrid",
                use_multipass=True,
                parallel_ocr=True,
                max_workers=max_workers,
                batch_size=20 if is_cuda else 24,
                parallel_min_regions=10 if is_cuda else 12,
                cascade_probe_variants=2,
                backend_cascade=bool(options.allow_fallback),
                allow_fallback=bool(options.allow_fallback),
                debug_raw_results=bool(debug_raw),
                ocr_device=ocr_device,
            )
        if mode == "balanced":
            max_workers = min(4, workers)
            if is_cuda:
                max_workers = min(6, max(2, workers))
            return RecognitionConfig(
                backend=backend,
                mode="hybrid",
                use_multipass=False,
                parallel_ocr=True,
                max_workers=max_workers,
                batch_size=22 if is_cuda else 28,
                parallel_min_regions=8 if is_cuda else 14,
                cascade_probe_variants=2,
                bridge_fallback_enabled=bool(options.allow_fallback),
                bridge_fallback_confidence_threshold=0.82,
                bridge_fallback_score_threshold=0.98,
                backend_cascade=bool(options.allow_fallback),
                allow_fallback=bool(options.allow_fallback),
                debug_raw_results=bool(debug_raw),
                ocr_device=ocr_device,
            )
        # eco (default): keep CPU usage low for personal machines
        if is_cuda:
            return RecognitionConfig(
                backend=backend,
                mode="fast",
                use_multipass=False,
                parallel_ocr=True,
                max_workers=min(4, max(2, workers)),
                batch_size=20,
                parallel_min_regions=8,
                retry_confidence_threshold=0.70,
                accept_confidence_threshold=0.92,
                accept_score_threshold=1.04,
                accept_min_length=6,
                cascade_probe_variants=2,
                bridge_fallback_enabled=bool(options.allow_fallback),
                bridge_fallback_confidence_threshold=0.82,
                bridge_fallback_score_threshold=0.98,
                backend_cascade=bool(options.allow_fallback),
                allow_fallback=bool(options.allow_fallback),
                debug_raw_results=bool(debug_raw),
                ocr_device=ocr_device,
            )
        return RecognitionConfig(
            backend=backend,
            mode="fast",
            use_multipass=False,
            parallel_ocr=False,
            max_workers=1,
            batch_size=24,
            parallel_min_regions=12,
            retry_confidence_threshold=0.70,
            accept_confidence_threshold=0.92,
            accept_score_threshold=1.04,
            accept_min_length=6,
            cascade_probe_variants=2,
            bridge_fallback_enabled=bool(options.allow_fallback),
            bridge_fallback_confidence_threshold=0.82,
            bridge_fallback_score_threshold=0.98,
            backend_cascade=bool(options.allow_fallback),
            allow_fallback=bool(options.allow_fallback),
            debug_raw_results=bool(debug_raw),
            ocr_device=ocr_device,
        )

    def _tune_pipeline_config(self, config: PipelineConfig, options: GenerateOptions) -> None:
        mode = (options.ocr_mode or "eco").lower()
        if mode == "eco":
            config.dpi = min(int(config.dpi), 360)
            config.detector.score_threshold = max(0.24, float(config.detector.score_threshold))
            config.detector.min_area = max(20, int(config.detector.min_area))
            return
        if mode == "balanced":
            config.dpi = min(int(config.dpi), 380)
            config.detector.score_threshold = max(0.22, float(config.detector.score_threshold))
            config.detector.min_area = max(18, int(config.detector.min_area))

    def get_status(self, project_id: str) -> dict:
        status_path = self._project_dir(project_id) / "status.json"
        if not status_path.exists():
            raise HTTPException(status_code=404, detail="Project not found.")
        status = _read_json_safe(status_path, default={})
        if not isinstance(status, dict):
            status = {}
        # If status.json was transiently empty/invalid, keep polling stable instead of 500.
        status.setdefault("project_id", project_id)
        status.setdefault("status", "running")
        status.setdefault("stage", "init")
        status.setdefault("updated_at", _utc_now())

        # Best-effort: keep translation overview fresh and only mark the project
        # fully done when translation settled.
        try:
            stage = str(status.get("stage") or "").lower()
            pipeline_done = str(status.get("pipeline_status") or "") == "done"
            # Backward/upgrade compatibility: older status.json files won't have paddle_runtime.
            # Fill it lazily so the UI can reliably show CPU/GPU.
            if status.get("paddle_runtime") is None:
                probe = self._probe_paddle_device_runtime()
                cuda_ok = bool(probe.get("paddle_cuda_available"))
                requested = "cuda" if cuda_ok else "cpu"
                status["paddle_runtime"] = {
                    "requested_device": requested,
                    "effective_device": "cuda" if requested == "cuda" and cuda_ok else "cpu",
                    **probe,
                }
            # During OCR/mask we only scan a small set of pages to keep /status cheap,
            # but still expose queue metrics so the UI can show parallel translation activity.
            overview_pages = 250 if pipeline_done else 8
            if pipeline_done or stage in {"ocr", "mask"}:
                overview = self._refresh_translation_overview(project_id, lang="ru", max_pages=overview_pages)
                status["translation"] = overview
                if pipeline_done and str(status.get("status") or "") == "running" and str(overview.get("state") or "") == "done":
                    status["status"] = "done"
                    status["stage"] = "done"
                    status["updated_at"] = _utc_now()
                # Persist updated snapshot (keeps polling consistent).
                _atomic_write_text(status_path, json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        return status

    def list_pages(self, project_id: str) -> list[str]:
        output_dir = self._project_dir(project_id) / "output" / "rendered_pages"
        pages = [page.stem for page in sorted(output_dir.glob("page_*.png"))]
        if not pages:
            raise HTTPException(status_code=404, detail="No generated pages found.")
        return pages

    def load_page_regions(self, project_id: str, page_id: str) -> list[dict]:
        region_path = self._project_dir(project_id) / "output" / "regions" / f"{page_id}_regions.json"
        if not region_path.exists():
            return []
        payload = json.loads(region_path.read_text(encoding="utf-8"))
        return payload.get("regions", [])

    def load_region_by_id(self, project_id: str, region_id: str) -> dict | None:
        output_dir = self._project_dir(project_id) / "output" / "regions"
        for regions_file in sorted(output_dir.glob("page_*_regions.json")):
            payload = json.loads(regions_file.read_text(encoding="utf-8"))
            for region in payload.get("regions", []):
                if region.get("region_id") == region_id:
                    return region
        return None

    def retry_region_ocr(self, project_id: str, region_id: str, *, allow_fallback: bool = False) -> dict:
        region = self.load_region_by_id(project_id, region_id)
        if region is None:
            raise HTTPException(status_code=404, detail="Region not found.")
        crop_path = region.get("crop_path")
        crop_file: Path | None = None

        if crop_path:
            raw = Path(str(crop_path))
            candidates: list[Path] = []
            if raw.is_absolute():
                candidates.append(raw)
            # Sometimes stored relative to repo root, e.g. "web_jobs\\<id>\\output\\..."
            candidates.append(Path.cwd() / raw)
            # Sometimes stored relative to project dir
            candidates.append(self._project_dir(project_id) / raw)
            # If it includes "web_jobs/<project_id>/...", strip prefix and treat as project-relative
            prefix = Path("web_jobs") / project_id
            try:
                rel_after_prefix = raw.relative_to(prefix)
                candidates.append(self._project_dir(project_id) / rel_after_prefix)
            except Exception:
                pass

            for cand in candidates:
                try:
                    resolved = cand.resolve()
                except Exception:
                    continue
                if resolved.exists():
                    crop_file = resolved
                    break

        import cv2
        import numpy as np

        roi = None
        if crop_file is not None:
            roi = cv2.imread(str(crop_file), cv2.IMREAD_GRAYSCALE)

        # Fallback: regenerate ROI crop from rendered page + polygon
        if roi is None:
            page_id = region.get("page_id")
            polygon = region.get("polygon")
            if not page_id or not polygon:
                raise HTTPException(status_code=404, detail="Crop file not found.")
            page_img = cv2.imread(str(self.image_path(project_id, str(page_id))), cv2.IMREAD_GRAYSCALE)
            if page_img is None:
                raise HTTPException(status_code=500, detail="Failed to read rendered page image.")
            poly = np.asarray(polygon, dtype=np.float32)
            roi = extract_region_roi(page_img, poly)
            if roi is None or roi.size == 0:
                raise HTTPException(status_code=500, detail="Failed to extract ROI from polygon.")

        # Accurate single-region retry (still CPU-friendly because it's one ROI).
        # If PaddleOCR is not available in this interpreter (e.g. Python 3.14),
        # you can provide OCR_PADDLE_PYTHON pointing to a Python 3.10–3.12 env
        # with paddleocr installed, and we'll call it via subprocess bridge.
        rec_cfg = RecognitionConfig(
            backend="paddleocr",
            mode="accurate",
            use_multipass=True,
            parallel_ocr=False,
            max_workers=1,
            allow_fallback=bool(allow_fallback),
            backend_cascade=bool(allow_fallback),
            bridge_fallback_enabled=bool(allow_fallback),
        )
        recognizer = RegionTextRecognizer(rec_cfg)
        text, confidence, variant, score = recognizer.recognize_roi(roi)
        trace = recognizer.consume_last_trace()
        text_clean = (text or "").strip() or "Текст не найден"
        if text_clean == "Текст не найден":
            confidence = 0.0
            score = 0.0
        return {
            "region_id": region_id,
            "text": text_clean,
            "ocr_confidence": float(confidence),
            "ocr_variant": variant,
            "ocr_score": float(score),
            "ocr_trace": trace,
        }

    def image_path(self, project_id: str, page_id: str) -> Path:
        path = self._project_dir(project_id) / "output" / "rendered_pages" / f"{page_id}.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Rendered page not found.")
        return path

    def mask_path(self, project_id: str, page_id: str) -> Path:
        path = self._project_dir(project_id) / "output" / "masks" / f"{page_id}_mask.png"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Mask not found.")
        return path

    def normalize_page_id(self, page: str) -> str:
        if page.startswith("page_"):
            return page
        if page.isdigit():
            return f"page_{int(page):04d}"
        raise HTTPException(status_code=400, detail="Page must be number or page_XXXX format.")

    def _project_dir(self, project_id: str) -> Path:
        project_dir = self.root_dir / project_id
        if not project_dir.exists():
            raise HTTPException(status_code=404, detail="Project not found.")
        return project_dir

    def _write_status(self, project_id: str, status: dict) -> None:
        path = self._project_dir(project_id) / "status.json"
        _atomic_write_text(path, json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_report(self, project_id: str, report: dict) -> None:
        project_dir = self._project_dir(project_id)
        out_dir = project_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        # Convenience: persist OCR trace-only view for quick debugging.
        try:
            ocr_profile = (report.get("region_precompute") or {}).get("profiling") or {}
        except Exception:
            ocr_profile = {}
        if ocr_profile:
            (out_dir / "ocr_profile.json").write_text(
                json.dumps(ocr_profile, ensure_ascii=False, indent=2), encoding="utf-8"
            )
