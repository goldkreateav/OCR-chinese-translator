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

from fastapi import HTTPException, UploadFile

from ..pipeline import PipelineConfig, run_mask_pipeline_with_regions
from ..render import PdfRenderOptions, render_pdf_to_images
from ..detect import OrientedTextDetector
from ..recognize import RecognitionConfig, RegionTextRecognizer, extract_region_roi
from ..translate import (
    OpenAICompatConfig,
    load_openai_compat_config,
    translate_page_context,
    translate_region_draft,
    translate_region_refine,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        self._generate_threads: dict[str, threading.Thread] = {}
        self._generate_lock = threading.Lock()

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
    def _probe_ocr_runtime() -> dict:
        providers: list[str] = []
        ort_error: str | None = None
        cuda_usable = False
        cuda_error: str | None = None
        try:
            import onnxruntime as ort  # type: ignore

            providers = list(ort.get_available_providers() or [])
        except Exception as exc:
            ort_error = str(exc)

        # Some ORT builds report CUDA as "available" but fail to load CUDA DLLs
        # at runtime. Try to actually create a CUDA session to validate.
        try:
            if "CUDAExecutionProvider" in providers:
                import onnxruntime as ort  # type: ignore
                from rapidocr_onnxruntime import rapid_ocr_api  # type: ignore
                from rapidocr_onnxruntime.utils import read_yaml  # type: ignore
                from pathlib import Path as _Path

                root_dir = _Path(rapid_ocr_api.__file__).resolve().parent
                cfg_path = root_dir / "config.yaml"
                cfg = read_yaml(str(cfg_path))
                det_rel = str(((cfg.get("Det") or {}).get("model_path") or "")).strip()
                model_path = root_dir / det_rel if det_rel else None
                if model_path is not None and model_path.exists():
                    sess = ort.InferenceSession(
                        str(model_path),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    cuda_usable = "CUDAExecutionProvider" in (sess.get_providers() or [])
        except Exception as exc:
            cuda_error = str(exc)
            cuda_usable = False

        return {
            "python_executable": sys.executable,
            "ort_available_providers": providers,
            "ort_cuda_available": bool(cuda_usable),
            "ort_probe_error": ort_error,
            "ort_cuda_error": cuda_error,
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
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"page_id": page_id, "target_lang": lang, "items": {}}

    def _write_regions_translation_payload(self, project_id: str, page_id: str, lang: str, payload: dict) -> None:
        path = self._page_regions_translation_path(project_id, page_id, lang)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        self._ensure_translate_workers()
        self._translate_queue.put(
            {
                "type": "region_draft",
                "project_id": project_id,
                "page_id": page_id,
                "region_id": str(region.get("region_id") or ""),
                "lang": lang,
                "source_text": text,
            }
        )

    def enqueue_page_context_and_refine(
        self, project_id: str, page_id: str, regions: list[dict], *, lang: str = "ru"
    ) -> None:
        self._ensure_translate_workers()
        page_text = "\n---\n".join(
            str(r.get("text") or "").strip()
            for r in (regions or [])
            if str(r.get("text") or "").strip() and str(r.get("text") or "").strip() != "Текст не найден"
        ).strip()
        self._translate_queue.put(
            {
                "type": "page_context",
                "project_id": project_id,
                "page_id": page_id,
                "lang": lang,
                "page_text": page_text,
            }
        )
        for r in regions or []:
            region_id = str(r.get("region_id") or "")
            text = str(r.get("text") or "").strip()
            if not region_id or not text or text == "Текст не найден":
                continue
            self._translate_queue.put(
                {
                    "type": "region_refine",
                    "project_id": project_id,
                    "page_id": page_id,
                    "region_id": region_id,
                    "lang": lang,
                    "source_text": text,
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

        if job_type == "page_context":
            page_text = str(job.get("page_text") or "").strip()
            if not page_text:
                return
            src_hash = self._hash_text(page_text, cfg.model, lang, cfg.prompt_version, "page_context")
            ctx_path = self._page_context_path(project_id, page_id, lang)
            if ctx_path.exists():
                try:
                    existing = json.loads(ctx_path.read_text(encoding="utf-8"))
                    if str(existing.get("source_hash") or "") == src_hash:
                        return
                except Exception:
                    pass
            try:
                translation = translate_page_context(cfg, page_text, target_lang=lang, delimiter="\n---\n")
                ctx_path.write_text(
                    json.dumps(
                        {
                            "page_id": page_id,
                            "target_lang": lang,
                            "model": cfg.model,
                            "prompt_version": cfg.prompt_version,
                            "source_hash": src_hash,
                            "source_text": page_text,
                            "context_translation": translation,
                            "created_at": _utc_now(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            except Exception as e:
                # Persist page-level error (best-effort)
                ctx_path.write_text(
                    json.dumps(
                        {
                            "page_id": page_id,
                            "target_lang": lang,
                            "model": cfg.model,
                            "prompt_version": cfg.prompt_version,
                            "source_hash": src_hash,
                            "error": str(e),
                            "created_at": _utc_now(),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            return

        if job_type == "region_draft":
            region_id = str(job.get("region_id") or "")
            source_text = str(job.get("source_text") or "").strip()
            if not region_id or not source_text:
                return
            src_hash = self._hash_text(source_text, cfg.model, lang, cfg.prompt_version, "region_draft")
            payload = self._load_regions_translation_payload(project_id, page_id, lang)
            existing = ((payload.get("items") or {}).get(region_id) or {})
            if str(existing.get("draft_source_hash") or "") == src_hash and str(existing.get("status_draft") or "") == "done":
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
            except Exception as e:
                self._upsert_region_translation(
                    project_id,
                    page_id,
                    region_id,
                    lang=lang,
                    source_text=source_text,
                    patch={"status_draft": "error", "error_draft": str(e)},
                )
            return

        if job_type == "region_refine":
            region_id = str(job.get("region_id") or "")
            source_text = str(job.get("source_text") or "").strip()
            if not region_id or not source_text:
                return
            ctx_path = self._page_context_path(project_id, page_id, lang)
            if not ctx_path.exists():
                # Context not ready yet; requeue with a small delay.
                time.sleep(0.5)
                self._translate_queue.put(job)
                return
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
            except Exception:
                time.sleep(0.5)
                self._translate_queue.put(job)
                return
            context_translation = str(ctx.get("context_translation") or "").strip()
            if not context_translation:
                return

            src_hash = self._hash_text(source_text, context_translation, cfg.model, lang, cfg.prompt_version, "region_refine")
            payload = self._load_regions_translation_payload(project_id, page_id, lang)
            existing = ((payload.get("items") or {}).get(region_id) or {})
            if str(existing.get("refine_source_hash") or "") == src_hash and str(existing.get("status_refine") or "") == "done":
                return

            self._upsert_region_translation(
                project_id,
                page_id,
                region_id,
                lang=lang,
                source_text=source_text,
                patch={"status_refine": "running", "refine_source_hash": src_hash, "error_refine": None},
            )
            try:
                translation = translate_region_refine(cfg, source_text, context_translation, target_lang=lang)
                self._upsert_region_translation(
                    project_id,
                    page_id,
                    region_id,
                    lang=lang,
                    source_text=source_text,
                    patch={"refined_translation": translation, "status_refine": "done"},
                )
            except Exception as e:
                self._upsert_region_translation(
                    project_id,
                    page_id,
                    region_id,
                    lang=lang,
                    source_text=source_text,
                    patch={"status_refine": "error", "error_refine": str(e)},
                )
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
        runtime_probe = self._probe_ocr_runtime()
        effective_device = (
            "cuda"
            if requested_device == "cuda" and bool(runtime_probe.get("ort_cuda_available"))
            else "cpu"
        )
        status["ocr_runtime"] = {
            "requested_device": requested_device,
            "effective_device": effective_device,
            **runtime_probe,
        }
        status["updated_at"] = _utc_now()
        status["translate_status"] = "running"
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
                region_ready_callback=lambda page_id, rec: self.enqueue_region_draft(
                    project_id, str(page_id), rec, lang="ru"
                ),
                page_done_callback=lambda page_id, regions: self.enqueue_page_context_and_refine(
                    project_id, str(page_id), regions, lang="ru"
                ),
            )
            status["status"] = "done"
            status["pages"] = len(report.get("pages", []))
            status["updated_at"] = _utc_now()
            status["translate_status"] = "running"
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
        ctx_path = self._page_context_path(project_id, page_id, lang)
        ctx_ready = ctx_path.exists()
        ctx_error = None
        if ctx_ready:
            try:
                ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                ctx_error = ctx.get("error")
                if ctx_error:
                    ctx_ready = False
            except Exception:
                ctx_ready = False
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        items = payload.get("items") or {}
        draft_done = sum(1 for it in items.values() if str(it.get("status_draft") or "") == "done")
        draft_err = sum(1 for it in items.values() if str(it.get("status_draft") or "") == "error")
        refine_done = sum(1 for it in items.values() if str(it.get("status_refine") or "") == "done")
        refine_err = sum(1 for it in items.values() if str(it.get("status_refine") or "") == "error")
        return {
            "page_id": page_id,
            "lang": lang,
            "page_context_ready": bool(ctx_ready),
            "page_context_error": ctx_error,
            "regions_total": int(len(items)),
            "draft_done": int(draft_done),
            "draft_error": int(draft_err),
            "refine_done": int(refine_done),
            "refine_error": int(refine_err),
        }

    def load_region_translation(self, project_id: str, page_id: str, region_id: str, *, lang: str = "ru") -> dict:
        payload = self._load_regions_translation_payload(project_id, page_id, lang)
        entry = (payload.get("items") or {}).get(region_id) or {}
        return {
            "page_id": page_id,
            "region_id": region_id,
            "lang": lang,
            "draft_translation": entry.get("draft_translation"),
            "refined_translation": entry.get("refined_translation"),
            "status_draft": entry.get("status_draft") or ("pending" if not entry else "unknown"),
            "status_refine": entry.get("status_refine") or ("pending" if not entry else "unknown"),
            "error_draft": entry.get("error_draft"),
            "error_refine": entry.get("error_refine"),
            "updated_at": entry.get("updated_at"),
        }

    def _build_recognition_config(self, options: GenerateOptions) -> RecognitionConfig:
        mode = (options.ocr_mode or "eco").lower()
        ocr_device = (options.ocr_device or "cpu").strip().lower()
        if ocr_device not in {"cpu", "cuda"}:
            ocr_device = "cpu"
        is_cuda = ocr_device == "cuda"
        cpu = int(os.cpu_count() or 4)
        auto_workers = max(2, min(8, cpu // 2))
        workers = options.ocr_workers if options.ocr_workers is not None else auto_workers
        workers = max(1, int(workers))
        if mode == "max":
            max_workers = min(8, workers)
            if is_cuda:
                max_workers = min(8, max(2, workers))
            return RecognitionConfig(
                backend="paddleocr",
                mode="hybrid",
                use_multipass=True,
                parallel_ocr=True,
                max_workers=max_workers,
                batch_size=20 if is_cuda else 24,
                parallel_min_regions=10 if is_cuda else 12,
                cascade_probe_variants=2,
                backend_cascade=bool(options.allow_fallback),
                allow_fallback=bool(options.allow_fallback),
                debug_raw_results=True,
                ocr_device=ocr_device,
            )
        if mode == "balanced":
            max_workers = min(4, workers)
            if is_cuda:
                max_workers = min(6, max(2, workers))
            return RecognitionConfig(
                backend="paddleocr",
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
                debug_raw_results=True,
                ocr_device=ocr_device,
            )
        # eco (default): keep CPU usage low for personal machines
        if is_cuda:
            return RecognitionConfig(
                backend="paddleocr",
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
                debug_raw_results=True,
                ocr_device=ocr_device,
            )
        return RecognitionConfig(
            backend="paddleocr",
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
            debug_raw_results=True,
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
        return json.loads(status_path.read_text(encoding="utf-8"))

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
        path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
