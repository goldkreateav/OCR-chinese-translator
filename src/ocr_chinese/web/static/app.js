import "./tailwind.css";
import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import htm from "htm";

const html = htm.bind(React.createElement);

function dbg() {}

const APP_BUILD = globalThis.APP_BUILD ?? "-";

const POLL_STATUS_MS = 1000;
const POLL_TRANSLATE_MS = 2000;
const ETA_WARMUP_MS = 5000;
const ETA_ALPHA = 0.32;

function routeFromPath(pathname) {
  return pathname === "/import" ? "import" : "workspace";
}

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return "уточняется";
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}ч ${m}м`;
  if (m > 0) return `${m}м ${s}с`;
  return `${s}с`;
}

function formatStage(stage) {
  if (!stage) return "ожидание";
  const value = String(stage).toLowerCase();
  if (value.includes("mask")) return "Mask";
  if (value.includes("ocr")) return "OCR";
  if (value.includes("translate")) return "Translate";
  if (value.includes("render")) return "Render";
  return value;
}

function toPct(cur, total) {
  if (!Number.isFinite(cur) || !Number.isFinite(total) || total <= 0) return 0;
  return Math.min(100, Math.max(0, Math.round((cur * 100) / total)));
}

function getPolygonStyle(conf) {
  if (!Number.isFinite(conf) || conf <= 0) {
    return { fill: "rgba(96,97,100,0.22)", stroke: "#292523" };
  }
  if (conf < 0.4) {
    return { fill: "rgba(201,37,44,0.20)", stroke: "#C9252C" };
  }
  if (conf < 0.7) {
    return { fill: "rgba(255,200,5,0.22)", stroke: "#FFC805" };
  }
  return { fill: "rgba(1,70,55,0.20)", stroke: "#014637" };
}

function pointsToAttr(points) {
  return (points || []).map((xy) => `${xy[0]},${xy[1]}`).join(" ");
}

function pointsToAttrScaled(points, sx, sy) {
  const sxx = Number.isFinite(sx) && sx > 0 ? sx : 1;
  const syy = Number.isFinite(sy) && sy > 0 ? sy : 1;
  return (points || [])
    .map((xy) => `${Number(xy?.[0] || 0) * sxx},${Number(xy?.[1] || 0) * syy}`)
    .join(" ");
}

function pointsToAttrNormalize(points, sx, sy, ox, oy) {
  const sxx = Number.isFinite(sx) && sx > 0 ? sx : 1;
  const syy = Number.isFinite(sy) && sy > 0 ? sy : 1;
  const oxx = Number.isFinite(ox) ? ox : 0;
  const oyy = Number.isFinite(oy) ? oy : 0;
  return (points || [])
    .map((xy) => `${(Number(xy?.[0] || 0) - oxx) * sxx},${(Number(xy?.[1] || 0) - oyy) * syy}`)
    .join(" ");
}

function readImageGeom(img) {
  if (!img) return null;
  const rect = img.getBoundingClientRect();
  const parentRect = (img.parentElement || img).getBoundingClientRect();
  const w = Number(img.naturalWidth || 0);
  const h = Number(img.naturalHeight || 0);
  const cw = Number(rect.width || 0);
  const ch = Number(rect.height || 0);
  const pw = Number(parentRect.width || 0);
  const ph = Number(parentRect.height || 0);
  if (!(w > 0 && h > 0 && cw > 0 && ch > 0)) return null;
  return { w, h, cw, ch, pw, ph };
}

function upsertGeom(prev, pageId, nextGeom) {
  if (!pageId || !nextGeom) return prev;
  const cur = prev[pageId];
  if (
    cur &&
    cur.w === nextGeom.w &&
    cur.h === nextGeom.h &&
    cur.cw === nextGeom.cw &&
    cur.ch === nextGeom.ch &&
    cur.pw === nextGeom.pw &&
    cur.ph === nextGeom.ph
  ) {
    return prev;
  }
  return {
    ...prev,
    [pageId]: {
      ...(cur || {}),
      ...nextGeom,
    },
  };
}

function inferViewBox(regions) {
  let maxX = 1000;
  let maxY = 1000;
  for (const region of regions || []) {
    for (const point of region.polygon || []) {
      maxX = Math.max(maxX, Number(point[0] || 0));
      maxY = Math.max(maxY, Number(point[1] || 0));
    }
  }
  return `0 0 ${Math.max(1, Math.ceil(maxX))} ${Math.max(1, Math.ceil(maxY))}`;
}

function parseJsonSafe(raw) {
  try {
    return JSON.parse(raw);
  } catch (_) {
    return null;
  }
}

function buildOfflineAssets(report, pageId) {
  const regions = report?.regionsByPage?.[pageId] || [];
  const tPage = report?.translationsByPage?.[pageId]?.regions || {};
  const regionsAug = regions.map((r) => {
    const t = tPage?.[r.region_id] || {};
    return {
      ...r,
      page_id: r.page_id || pageId,
      draft_translation: t.draft_translation,
      refined_translation: t.refined_translation,
      status_draft: t.status_draft,
      status_refine: t.status_refine,
    };
  });
  return { page_id: pageId, regions: regionsAug };
}

function countMissingTranslations(report) {
  const pages = report?.pages || [];
  const regionsByPage = report?.regionsByPage || {};
  const tByPage = report?.translationsByPage || {};
  let missing = 0;
  for (const p of pages) {
    const pageId = String(p?.page_id || "");
    if (!pageId) continue;
    const regions = regionsByPage?.[pageId] || [];
    const tRegions = tByPage?.[pageId]?.regions || {};
    for (const r of regions || []) {
      const rid = r?.region_id;
      if (!rid) continue;
      const text = String(r?.text || "").trim();
      if (!text || text === "Текст не найден") continue;
      const t = tRegions?.[rid] || {};
      const refined = String(t?.refined_translation || "").trim();
      if (!refined) missing += 1;
    }
  }
  return missing;
}

function useEtaModel() {
  const refs = useRef({
    ocr: {},
    draft: {},
    refine: {},
  });

  function updateRate(bucket, pageId, current) {
    const state = refs.current[bucket][pageId] || {
      lastAt: Date.now(),
      lastVal: 0,
      rate: 0,
      startedAt: Date.now(),
      maxVal: 0,
    };
    const now = Date.now();
    const dt = Math.max(0.001, (now - state.lastAt) / 1000);
    const dv = Math.max(0, current - state.lastVal);
    const instant = dv / dt;
    const nextMax = Math.max(Number(state.maxVal || 0), Number(current || 0));
    const elapsed = Math.max(0.001, (now - (state.startedAt || now)) / 1000);
    const cumulative = nextMax / elapsed;

    // Blend cumulative average (stable) with EWMA of instant (responsive).
    const ewma = state.rate > 0 ? state.rate * (1 - ETA_ALPHA) + instant * ETA_ALPHA : instant;
    const nextRate = 0.72 * cumulative + 0.28 * ewma;
    refs.current[bucket][pageId] = {
      lastAt: now,
      lastVal: current,
      rate: nextRate,
      startedAt: state.startedAt || now,
      maxVal: nextMax,
    };
    dbg("H9", "static/app.js:updateRate", "rate update", {
      bucket,
      pageId,
      current,
      dt_s: dt,
      dv: dv,
      instant,
      cumulative,
      ewma,
      nextRate,
      age_ms: now - (state.startedAt || now),
    });
  }

  function estimate(bucket, pageId, current, total) {
    const state = refs.current[bucket][pageId];
    if (!state || !Number.isFinite(total) || total <= 0) return null;
    if (Date.now() - state.startedAt < ETA_WARMUP_MS) return null;
    if (!Number.isFinite(state.rate) || state.rate <= 0.01) return null;
    const remaining = Math.max(0, total - current);
    return remaining / state.rate;
  }

  function globalRate(bucket) {
    const items = Object.values(refs.current[bucket] || {});
    const rates = items.map((s) => Number(s?.rate || 0)).filter((r) => Number.isFinite(r) && r > 0.01);
    if (rates.length === 0) return null;
    rates.sort((a, b) => a - b);
    return rates[Math.floor(rates.length / 2)];
  }

  function estimateWithFallback(bucket, pageId, current, total, fallbackRate) {
    const direct = estimate(bucket, pageId, current, total);
    if (direct != null) return direct;
    const rate = fallbackRate || globalRate(bucket);
    if (!Number.isFinite(rate) || rate <= 0.01) return null;
    if (!Number.isFinite(total) || total <= 0) return null;
    const remaining = Math.max(0, total - current);
    const eta = remaining / rate;
    if (eta > 60 * 60) {
      dbg("H10", "static/app.js:estimateWithFallback", "huge ETA detected", {
        bucket,
        pageId,
        current,
        total,
        remaining,
        rate,
        eta_s: eta,
      });
    }
    return eta;
  }

  return { updateRate, estimate, estimateWithFallback, globalRate };
}

function App() {
  const [route, setRoute] = useState(routeFromPath(window.location.pathname));
  const [version, setVersion] = useState("-");
  const [connectionOk, setConnectionOk] = useState(true);
  const [statusText, setStatusText] = useState("Idle");
  const [projectId, setProjectId] = useState(null);
  const [filename, setFilename] = useState("");
  const [pages, setPages] = useState([]);
  const [pageIndex, setPageIndex] = useState(0);
  const [statusPayload, setStatusPayload] = useState(null);
  const [runtimeInfo, setRuntimeInfo] = useState(null);
  const [translationByPage, setTranslationByPage] = useState({});
  const [pageTranslationsById, setPageTranslationsById] = useState({}); // pageId -> {items: {region_id -> entry}}
  const [assetsByPage, setAssetsByPage] = useState({});
  const [maskOkByPage, setMaskOkByPage] = useState({});
  const [maskNonce, setMaskNonce] = useState(0);
  const [pageGeomById, setPageGeomById] = useState({});
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [regionTranslation, setRegionTranslation] = useState({
    statusLabel: "pending",
    text: "",
    error: "",
  });
  const [generateInFlight, setGenerateInFlight] = useState(false);
  const [importReport, setImportReport] = useState(null);
  const [importError, setImportError] = useState("");
  const [importPdf, setImportPdf] = useState(null);
  const [importProjectId, setImportProjectId] = useState(null);
  const [importPages, setImportPages] = useState([]);
  const [importPageIndex, setImportPageIndex] = useState(0);
  const [importAssetsByPage, setImportAssetsByPage] = useState({});
  const [importPageGeomById, setImportPageGeomById] = useState({});
  const [uploadFile, setUploadFile] = useState(null);
  const [dpi, setDpi] = useState(400);
  const [ocrMode, setOcrMode] = useState("eco");
  const [ocrDevice, setOcrDevice] = useState("cpu");

  const etaModel = useEtaModel();
  const routeRef = useRef(route);
  routeRef.current = route;
  const etaJumpRef = useRef({}); // pageId -> {lastAt, lastEta}
  const pageSetRef = useRef({ key: "" });
  const etaDisplayRef = useRef({}); // key -> {eta, at}
  const workspaceImageRef = useRef(null);
  const importImageRef = useRef(null);

  useEffect(() => {
    dbg("H_build", "static/app.js:App", "app mounted", { build: APP_BUILD, path: window.location.pathname });
  }, []);

  function smoothEta(key, targetEta, kind) {
    if (targetEta == null || !Number.isFinite(targetEta) || targetEta < 0) return null;
    const now = Date.now();
    const prev = etaDisplayRef.current[key];
    if (!prev || !Number.isFinite(prev.eta)) {
      etaDisplayRef.current[key] = { eta: targetEta, at: now };
      return targetEta;
    }
    const dt = Math.max(0.001, (now - (prev.at || now)) / 1000);
    const from = prev.eta;
    const to = targetEta;

    // Heavier smoothing for drops (your request) and for early instability.
    const baseTauUp = kind === "page" ? 10.0 : 12.0;
    const baseTauDown = kind === "page" ? 18.0 : 22.0;
    const tau = to < from ? baseTauDown : baseTauUp; // seconds
    const alpha = 1 - Math.exp(-dt / tau);

    // Clamp per-update change so it can't "rocket" in either direction.
    const maxRatioUp = 1.22;
    const maxRatioDown = 0.94;
    const cappedTo =
      to > from ? Math.min(to, from * maxRatioUp) : Math.max(to, from * maxRatioDown);

    const next = from + (cappedTo - from) * alpha;
    etaDisplayRef.current[key] = { eta: next, at: now };

    if (Math.abs(to - next) / Math.max(1, to) > 0.35) {
      dbg("H13", "static/app.js:smoothEta", "ETA smoothing applied", {
        key,
        kind,
        from,
        target: to,
        cappedTarget: cappedTo,
        next,
        dt_s: dt,
        tau,
        alpha,
      });
    }
    return next;
  }

  const progressPages = statusPayload?.progress_pages || {};
  const stage = formatStage(statusPayload?.stage);
  const statusState = statusPayload?.status || "idle";

  const derivedPageIds = useMemo(() => {
    const fromPages = Array.isArray(pages) ? pages.filter(Boolean) : [];
    if (fromPages.length > 0) return fromPages;
    const keys = Object.keys(progressPages || {});
    const pageFromProgress = statusPayload?.progress?.page_id ? [String(statusPayload.progress.page_id)] : [];
    const merged = Array.from(new Set([...keys, ...pageFromProgress])).filter(Boolean);
    merged.sort();
    dbg("H2", "static/app.js:derivedPageIds", "Derived page ids", {
      fromPagesLen: fromPages.length,
      progressPagesKeysHead: keys.slice(0, 5),
      progressPageId: pageFromProgress[0] || null,
      derivedLen: merged.length,
      derivedHead: merged.slice(0, 5),
    });
    return merged;
  }, [pages, progressPages, statusPayload?.progress?.page_id]);

  useEffect(() => {
    const key = derivedPageIds.join(",");
    if (pageSetRef.current.key && pageSetRef.current.key !== key) {
      dbg("H11", "static/app.js:pageSet", "Derived pages changed", {
        prev: pageSetRef.current.key.split(",").filter(Boolean).slice(0, 10),
        next: derivedPageIds.slice(0, 10),
        nextLen: derivedPageIds.length,
        stage,
      });
    }
    pageSetRef.current.key = key;
  }, [derivedPageIds, stage]);

  const currentPageId = derivedPageIds[pageIndex] || derivedPageIds[0] || null;
  const currentAssets = currentPageId ? assetsByPage[currentPageId] : null;
  const importCurrentPageId = importPages[importPageIndex] || null;

  function syncWorkspaceGeom(img = workspaceImageRef.current) {
    if (!currentPageId || !img) return;
    const geom = readImageGeom(img);
    if (!geom) return;
    setPageGeomById((prev) => upsertGeom(prev, currentPageId, geom));
  }

  function syncImportGeom(pageId, img = importImageRef.current) {
    if (!pageId || !img) return;
    const geom = readImageGeom(img);
    if (!geom) return;
    setImportPageGeomById((prev) => upsertGeom(prev, pageId, geom));
  }

  const avgRegionsPerPage = useMemo(() => {
    const totals = Object.values(progressPages || {})
      .map((p) => Number(p?.total_regions || 0))
      .filter((n) => Number.isFinite(n) && n > 0);
    if (totals.length === 0) return null;
    const sum = totals.reduce((a, b) => a + b, 0);
    return sum / totals.length;
  }, [progressPages]);

  useEffect(() => {
    const handlePop = () => setRoute(routeFromPath(window.location.pathname));
    window.addEventListener("popstate", handlePop);
    return () => window.removeEventListener("popstate", handlePop);
  }, []);

  function goTo(nextRoute) {
    if (nextRoute === routeRef.current) return;
    const nextPath = nextRoute === "import" ? "/import" : "/";
    window.history.pushState({}, "", nextPath);
    setRoute(nextRoute);
  }

  useEffect(() => {
    (async () => {
      try {
        const resp = await fetch("/api/version");
        if (!resp.ok) return;
        const payload = await resp.json();
        const appVer = payload.app_version || "unknown";
        const bridge = payload.quality_bridge_enabled ? "bridge:on" : "bridge:off";
        setVersion(`${appVer} (${bridge})`);
        setRuntimeInfo(payload || null);
      } catch (_) {
        // noop
      }
    })();
  }, []);

  const derivedRequestedDevice =
    String(statusPayload?.ocr_runtime?.requested_device || ocrDevice || "cpu").toLowerCase() === "cuda" ? "cuda" : "cpu";
  const derivedCudaAvailable = Boolean(
    statusPayload?.ocr_runtime?.ort_cuda_available ?? runtimeInfo?.ort_cuda_available
  );
  const derivedEffectiveDevice = statusPayload?.ocr_runtime?.effective_device
    ? String(statusPayload.ocr_runtime.effective_device)
    : derivedRequestedDevice === "cuda" && derivedCudaAvailable
      ? "cuda"
      : "cpu";

  const webEnableRetryOcr = Boolean(runtimeInfo?.web_enable_retry_ocr);

  async function fetchStatus() {
    if (!projectId) return;
    try {
      const resp = await fetch(`/api/projects/${projectId}/status`);
      if (!resp.ok) return;
      const payload = await resp.json();
      dbg("H1", "static/app.js:fetchStatus", "Fetched status payload", {
        httpOk: resp.ok,
        status: payload?.status,
        stage: payload?.stage,
        progressKeys: payload?.progress ? Object.keys(payload.progress) : null,
        progressPageId: payload?.progress?.page_id ?? null,
        progressPagesCount: payload?.progress_pages ? Object.keys(payload.progress_pages).length : 0,
        progressPagesKeysHead: payload?.progress_pages ? Object.keys(payload.progress_pages).slice(0, 3) : [],
      });
      setConnectionOk(true);
      setStatusPayload(payload);
      const tState = String(payload?.translation?.state || "");
      const stageLabel = formatStage(payload.stage);
      if (payload.status === "running") {
        setStatusText(`Running (${stageLabel})`);
      } else if (payload.status === "done" && tState && tState !== "done") {
        setStatusText(`Done (${stageLabel})`);
      } else {
        setStatusText(payload.status);
      }
      const pageMap = payload.progress_pages || {};
      Object.entries(pageMap).forEach(([pageId, item]) => {
        etaModel.updateRate("ocr", pageId, Number(item.current_region || 0));
      });
      if (pages.length === 0 && Number(payload.pages || 0) > 0) {
        await loadPages(projectId);
      }
    } catch (_) {
      setConnectionOk(false);
      dbg("H4", "static/app.js:fetchStatus", "Status fetch exception", { projectId });
    }
  }

  async function loadPages(pid) {
    const resp = await fetch(`/api/projects/${pid}/pages`);
    if (!resp.ok) return;
    const payload = await resp.json();
    const list = payload.pages || [];
    setPages(list);
    if (list.length > 0) {
      setPageIndex(0);
      await ensureAssets(pid, list[0]);
    }
  }

  async function ensureAssets(pid, pageId, opts = null) {
    if (!pid || !pageId) return;
    const force = Boolean(opts?.force);
    if (!force && assetsByPage[pageId]) return;
    const resp = await fetch(`/api/projects/${pid}/pages/${pageId}/assets`);
    if (!resp.ok) return;
    const payload = await resp.json();
    setAssetsByPage((prev) => ({ ...prev, [pageId]: payload }));
  }

  useEffect(() => {
    if (!projectId) return undefined;
    fetchStatus();
    const timer = window.setInterval(fetchStatus, POLL_STATUS_MS);
    return () => window.clearInterval(timer);
  }, [projectId, pages.length]);

  useEffect(() => {
    if (!projectId || pages.length === 0) return undefined;
    const timer = window.setInterval(async () => {
      const entries = [];
      for (const pageId of pages) {
        const hasFinished = translationByPage[pageId]?.regions_total > 0 &&
          translationByPage[pageId]?.regions_total === translationByPage[pageId]?.refine_done + translationByPage[pageId]?.refine_error;
        if (hasFinished) continue;
        try {
          const resp = await fetch(`/api/projects/${projectId}/pages/${pageId}/translations/status?lang=ru`);
          if (!resp.ok) continue;
          const payload = await resp.json();
          entries.push([pageId, payload]);
          etaModel.updateRate("draft", pageId, Number(payload.draft_done || 0));
          etaModel.updateRate("refine", pageId, Number(payload.refine_done || 0));
        } catch (_) {
          // noop
        }
      }
      if (entries.length > 0) {
        setTranslationByPage((prev) => {
          const next = { ...prev };
          for (const [pageId, payload] of entries) next[pageId] = payload;
          return next;
        });
      }
    }, POLL_TRANSLATE_MS);
    return () => window.clearInterval(timer);
  }, [projectId, pages, translationByPage]);

  useEffect(() => {
    if (!projectId || pages.length === 0) return undefined;
    const timer = window.setInterval(async () => {
      const updates = [];
      for (const pageId of pages) {
        try {
          const resp = await fetch(`/api/projects/${projectId}/pages/${pageId}/translations?lang=ru`);
          if (!resp.ok) continue;
          const payload = await resp.json();
          updates.push([pageId, payload]);
        } catch (_) {
          // noop
        }
      }
      if (updates.length > 0) {
        setPageTranslationsById((prev) => {
          const next = { ...prev };
          for (const [pageId, payload] of updates) next[pageId] = payload;
          return next;
        });
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [projectId, pages]);

  useEffect(() => {
    if (!projectId || !currentPageId) return;
    ensureAssets(projectId, currentPageId);
  }, [projectId, currentPageId]);

  useEffect(() => {
    if (route !== "workspace") return undefined;
    if (!projectId || !currentPageId) return undefined;
    if (statusPayload?.status !== "running") return undefined;
    const st = String(statusPayload?.stage || "").toLowerCase();
    if (!st.includes("ocr") && !st.includes("mask")) return undefined;
    const timer = window.setInterval(() => {
      ensureAssets(projectId, currentPageId, { force: true });
    }, 1200);
    return () => window.clearInterval(timer);
  }, [route, projectId, currentPageId, statusPayload?.status, statusPayload?.stage]);

  useEffect(() => {
    if (route !== "workspace" || !currentPageId) return undefined;
    const img = workspaceImageRef.current;
    if (!img) return undefined;
    const sync = () => syncWorkspaceGeom(img);
    sync();
    const onResize = () => sync();
    window.addEventListener("resize", onResize);
    let ro = null;
    if (typeof window.ResizeObserver === "function") {
      ro = new window.ResizeObserver(() => sync());
      ro.observe(img);
    }
    return () => {
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
    };
  }, [route, currentPageId, currentAssets?.image_url]);

  useEffect(() => {
    if (route !== "import" || !importCurrentPageId) return undefined;
    const img = importImageRef.current;
    if (!img) return undefined;
    const sync = () => syncImportGeom(importCurrentPageId, img);
    sync();
    const onResize = () => sync();
    window.addEventListener("resize", onResize);
    let ro = null;
    if (typeof window.ResizeObserver === "function") {
      ro = new window.ResizeObserver(() => sync());
      ro.observe(img);
    }
    return () => {
      window.removeEventListener("resize", onResize);
      if (ro) ro.disconnect();
    };
  }, [route, importCurrentPageId, importProjectId]);

  useEffect(() => {
    // When stage advances, retry loading mask for current page.
    if (!currentPageId) return;
    setMaskNonce((n) => n + 1);
    setMaskOkByPage((prev) => ({ ...prev, [currentPageId]: true }));
  }, [currentPageId, statusPayload?.stage]);

  useEffect(() => {
    const effectivePid = projectId || importProjectId;
    if (!effectivePid || !selectedRegion?.region_id || !selectedRegion?.page_id) return undefined;
    const pageId = String(selectedRegion.page_id);
    const regionId = String(selectedRegion.region_id);

    const apply = (entry) => {
      if (!entry) {
        setRegionTranslation({ statusLabel: "pending", text: "", error: "" });
        return;
      }
      if (entry.refined_translation) {
        setRegionTranslation({
          statusLabel: `refined (${entry.status_refine || "done"})`,
          text: entry.refined_translation,
          error: "",
        });
        return;
      }
      if (entry.draft_translation) {
        setRegionTranslation({
          statusLabel: `draft (${entry.status_draft || "done"}) → refining`,
          text: entry.draft_translation,
          error: "",
        });
        return;
      }
      if (entry.error_draft || entry.error_refine) {
        setRegionTranslation({
          statusLabel: "error",
          text: "",
          error: String(entry.error_refine || entry.error_draft || "Translation failed"),
        });
        return;
      }
      setRegionTranslation({
        statusLabel: `pending (draft:${entry.status_draft || "pending"}, refine:${entry.status_refine || "pending"})`,
        text: "",
        error: "",
      });
    };

    // Workspace: use page-level cache from backend.
    if (projectId && pageTranslationsById?.[pageId]?.items) {
      apply(pageTranslationsById[pageId].items?.[regionId] || null);
      return undefined;
    }

    // Import viewer: use data embedded into region object.
    if (!projectId) {
      apply({
        draft_translation: selectedRegion.draft_translation,
        refined_translation: selectedRegion.refined_translation,
        status_draft: selectedRegion.status_draft,
        status_refine: selectedRegion.status_refine,
        error_draft: selectedRegion.error_draft,
        error_refine: selectedRegion.error_refine,
      });
      return undefined;
    }

    // Fallback (should be rare): do one direct fetch, no polling.
    (async () => {
      try {
        const resp = await fetch(`/api/projects/${effectivePid}/pages/${pageId}/translations/region/${regionId}?lang=ru`);
        if (!resp.ok) return;
        const payload = await resp.json();
        apply(payload);
      } catch (_) {}
    })();
    return undefined;
  }, [projectId, importProjectId, selectedRegion?.region_id, selectedRegion?.page_id, pageTranslationsById]);

  async function handleUpload() {
    if (!uploadFile) return;
    const fd = new FormData();
    fd.append("file", uploadFile);
    setStatusText("Uploading PDF...");
    const resp = await fetch("/api/projects", { method: "POST", body: fd });
    if (!resp.ok) {
      setStatusText("Upload failed");
      alert(await resp.text());
      return;
    }
    const payload = await resp.json();
    setProjectId(payload.project_id);
    setFilename(payload.filename || uploadFile.name || "");
    setPages([]);
    setAssetsByPage({});
    setTranslationByPage({});
    setStatusText("Uploaded");
  }

  async function handleGenerate() {
    if (!projectId) return;
    setGenerateInFlight(true);
    setStatusText("Generating...");
    const resp = await fetch(`/api/projects/${projectId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dpi: Number(dpi || 400),
        ocr_mode: ocrMode,
        ocr_device: ocrDevice,
      }),
    });
    if (!resp.ok) {
      setGenerateInFlight(false);
      setStatusText("Generation failed");
      alert(await resp.text());
      return;
    }
    // Generation runs in background; /status polling reflects real completion.
    await loadPages(projectId);
    setGenerateInFlight(false);
    setStatusText("Started");
  }

  async function handleRetryRegion() {
    if (!projectId || !selectedRegion?.region_id || !selectedRegion?.page_id) return;
    const resp = await fetch(`/api/projects/${projectId}/regions/${selectedRegion.region_id}/retry`, {
      method: "POST",
    });
    if (!resp.ok) {
      alert(await resp.text());
      return;
    }
    const payload = await resp.json();
    const pageId = selectedRegion.page_id;
    setAssetsByPage((prev) => {
      const pageAssets = prev[pageId];
      if (!pageAssets) return prev;
      const nextRegions = (pageAssets.regions || []).map((r) => {
        if (r.region_id !== payload.region_id) return r;
        return {
          ...r,
          text: payload.text,
          ocr_confidence: payload.ocr_confidence,
          ocr_variant: payload.ocr_variant,
          ocr_score: payload.ocr_score,
        };
      });
      return {
        ...prev,
        [pageId]: { ...pageAssets, regions: nextRegions },
      };
    });
    setSelectedRegion((prev) => (prev ? { ...prev, ...payload } : prev));
  }

  async function exportReport() {
    if (!projectId) return;
    const report = {
      meta: {
        schema_version: "1.0.0",
        exported_at: new Date().toISOString(),
        project_id: projectId,
        filename: filename || null,
        app_version: version,
        dpi: Number(dpi || 400),
        ocr_mode: String(ocrMode || "eco"),
        ocr_device: String(ocrDevice || "cpu"),
      },
      pages: [],
      regionsByPage: {},
      translationsByPage: {},
    };
    const pagesResp = await fetch(`/api/projects/${projectId}/pages`);
    if (!pagesResp.ok) {
      alert("Не удалось получить список страниц.");
      return;
    }
    const pagesPayload = await pagesResp.json();
    const pageIds = pagesPayload.pages || [];
    report.pages = pageIds.map((page_id) => ({ page_id }));

    for (const pageId of pageIds) {
      const regionsResp = await fetch(`/api/projects/${projectId}/pages/${pageId}/regions`);
      const regionsPayload = regionsResp.ok ? await regionsResp.json() : { regions: [] };
      const regions = regionsPayload.regions || [];
      report.regionsByPage[pageId] = regions;

      const tStatusResp = await fetch(`/api/projects/${projectId}/pages/${pageId}/translations/status?lang=ru`);
      const tStatusPayload = tStatusResp.ok ? await tStatusResp.json() : {};

      let translations = {};
      try {
        const tResp = await fetch(`/api/projects/${projectId}/pages/${pageId}/translations?lang=ru`);
        if (tResp.ok) {
          const tPayload = await tResp.json();
          translations = tPayload?.items || {};
        }
      } catch (_) {}
      report.translationsByPage[pageId] = {
        status: tStatusPayload,
        regions: translations,
      };
    }

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ocr-report-${projectId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importReportFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const payload = parseJsonSafe(String(reader.result || ""));
      dbg("H_import_1", "static/app.js:importReportFile", "import json read", {
        hasPayload: !!payload,
        hasPages: !!payload?.pages,
        hasRegionsByPage: !!payload?.regionsByPage,
        pagesLen: Array.isArray(payload?.pages) ? payload.pages.length : null,
      });
      if (!payload || !payload.pages || !payload.regionsByPage) {
        setImportError("Неверный формат отчёта: ожидается JSON export schema v1.0.0.");
        setImportReport(null);
        return;
      }
      setImportError("");
      setImportReport(payload);
      dbg("H_import_2", "static/app.js:importReportFile", "import report set", {
        pagesLen: Array.isArray(payload.pages) ? payload.pages.length : null,
        meta: payload?.meta ? Object.keys(payload.meta) : null,
      });
    };
    reader.readAsText(file, "utf-8");
  }

  async function openImportedViewer() {
    if (!importReport || !importPdf) return;
    const metaDpi = Number(importReport?.meta?.dpi || 400);
    const metaMode = String(importReport?.meta?.ocr_mode || "eco").toLowerCase();
    const clamp = metaMode === "eco" ? 360 : (metaMode === "balanced" ? 380 : metaDpi);
    const importDpi = Math.min(metaDpi, clamp);
    const fd = new FormData();
    fd.append("file", importPdf);
    fd.append("dpi", String(importDpi));
    dbg("H_import_3", "static/app.js:openImportedViewer", "start import render", {
      pdfName: importPdf?.name,
      pages: importReport?.pages?.length || 0,
      importDpi,
      metaDpi,
      metaMode,
    });
    const resp = await fetch("/api/import/projects", {
      method: "POST",
      body: fd,
    });
    if (!resp.ok) {
      const t = await resp.text();
      setImportError(`Не удалось отрендерить PDF: ${t}`);
      dbg("H_import_4", "static/app.js:openImportedViewer", "import render failed", { status: resp.status });
      return;
    }
    const payload = await resp.json();
    const pid = payload.project_id;
    const pages = payload.pages || [];
    setImportProjectId(pid);
    setImportPages(pages);
    setImportPageIndex(0);
    // Build offline assets from report for each page (regions + translations).
    const map = {};
    for (const pageId of pages) {
      map[pageId] = buildOfflineAssets(importReport, pageId);
    }
    setImportAssetsByPage(map);
    dbg("H_import_5", "static/app.js:openImportedViewer", "import viewer ready", { pid, pagesLen: pages.length });
  }

  const rows = useMemo(() => {
    return derivedPageIds.map((pageId) => {
      const o = progressPages[pageId] || {};
      const t = translationByPage[pageId] || {};
      const ocrCur = Number(o.current_region || 0);
      const ocrTotRaw = Number(o.total_regions || 0);
      const ocrTot = ocrTotRaw > 0 ? ocrTotRaw : Number(avgRegionsPerPage || 0);
      const draftDone = Number(t.draft_done || 0);
      const refineDone = Number(t.refine_done || 0);
      const regionsTotalRaw = Number(t.regions_total || 0);
      const regionsTotal = regionsTotalRaw > 0 ? regionsTotalRaw : (ocrTot > 0 ? ocrTot : Number(avgRegionsPerPage || 0));

      const ocrEtaRaw = etaModel.estimateWithFallback("ocr", pageId, ocrCur, ocrTot, null);
      const draftEta = etaModel.estimateWithFallback("draft", pageId, draftDone, regionsTotal, null);
      const refineEta = etaModel.estimateWithFallback("refine", pageId, refineDone, regionsTotal, null);

      const translateEta = (draftEta != null && refineEta != null) ? (draftEta + refineEta) : (draftEta ?? refineEta ?? null);
      const ocrEta = smoothEta(`${pageId}:ocr`, ocrEtaRaw, "stage");
      const translateEtaSmooth = smoothEta(`${pageId}:translate`, translateEta, "stage");
      const pageEtaRaw =
        (ocrEtaRaw != null || translateEta != null) ? ((ocrEtaRaw || 0) + (translateEta || 0)) : null;
      const pageEta = smoothEta(`${pageId}:page`, pageEtaRaw, "page");

      // Log big ETA drops/spikes (page-level).
      if (pageEta != null && Number.isFinite(pageEta)) {
        const now = Date.now();
        const prev = etaJumpRef.current[pageId];
        if (prev && Number.isFinite(prev.lastEta) && prev.lastEta > 0) {
          const ratio = pageEta / prev.lastEta;
          const dt = (now - (prev.lastAt || now)) / 1000;
          if (dt > 0.2 && (ratio < 0.35 || ratio > 2.5)) {
            dbg("H12", "static/app.js:etaJump", "ETA jump detected", {
              pageId,
              prevEta: prev.lastEta,
              nextEta: pageEta,
              ratio,
              dt_s: dt,
              ocrCur,
              ocrTotRaw,
              ocrTot,
              regionsTotalRaw,
              regionsTotal,
              stage,
            });
          }
        }
        etaJumpRef.current[pageId] = { lastAt: now, lastEta: pageEta };
      }

      return {
        pageId,
        ocrPct: toPct(ocrCur, ocrTot || 0),
        translatePct: toPct(refineDone + Number(t.refine_error || 0), regionsTotal || 0),
        ocrLabel: `${ocrCur}/${ocrTotRaw || (ocrTot > 0 ? `~${Math.round(ocrTot)}` : "?")}`,
        translateLabel: `${refineDone}/${regionsTotalRaw || (regionsTotal > 0 ? `~${Math.round(regionsTotal)}` : "?")}`,
        ocrEta,
        translateEta: translateEtaSmooth,
        pageEta,
      };
    });
  }, [derivedPageIds, progressPages, translationByPage, avgRegionsPerPage]);

  const isBusy = generateInFlight || statusState === "running";

  return html`
    <div className="min-h-[100dvh] bg-sollers-white">
      <header className="border-b border-sollers-grayBorder bg-sollers-white">
        <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-4xl tracking-tight text-sollers-graphite font-semibold">
              SOLLERS OCR Workspace
            </h1>
            <p className="text-sm text-sollers-gray mt-1">OCR + перевод, прогресс по страницам, импорт/экспорт отчётов</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              className=${`px-4 py-2 rounded-xl border text-sm transition-transform active:scale-[0.98] ${
                route === "workspace"
                  ? "bg-sollers-orange text-sollers-white border-sollers-orange"
                  : "bg-sollers-white text-sollers-graphite border-sollers-grayBorder"
              }`}
              onClick=${() => goTo("workspace")}
            >
              Workspace
            </button>
            <button
              className=${`px-4 py-2 rounded-xl border text-sm transition-transform active:scale-[0.98] ${
                route === "import"
                  ? "bg-sollers-orange text-sollers-white border-sollers-orange"
                  : "bg-sollers-white text-sollers-graphite border-sollers-grayBorder"
              }`}
              onClick=${() => goTo("import")}
            >
              Import JSON
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-4 md:px-8 py-6 grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-6">
        ${route === "workspace"
          ? html`
              <section className="space-y-5">
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Проект</h2>
                  <div className="mt-4 space-y-4">
                    <label className="block space-y-2">
                      <span className="text-sm text-sollers-gray">PDF файл</span>
                      <input
                        type="file"
                        accept="application/pdf"
                        className="w-full rounded-xl border border-sollers-grayBorder p-2"
                        onChange=${(e) => setUploadFile(e.target.files?.[0] || null)}
                      />
                    </label>
                    <label className="block space-y-2">
                      <span className="text-sm text-sollers-gray">DPI</span>
                      <input
                        type="number"
                        min="72"
                        max="1200"
                        value=${dpi}
                        onChange=${(e) => setDpi(Number(e.target.value || 400))}
                        className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      />
                    </label>
                    <label className="block space-y-2">
                      <span className="text-sm text-sollers-gray">OCR mode</span>
                      <select
                        value=${ocrMode}
                        onChange=${(e) => setOcrMode(e.target.value)}
                        className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      >
                        <option value="eco">eco</option>
                        <option value="balanced">balanced</option>
                        <option value="max">max</option>
                      </select>
                    </label>
                    <label className="block space-y-2">
                      <span className="text-sm text-sollers-gray">OCR device</span>
                      <select
                        value=${ocrDevice}
                        onChange=${(e) => setOcrDevice(e.target.value)}
                        className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      >
                        <option value="cpu">cpu</option>
                        <option value="cuda">cuda (NVIDIA)</option>
                      </select>
                    </label>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <button
                        onClick=${handleUpload}
                        disabled=${!uploadFile}
                        className="px-4 py-2 rounded-xl border border-sollers-orange bg-sollers-orange text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                      >
                        Upload PDF
                      </button>
                      <button
                        onClick=${handleGenerate}
                        disabled=${!projectId || isBusy}
                        className="px-4 py-2 rounded-xl border border-sollers-graphite bg-sollers-graphite text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                      >
                        Generate
                      </button>
                    </div>
                    <button
                      onClick=${exportReport}
                      disabled=${!projectId}
                      className="w-full px-4 py-2 rounded-xl border border-sollers-blue text-sollers-blue disabled:opacity-50 transition-transform active:scale-[0.98]"
                    >
                      Export report.json
                    </button>
                  </div>
                </article>

                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Статус</h2>
                  <div className="mt-3 space-y-2 text-sm">
                    <p><span className="text-sollers-gray">Состояние:</span> <span className="font-medium">${statusText}</span></p>
                    <p><span className="text-sollers-gray">Этап:</span> <span className="font-medium">${stage}</span></p>
                    <p><span className="text-sollers-gray">Project ID:</span> <span className="mono">${projectId || "-"}</span></p>
                    <p><span className="text-sollers-gray">Файл:</span> <span className="mono">${filename || "-"}</span></p>
                    <p><span className="text-sollers-gray">Версия:</span> <span className="mono">${version}</span></p>
                    <p><span className="text-sollers-gray">OCR device (request):</span> <span className="mono">${derivedRequestedDevice}</span></p>
                    <p><span className="text-sollers-gray">OCR device (effective):</span> <span className=${`mono ${derivedEffectiveDevice === "cuda" ? "text-sollers-green" : "text-sollers-red"}`}>${derivedEffectiveDevice}</span></p>
                    <p><span className="text-sollers-gray">ORT CUDA available:</span> <span className=${`mono ${derivedCudaAvailable ? "text-sollers-green" : "text-sollers-red"}`}>${derivedCudaAvailable ? "yes" : "no"}</span></p>
                    <p><span className="text-sollers-gray">Python:</span> <span className="mono">${statusPayload?.ocr_runtime?.python_executable || runtimeInfo?.python_executable || "-"}</span></p>
                    <p>
                      <span className="text-sollers-gray">API:</span>
                      <span className=${`ml-2 font-medium ${connectionOk ? "text-sollers-green" : "text-sollers-red"}`}>
                        ${connectionOk ? "online" : "offline"}
                      </span>
                    </p>
                  </div>
                </article>
              </section>
            `
          : html`
              <section className="space-y-5">
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Импорт отчёта (view-only)</h2>
                  <p className="text-sm text-sollers-gray mt-2">
                    JSON + PDF откроют быстрый viewer без OCR (только рендер страниц).
                  </p>
                  <label className="block space-y-2 mt-4">
                    <span className="text-sm text-sollers-gray">JSON файл отчёта</span>
                    <input
                      type="file"
                      accept="application/json"
                      className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      onChange=${(e) => importReportFile(e.target.files?.[0] || null)}
                    />
                  </label>
                  <label className="block space-y-2 mt-4">
                    <span className="text-sm text-sollers-gray">PDF файл (для рендера страниц)</span>
                    <input
                      type="file"
                      accept="application/pdf"
                      className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      onChange=${(e) => setImportPdf(e.target.files?.[0] || null)}
                    />
                  </label>
                  <button
                    className="mt-4 w-full px-4 py-2 rounded-xl border border-sollers-orange bg-sollers-orange text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                    disabled=${!importReport || !importPdf}
                    onClick=${openImportedViewer}
                  >
                    Open viewer
                  </button>
                  ${importError
                    ? html`<p className="mt-3 text-sm text-sollers-red">${importError}</p>`
                    : null}
                </article>
              </section>
            `}

        <section className="space-y-5">
          ${route === "workspace"
            ? html`
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-lg font-semibold">Прогресс по страницам</h2>
                    <span className="text-sm text-sollers-gray">ETA приблизительная</span>
                  </div>
                  ${derivedPageIds.length === 0
                    ? html`
                        <div className="space-y-2">
                          <div className="skeleton h-4 rounded"></div>
                          <div className="skeleton h-4 rounded w-5/6"></div>
                          <div className="skeleton h-4 rounded w-4/6"></div>
                        </div>
                      `
                    : html`
                        <div className="space-y-3">
                          ${rows.map(
                            (row) => html`
                              <button
                                className=${`w-full text-left rounded-xl border p-3 transition-transform active:scale-[0.98] ${
                                  row.pageId === currentPageId
                                    ? "border-sollers-orange bg-[#FFF6F0]"
                                    : "border-sollers-grayBorder bg-sollers-white"
                                }`}
                                onClick=${() => setPageIndex(Math.max(0, derivedPageIds.indexOf(row.pageId)))}
                              >
                                <div className="flex items-center justify-between gap-2 text-sm">
                                  <span className="mono">${row.pageId}</span>
                                  <span className="text-sollers-gray">
                                    Page ETA: ${formatDuration(row.pageEta)} | OCR: ${formatDuration(row.ocrEta)} | Translate: ${formatDuration(row.translateEta)}
                                  </span>
                                </div>
                                <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-2">
                                  <div>
                                    <div className="flex justify-between text-xs text-sollers-gray">
                                      <span>OCR</span><span>${row.ocrLabel}</span>
                                    </div>
                                    <div className="h-2 bg-sollers-graySoft rounded mt-1">
                                      <div className="h-2 bg-sollers-orange rounded" style=${{ width: `${row.ocrPct}%` }}></div>
                                    </div>
                                  </div>
                                  <div>
                                    <div className="flex justify-between text-xs text-sollers-gray">
                                      <span>Translate</span><span>${row.translateLabel}</span>
                                    </div>
                                    <div className="h-2 bg-sollers-graySoft rounded mt-1">
                                      <div className="h-2 bg-sollers-blue rounded" style=${{ width: `${row.translatePct}%` }}></div>
                                    </div>
                                  </div>
                                </div>
                              </button>
                            `
                          )}
                        </div>
                      `}
                </article>

                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold">Viewer</h2>
                    <div className="flex items-center gap-2">
                      <button
                        className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                        disabled=${pageIndex <= 0}
                        onClick=${() => setPageIndex((prev) => Math.max(0, prev - 1))}
                      >
                        Prev
                      </button>
                      <span className="mono text-sm">${currentPageId || "-"}</span>
                      <button
                        className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                        disabled=${pageIndex >= pages.length - 1}
                        onClick=${() => setPageIndex((prev) => Math.min(pages.length - 1, prev + 1))}
                      >
                        Next
                      </button>
                    </div>
                  </div>

                  ${currentAssets
                    ? html`
                        <div className="viewer-surface">
                          <img
                            src=${`${currentAssets.image_url}?t=${Date.now()}`}
                            alt="Rendered page"
                            className="viewer-page"
                            ref=${workspaceImageRef}
                            onLoad=${(e) => {
                              try {
                                const img = e.currentTarget;
                                workspaceImageRef.current = img;
                                const geom = readImageGeom(img);
                                // #region agent log
                                try {
                                  const maskEl = (img.parentElement || document).querySelector("img.viewer-mask");
                                  const svgEl = (img.parentElement || document).querySelector("svg.viewer-overlay");
                                  const maskRect = maskEl?.getBoundingClientRect?.() || null;
                                  const svgRect = svgEl?.getBoundingClientRect?.() || null;
                                  const cs = window.getComputedStyle(img);
                                  dbg("H_ws_layers_1", "static/app.js:viewerImage:onLoad", "workspace layer rects", {
                                    pageId: currentPageId,
                                    build: APP_BUILD,
                                    imgNaturalW: Number(img.naturalWidth || 0),
                                    imgNaturalH: Number(img.naturalHeight || 0),
                                    imgRectW: geom?.cw || 0,
                                    imgRectH: geom?.ch || 0,
                                    parentW: geom?.pw || 0,
                                    parentH: geom?.ph || 0,
                                    objectFit: cs.objectFit,
                                    objectPosition: cs.objectPosition,
                                    maskRect,
                                    svgRect,
                                  });
                                } catch (_) {}
                                // #endregion
                                if (!geom || !currentPageId) return;
                                setPageGeomById((prev) => upsertGeom(prev, currentPageId, geom));
                                dbg("H_overlay_1", "static/app.js:viewerImage:onLoad", "image loaded", {
                                  pageId: currentPageId,
                                  naturalWidth: geom.w,
                                  naturalHeight: geom.h,
                                  rectW: geom.cw,
                                  rectH: geom.ch,
                                  parentW: geom.pw,
                                  parentH: geom.ph,
                                  regionsCount: (currentAssets?.regions || []).length,
                                });
                              } catch (_) {}
                            }}
                          />
                          ${maskOkByPage[currentPageId] !== false
                            ? html`
                                <img
                                  src=${`${currentAssets.mask_url}?t=${maskNonce}`}
                                  alt="Mask overlay"
                                  className="viewer-mask"
                                  style=${(() => {
                                    const g = pageGeomById[currentPageId] || {};
                                    return g.cw && g.ch ? { width: `${g.cw}px`, height: `${g.ch}px` } : null;
                                  })()}
                                  onLoad=${() => {
                                    setMaskOkByPage((prev) => ({ ...prev, [currentPageId]: true }));
                                    // #region agent log
                                    try {
                                      const m = document.querySelector("img.viewer-mask");
                                      const r = m?.getBoundingClientRect?.() || null;
                                      dbg("H_ws_layers_2", "static/app.js:mask:onLoad", "mask rect", {
                                        pageId: currentPageId,
                                        build: APP_BUILD,
                                        rect: r,
                                      });
                                    } catch (_) {}
                                    // #endregion
                                    dbg("H_overlay_2", "static/app.js:mask:onLoad", "mask loaded", { pageId: currentPageId });
                                  }}
                                  onError=${() => {
                                    setMaskOkByPage((prev) => ({ ...prev, [currentPageId]: false }));
                                    dbg("H5", "static/app.js:mask", "Mask image load failed", {
                                      pageId: currentPageId,
                                      maskUrl: currentAssets.mask_url,
                                      stage,
                                    });
                                  }}
                                />
                              `
                            : null}
                          <svg
                            className="viewer-overlay"
                            style=${(() => {
                              const g = pageGeomById[currentPageId] || {};
                              return g.cw && g.ch ? { width: `${g.cw}px`, height: `${g.ch}px` } : null;
                            })()}
                            viewBox=${(() => {
                              const g = pageGeomById[currentPageId] || {};
                              return g.w && g.h ? `0 0 ${g.w} ${g.h}` : inferViewBox(currentAssets.regions || []);
                            })()}
                            preserveAspectRatio="xMinYMin meet"
                          >
                            ${(currentAssets.regions || []).map((region) => {
                              const conf = Number(region.ocr_confidence ?? region.confidence ?? 0);
                              const style = getPolygonStyle(conf);
                              return html`
                                <polygon
                                  key=${region.region_id}
                                  points=${pointsToAttr(region.polygon)}
                                  className="region-polygon"
                                  fill=${style.fill}
                                  stroke=${style.stroke}
                                  onMouseEnter=${(e) => {
                                    try {
                                      const bb = e.currentTarget.getBBox();
                                      dbg("H_overlay_3", "static/app.js:polygon:hover", "polygon bbox", {
                                        pageId: currentPageId,
                                        regionId: region.region_id,
                                        bbX: bb.x,
                                        bbY: bb.y,
                                        bbW: bb.width,
                                        bbH: bb.height,
                                      });
                                    } catch (_) {}
                                  }}
                                  onClick=${() => {
                                    setSelectedRegion(region);
                                    setRegionTranslation({ statusLabel: "loading", text: "", error: "" });
                                  }}
                                />
                              `;
                            })}
                          </svg>
                        </div>
                      `
                    : html`<p className="text-sm text-sollers-gray">Нет данных страницы.</p>`}
                </article>
              `
            : html`
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Offline report viewer</h2>
                  ${importReport && importProjectId
                    ? html`
                        <p className="text-sm text-sollers-gray mt-1">
                          Страниц: ${importPages.length || 0} | exported at:
                          <span className="mono">${importReport.meta?.exported_at || "-"}</span>
                        </p>
                        ${(() => {
                          const missing = countMissingTranslations(importReport);
                          if (!missing) return null;
                          return html`
                            <div className="mt-3 rounded-xl border border-sollers-grayBorder bg-[#F7F6F5] p-3 text-sm">
                              <p className="text-sollers-gray">Непереведённых блоков: <span className="mono">${missing}</span></p>
                              <button
                                className="mt-2 w-full px-4 py-2 rounded-xl border border-sollers-blue text-sollers-blue transition-transform active:scale-[0.98]"
                                onClick=${async () => {
                                  try {
                                    const resp = await fetch(`/api/import/projects/${importProjectId}/translations/enqueue?lang=ru`, {
                                      method: "POST",
                                      headers: { "Content-Type": "application/json" },
                                      body: JSON.stringify(importReport),
                                    });
                                    if (!resp.ok) {
                                      alert(await resp.text());
                                      return;
                                    }
                                    const payload = await resp.json();
                                    alert(`Задачи поставлены. Draft: ${payload?.queued?.region_draft || 0}, Refine: ${payload?.queued?.region_refine || 0}`);
                                  } catch (e) {
                                    alert(String(e || "enqueue failed"));
                                  }
                                }}
                              >
                                Перевести непереведённое
                              </button>
                            </div>
                          `;
                        })()}
                        <div className="mt-4 space-y-3">
                          <div className="flex items-center justify-between gap-2">
                            <button
                              className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                              disabled=${importPageIndex <= 0}
                              onClick=${() => setImportPageIndex((p) => Math.max(0, p - 1))}
                            >
                              Prev
                            </button>
                            <span className="mono text-sm">${importPages[importPageIndex] || "-"}</span>
                            <button
                              className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                              disabled=${importPageIndex >= importPages.length - 1}
                              onClick=${() => setImportPageIndex((p) => Math.min(importPages.length - 1, p + 1))}
                            >
                              Next
                            </button>
                          </div>
                          ${importPages.length > 0
                            ? (() => {
                                const pageId = importPages[importPageIndex];
                                const assets = importAssetsByPage[pageId] || buildOfflineAssets(importReport, pageId);
                                const imageUrl = `/api/projects/${importProjectId}/pages/${pageId}/image?t=${Date.now()}`;
                                return html`
                                  <div className="viewer-surface">
                                    <img
                                      src=${imageUrl}
                                      alt="Rendered page"
                                      className="viewer-page"
                                      ref=${importImageRef}
                                      onLoad=${(e) => {
                                        try {
                                          const img = e.currentTarget;
                                          importImageRef.current = img;
                                          const geom = readImageGeom(img);
                                          if (!geom) return;
                                          setImportPageGeomById((prev) => upsertGeom(prev, pageId, geom));
                                          dbg("H_overlay_1_import", "static/app.js:importViewerImage:onLoad", "import image loaded", {
                                            pageId,
                                            naturalWidth: geom.w,
                                            naturalHeight: geom.h,
                                            rectW: geom.cw,
                                            rectH: geom.ch,
                                            parentW: geom.pw,
                                            parentH: geom.ph,
                                            regionsCount: (assets.regions || []).length,
                                          });
                                        } catch (_) {}
                                      }}
                                    />
                                    <svg
                                      className="viewer-overlay"
                                      style=${(() => {
                                        const g = importPageGeomById[pageId] || {};
                                        return g.cw && g.ch ? { width: `${g.cw}px`, height: `${g.ch}px` } : null;
                                      })()}
                                      viewBox=${(() => {
                                        const g = importPageGeomById[pageId] || {};
                                        return g.w && g.h ? `0 0 ${g.w} ${g.h}` : inferViewBox(assets.regions || []);
                                      })()}
                                      preserveAspectRatio="xMinYMin meet"
                                    >
                                      ${(assets.regions || []).map((region) => {
                                        const conf = Number(region.ocr_confidence ?? region.confidence ?? 0);
                                        const style = getPolygonStyle(conf);
                                        return html`
                                          <polygon
                                            key=${region.region_id}
                                            points=${pointsToAttr(region.polygon)}
                                            className="region-polygon"
                                            fill=${style.fill}
                                            stroke=${style.stroke}
                                            onMouseEnter=${(e) => {
                                              try {
                                                const bb = e.currentTarget.getBBox();
                                                dbg("H_overlay_3_import", "static/app.js:importPolygon:hover", "import polygon bbox", {
                                                  pageId,
                                                  regionId: region.region_id,
                                                  bbX: bb.x,
                                                  bbY: bb.y,
                                                  bbW: bb.width,
                                                  bbH: bb.height,
                                                });
                                              } catch (_) {}
                                            }}
                                            onClick=${() => {
                                              setSelectedRegion(region);
                                              setRegionTranslation({ statusLabel: "loading", text: "", error: "" });
                                            }}
                                          />
                                        `;
                                      })}
                                    </svg>
                                  </div>
                                `;
                              })()
                            : null}
                        </div>
                      `
                    : html`<p className="text-sm text-sollers-gray mt-2">Импортируйте report.json и PDF, затем нажмите Open viewer.</p>`}
                </article>
              `}
        </section>
      </main>

      ${selectedRegion
        ? html`
            <div className="fixed inset-0 bg-[rgba(41,37,35,0.58)] flex items-center justify-center p-4 z-20">
              <div className="w-full max-w-3xl bg-sollers-white rounded-2xl border border-sollers-grayBorder p-5">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <h3 className="text-xl font-semibold">Region details</h3>
                    <p className="text-sm text-sollers-gray mono">
                      ${selectedRegion.region_id} | conf ${Number(
                        selectedRegion.ocr_confidence ?? selectedRegion.confidence ?? 0
                      ).toFixed(3)}
                    </p>
                  </div>
                  <button
                    className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder"
                    onClick=${() => setSelectedRegion(null)}
                  >
                    Close
                  </button>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div className="space-y-2">
                    <p className="text-sm text-sollers-gray">OCR text</p>
                    <textarea className="w-full min-h-[220px] rounded-xl border border-sollers-grayBorder p-3" value=${selectedRegion.text || ""} readOnly></textarea>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-sollers-gray">RU translation</p>
                    <p className="text-xs mono text-sollers-gray">${regionTranslation.statusLabel}</p>
                    <textarea
                      className="w-full min-h-[220px] rounded-xl border border-sollers-grayBorder p-3"
                      value=${regionTranslation.error || regionTranslation.text || ""}
                      readOnly
                    ></textarea>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap justify-end gap-2">
                  <button
                    className="px-4 py-2 rounded-xl border border-sollers-grayBorder transition-transform active:scale-[0.98]"
                    onClick=${async () => {
                      await navigator.clipboard.writeText(selectedRegion.text || "");
                    }}
                  >
                    Copy OCR
                  </button>
                  <button
                    className="px-4 py-2 rounded-xl border border-sollers-blue text-sollers-blue transition-transform active:scale-[0.98]"
                    onClick=${async () => {
                      await navigator.clipboard.writeText(regionTranslation.text || "");
                    }}
                  >
                    Copy translation
                  </button>
                  <button
                    className="px-4 py-2 rounded-xl border border-sollers-orange text-sollers-orange transition-transform active:scale-[0.98]"
                    onClick=${handleRetryRegion}
                    style=${webEnableRetryOcr ? "" : "display:none"}
                  >
                    Retry OCR (accurate)
                  </button>
                </div>
              </div>
            </div>
          `
        : null}
    </div>
  `;
}

createRoot(document.getElementById("root")).render(html`<${App} />`);
