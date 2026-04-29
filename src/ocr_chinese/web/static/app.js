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
  if (value.includes("mask")) return "маска";
  if (value.includes("ocr")) return "ocr";
  if (value.includes("translate")) return "ocr+перевод";
  if (value.includes("render")) return "рендер";
  if (value.includes("done")) return "готово";
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
  // Use layout sizes (clientWidth/Height) so transforms (scale/rotate) don't affect measurements.
  const parent = img.parentElement || img;
  const w = Number(img.naturalWidth || 0);
  const h = Number(img.naturalHeight || 0);
  const cw = Number(img.clientWidth || 0);
  const ch = Number(img.clientHeight || 0);
  const pw = Number(parent.clientWidth || 0);
  const ph = Number(parent.clientHeight || 0);
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

function filenameFromContentDisposition(h) {
  const raw = String(h || "");
  const m = raw.match(/filename="([^"]+)"/i);
  if (m && m[1]) return m[1];
  const m2 = raw.match(/filename=([^;]+)/i);
  if (m2 && m2[1]) return String(m2[1]).trim().replace(/^"|"$/g, "");
  return "";
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
      status_draft: t.status_draft,
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
      const draft = String(t?.draft_translation || "").trim();
      if (!draft) missing += 1;
    }
  }
  return missing;
}

function useEtaModel() {
  const refs = useRef({
    ocr: {},
    draft: {},
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
  const [assetsRevByPage, setAssetsRevByPage] = useState({}); // pageId -> number; bumps only when image/mask url changes
  const [maskOkByPage, setMaskOkByPage] = useState({});
  const [maskNonce, setMaskNonce] = useState(0);
  const [pageGeomById, setPageGeomById] = useState({});
  const [viewByPage, setViewByPage] = useState({}); // pageId -> {rotSteps, scale, panX, panY}
  const [selectedRegion, setSelectedRegion] = useState(null);
  const [regionTranslation, setRegionTranslation] = useState({
    statusLabel: "pending",
    text: "",
    error: "",
  });
  const [generateInFlight, setGenerateInFlight] = useState(false);
  const [importReport, setImportReport] = useState(null);
  const [importError, setImportError] = useState("");
  const [importProjectId, setImportProjectId] = useState(null);
  const [importPages, setImportPages] = useState([]);
  const [importPageIndex, setImportPageIndex] = useState(0);
  const [importAssetsByPage, setImportAssetsByPage] = useState({});
  const [importPageGeomById, setImportPageGeomById] = useState({});
  const [importViewByPage, setImportViewByPage] = useState({}); // pageId -> {rotSteps, scale, panX, panY}
  const [importAssetsRevByPage, setImportAssetsRevByPage] = useState({}); // pageId -> number (for stable cache buster)
  const [importBundleFile, setImportBundleFile] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [dpi, setDpi] = useState(400);
  const [ocrMode, setOcrMode] = useState("eco");
  const [ocrDevice, setOcrDevice] = useState("cuda");
  const [showStatus, setShowStatus] = useState(false);
  const [importOpenInFlight, setImportOpenInFlight] = useState(false);

  const etaModel = useEtaModel();
  const routeRef = useRef(route);
  routeRef.current = route;
  const assetsEnsureInFlightRef = useRef({}); // key -> Promise; prevents concurrent ensureAssets storms
  const etaJumpRef = useRef({}); // pageId -> {lastAt, lastEta}
  const pageSetRef = useRef({ key: "" });
  const etaDisplayRef = useRef({}); // key -> {eta, at}
  const workspaceImageRef = useRef(null);
  const importImageRef = useRef(null);
  const viewerDragRef = useRef({
    active: false,
    pageId: null,
    isImport: false,
    pointerId: null,
    startClientX: 0,
    startClientY: 0,
    startPanX: 0,
    startPanY: 0,
    moved: false,
    justDraggedAt: 0,
    thresholdPx: 4,
  });

  const VIEW_MIN_SCALE = 0.25;
  const VIEW_MAX_SCALE = 6.0;

  function clampScale(v) {
    if (!Number.isFinite(v)) return 1;
    return Math.max(VIEW_MIN_SCALE, Math.min(VIEW_MAX_SCALE, v));
  }

  function getViewState(map, pageId) {
    const base = map?.[pageId] || null;
    return {
      rotSteps: Number.isFinite(base?.rotSteps) ? ((base.rotSteps % 4) + 4) % 4 : 0,
      scale: clampScale(Number(base?.scale || 1)),
      panX: Number.isFinite(base?.panX) ? base.panX : 0,
      panY: Number.isFinite(base?.panY) ? base.panY : 0,
    };
  }

  function patchViewState(isImport, pageId, patch) {
    if (!pageId) return;
    const setFn = isImport ? setImportViewByPage : setViewByPage;
    setFn((prev) => {
      const cur = getViewState(prev, pageId);
      const next = typeof patch === "function" ? patch(cur) : { ...cur, ...(patch || {}) };
      const normalized = {
        rotSteps: Number.isFinite(next?.rotSteps) ? ((next.rotSteps % 4) + 4) % 4 : cur.rotSteps,
        scale: clampScale(Number(next?.scale || cur.scale)),
        panX: Number.isFinite(next?.panX) ? next.panX : cur.panX,
        panY: Number.isFinite(next?.panY) ? next.panY : cur.panY,
      };
      const same =
        normalized.rotSteps === cur.rotSteps &&
        normalized.scale === cur.scale &&
        normalized.panX === cur.panX &&
        normalized.panY === cur.panY;
      if (same) return prev;
      return { ...prev, [pageId]: normalized };
    });
  }

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
        // Default to GPU when Paddle reports it's usable; otherwise CPU.
        const cudaOk = Boolean(payload?.paddle_cuda_available);
        setOcrMode("eco");
        setOcrDevice(cudaOk ? "cuda" : "cpu");
      } catch (_) {
        // noop
      }
    })();
  }, []);

  const derivedRequestedDevice =
    String(statusPayload?.paddle_runtime?.requested_device || ocrDevice || "cpu").toLowerCase() === "cuda" ? "cuda" : "cpu";
  const derivedCudaAvailable = Boolean(
    statusPayload?.paddle_runtime?.paddle_cuda_available ?? runtimeInfo?.paddle_cuda_available
  );
  const derivedEffectiveDevice = statusPayload?.paddle_runtime?.effective_device
    ? String(statusPayload.paddle_runtime.effective_device)
    : derivedRequestedDevice === "cuda" && derivedCudaAvailable
      ? "cuda"
      : "cpu";

  const webEnableRetryOcr = Boolean(runtimeInfo?.web_enable_retry_ocr);
  const showProgress = Object.keys(progressPages || {}).length > 0;

  const stepMaskDone = stage !== "ожидание" && stage !== "рендер" ? true : stage === "маска" ? false : false;
  const stepMaskActive = stage === "маска";
  const stepOcrActive = stage === "ocr";
  const stepTranslateActive = stage === "перевод";
  const stepOcrOrTranslateActive = stepOcrActive || stepTranslateActive;
  const stepDone = statusState === "done";

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
    const inflightKey = `${pid}:${pageId}:${force ? 1 : 0}`;
    const inflight = assetsEnsureInFlightRef.current[inflightKey];
    if (inflight) {
      return inflight;
    }
    const url = force
      ? `/api/projects/${pid}/pages/${pageId}/assets?force=1`
      : `/api/projects/${pid}/pages/${pageId}/assets`;
    const p = (async () => {
      const resp = await fetch(url);
      if (!resp.ok) return;
      const payload = await resp.json();
      setAssetsByPage((prev) => {
        const cur = prev?.[pageId] || null;
        const next = { ...prev, [pageId]: payload };
        const curImg = String(cur?.image_url || "");
        const curMask = String(cur?.mask_url || "");
        const nextImg = String(payload?.image_url || "");
        const nextMask = String(payload?.mask_url || "");
        if (cur && (curImg !== nextImg || curMask !== nextMask)) {
          setAssetsRevByPage((rprev) => ({ ...rprev, [pageId]: Number(rprev?.[pageId] || 0) + 1 }));
        }
        if (!cur && (nextImg || nextMask)) {
          setAssetsRevByPage((rprev) => ({ ...rprev, [pageId]: Number(rprev?.[pageId] || 0) + 1 }));
        }
        return next;
      });
    })().finally(() => {
      try {
        delete assetsEnsureInFlightRef.current[inflightKey];
      } catch (_) {}
    });
    assetsEnsureInFlightRef.current[inflightKey] = p;
    return p;
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
          translationByPage[pageId]?.regions_total === translationByPage[pageId]?.draft_done + translationByPage[pageId]?.draft_error;
        if (hasFinished) continue;
        try {
          const resp = await fetch(`/api/projects/${projectId}/pages/${pageId}/translations/status?lang=ru`);
          if (!resp.ok) continue;
          const payload = await resp.json();
          entries.push([pageId, payload]);
          etaModel.updateRate("draft", pageId, Number(payload.draft_done || 0));
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
    // Do NOT poll translations for every page (can freeze UI on big PDFs).
    // Keep a warm cache only for the current page (enough for region modal + export on demand).
    if (!projectId || !currentPageId) return undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const resp = await fetch(`/api/projects/${projectId}/pages/${currentPageId}/translations?lang=ru`);
        if (!resp.ok) return;
        const payload = await resp.json();
        if (cancelled) return;
        setPageTranslationsById((prev) => ({ ...prev, [currentPageId]: payload }));
      } catch (_) {}
    };
    tick();
    const timer = window.setInterval(tick, 2500);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [projectId, currentPageId]);

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
      if (entry.draft_translation) {
        setRegionTranslation({
          statusLabel: `draft (${entry.status_draft || "done"})`,
          text: entry.draft_translation,
          error: "",
        });
        return;
      }
      if (entry.error_draft) {
        setRegionTranslation({
          statusLabel: "error",
          text: "",
          error: String(entry.error_draft || "Translation failed"),
        });
        return;
      }
      setRegionTranslation({
        statusLabel: `pending (draft:${entry.status_draft || "pending"})`,
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
        status_draft: selectedRegion.status_draft,
        error_draft: selectedRegion.error_draft,
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

  async function exportBundle() {
    if (!projectId) return;
    try {
      const resp = await fetch(`/api/projects/${projectId}/export/ocpkg`);
      if (!resp.ok) {
        alert(await resp.text());
        return;
      }
      const blob = await resp.blob();
      const cd = resp.headers.get("content-disposition") || "";
      const name = filenameFromContentDisposition(cd) || `${(filename || "document").replace(/\.pdf$/i, "")}_ocr.ocpkg`;
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(String(e || "Export failed"));
    }
  }

  async function handleGenerate() {
    if (!projectId) return;
    setGenerateInFlight(true);
    setStatusText("Запуск…");
    const resp = await fetch(`/api/projects/${projectId}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dpi: Number(dpi || 400),
        ocr_mode: "eco",
        ocr_device: ocrDevice, // backend will auto-fallback to CPU if CUDA is not usable
      }),
    });
    if (!resp.ok) {
      setGenerateInFlight(false);
      setStatusText("Ошибка запуска");
      alert(await resp.text());
      return;
    }
    // Generation runs in background; /status polling reflects real completion.
    await loadPages(projectId);
    setGenerateInFlight(false);
    setStatusText("В процессе");
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

  async function openImportedViewer() {
    if (!importBundleFile) return;
    if (importOpenInFlight) return;
    setImportOpenInFlight(true);
    setImportError("");
    try {
      const fd = new FormData();
      fd.append("file", importBundleFile);
      fd.append("dpi", String(Number(dpi || 400)));
      const resp = await fetch("/api/import/ocpkg", { method: "POST", body: fd });
      if (!resp.ok) {
        const t = await resp.text();
        setImportError(`Не удалось открыть файл: ${t}`);
        setImportOpenInFlight(false);
        return;
      }
      const payload = await resp.json();
      const pid = payload.project_id;
      const pages = payload.pages || [];
      const report = payload.report || null;
      if (!report || !report.pages || !report.regionsByPage) {
        setImportError("Неверный формат отчёта внутри файла.");
        setImportOpenInFlight(false);
        return;
      }
      setImportReport(report);
      setImportProjectId(pid);
      setImportPages(pages);
      setImportPageIndex(0);
      setImportAssetsRevByPage(() => {
        const rev = {};
        for (const pageId of pages || []) rev[String(pageId)] = 1;
        return rev;
      });
      const map = {};
      for (const pageId of pages) map[pageId] = buildOfflineAssets(report, pageId);
      setImportAssetsByPage(map);
    } catch (e) {
      setImportError(`Не удалось открыть файл: ${String(e || "ошибка")}`);
    } finally {
      setImportOpenInFlight(false);
    }
  }

  const rows = useMemo(() => {
    return derivedPageIds.map((pageId) => {
      const o = progressPages[pageId] || {};
      const t = translationByPage[pageId] || {};
      const ocrCur = Number(o.current_region || 0);
      const ocrTotRaw = Number(o.total_regions || 0);
      const ocrTot = ocrTotRaw > 0 ? ocrTotRaw : Number(avgRegionsPerPage || 0);
      const draftDone = Number(t.draft_done || 0);
      const draftError = Number(t.draft_error || 0);
      const draftRunning = Number(t.draft_running || 0);
      const regionsTotalRaw = Number(t.regions_total || 0);
      const regionsTotal = regionsTotalRaw > 0 ? regionsTotalRaw : (ocrTot > 0 ? ocrTot : Number(avgRegionsPerPage || 0));
      const translateHasRealData = regionsTotalRaw > 0 || draftDone > 0 || draftError > 0 || draftRunning > 0;

      const ocrEtaRaw = etaModel.estimateWithFallback("ocr", pageId, ocrCur, ocrTot, null);
      const draftEta = translateHasRealData
        ? etaModel.estimateWithFallback("draft", pageId, draftDone, regionsTotal, null)
        : null;
      const translateEta = draftEta ?? null;
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
        translatePct: translateHasRealData ? toPct(draftDone + draftError, regionsTotal || 0) : 0,
        ocrLabel: `${ocrCur}/${ocrTotRaw || (ocrTot > 0 ? `~${Math.round(ocrTot)}` : "?")}`,
        translateLabel: translateHasRealData
          ? `${draftDone}/${regionsTotalRaw || (regionsTotal > 0 ? `~${Math.round(regionsTotal)}` : "?")}`
          : "",
        ocrEta,
        translateEta: translateEtaSmooth,
        pageEta,
      };
    });
  }, [derivedPageIds, progressPages, translationByPage, avgRegionsPerPage]);

  function PageViewer({
    isImport,
    pageId,
    title,
    imageUrl,
    imageAlt,
    imageRef,
    onImageLoad,
    maskUrl,
    maskOk,
    maskStyle,
    onMaskLoad,
    onMaskError,
    regions,
    geomById,
    setGeomById,
    onRegionClick,
  }) {
    const geom = (pageId && geomById?.[pageId]) || {};
    const view = getViewState(isImport ? importViewByPage : viewByPage, pageId);
    const rotDeg = view.rotSteps * 90;

    const surfaceRef = useRef(null);
    const [surfaceSize, setSurfaceSize] = useState({ w: 0, h: 0 });

    useEffect(() => {
      const el = surfaceRef.current;
      if (!el) return undefined;
      const sync = () => {
        const r = el.getBoundingClientRect();
        const w = Math.max(0, Math.round(Number(r.width || 0)));
        const h = Math.max(0, Math.round(Number(r.height || 0)));
        setSurfaceSize((prev) => (prev.w === w && prev.h === h ? prev : { w, h }));
      };
      sync();
      let ro = null;
      if (typeof window.ResizeObserver === "function") {
        ro = new window.ResizeObserver(() => sync());
        ro.observe(el);
      }
      window.addEventListener("resize", sync);
      return () => {
        window.removeEventListener("resize", sync);
        if (ro) ro.disconnect();
      };
    }, []);

    const naturalW = Number(geom?.w || 0);
    const naturalH = Number(geom?.h || 0);
    const rotStepsNorm = Number.isFinite(view.rotSteps) ? ((view.rotSteps % 4) + 4) % 4 : 0;
    const isRotOdd = rotStepsNorm % 2 === 1;
    const fitW = isRotOdd ? naturalH : naturalW;
    const fitH = isRotOdd ? naturalW : naturalH;

    const lastFitScaleRef = useRef(null);
    const fitScaleRaw = useMemo(() => {
      if (!(fitW > 0 && fitH > 0)) return null;
      const sw = Number(surfaceSize.w || 0);
      const sh = Number(surfaceSize.h || 0);
      if (!(sw > 0 && sh > 0)) return null;
      const v = Math.min(sw / fitW, sh / fitH);
      return Number.isFinite(v) && v > 0 ? v : null;
    }, [surfaceSize.w, surfaceSize.h, fitW, fitH]);

    const fitScale = useMemo(() => {
      const v = Number(fitScaleRaw);
      return Number.isFinite(v) && v > 0 ? v : null;
    }, [fitScaleRaw]);

    // Avoid one-frame "flash" when fitScale briefly becomes invalid during layout/repaint.
    const stableFitScale = useMemo(() => {
      const v = Number(fitScale);
      if (Number.isFinite(v) && v > 0) return v;
      const prev = Number(lastFitScaleRef.current);
      return Number.isFinite(prev) && prev > 0 ? prev : null;
    }, [fitScale]);

    useEffect(() => {
      const v = Number(fitScale);
      if (Number.isFinite(v) && v > 0) lastFitScaleRef.current = v;
    }, [fitScale]);

    const canRenderStable = Number.isFinite(Number(stableFitScale)) && Number(stableFitScale) > 0;
    const hasEverRenderedRef = useRef(false);
    if (canRenderStable) hasEverRenderedRef.current = true;

    const transformStyle = useMemo(() => {
      const tx = Number.isFinite(view.panX) ? view.panX : 0;
      const ty = Number.isFinite(view.panY) ? view.panY : 0;
      const baseFit = canRenderStable ? Number(stableFitScale) : 1;
      const sc = clampScale(view.scale) * baseFit;
      return {
        // translate3d helps avoid intermittent compositor repaint issues on large images.
        transform: `translate3d(${tx}px, ${ty}px, 0) scale(${sc}) rotate(${rotDeg}deg)`,
        transformOrigin: "0 0",
      };
    }, [view.panX, view.panY, view.scale, rotDeg, stableFitScale, canRenderStable]);

    const layerSizeStyle = useMemo(() => {
      return naturalW > 0 && naturalH > 0 ? { width: `${naturalW}px`, height: `${naturalH}px` } : null;
    }, [naturalW, naturalH]);

    const viewBox = useMemo(() => {
      return geom?.w && geom?.h ? `0 0 ${geom.w} ${geom.h}` : inferViewBox(regions || []);
    }, [geom?.w, geom?.h, regions]);

    const rev = Number((isImport ? importAssetsRevByPage : assetsRevByPage)?.[pageId] || 0);
    const stableImageSrc = imageUrl ? `${imageUrl}${String(imageUrl).includes("?") ? "&" : "?"}v=${rev}` : "";
    const stableMaskSrc = maskUrl ? `${maskUrl}${String(maskUrl).includes("?") ? "&" : "?"}v=${rev}` : "";

    const startDrag = (e) => {
      if (!pageId) return;
      if (!e.isPrimary) return;
      if (e.button !== 0) return;
      // Allow clicking interactive overlay elements (regions) without starting a pan drag.
      const t = e.target;
      if (t && typeof t.closest === "function") {
        if (t.closest(".region-polygon")) return;
        if (t.closest("button,a,input,textarea,select")) return;
      }
      try {
        e.currentTarget?.setPointerCapture?.(e.pointerId);
      } catch (_) {}
      viewerDragRef.current = {
        active: true,
        pageId,
        isImport: Boolean(isImport),
        pointerId: e.pointerId,
        startClientX: e.clientX,
        startClientY: e.clientY,
        startPanX: view.panX,
        startPanY: view.panY,
        moved: false,
        justDraggedAt: Number(viewerDragRef.current?.justDraggedAt || 0),
        thresholdPx: viewerDragRef.current?.thresholdPx || 4,
      };
    };

    const moveDrag = (e) => {
      const st = viewerDragRef.current;
      if (
        !st?.active ||
        st.pageId !== pageId ||
        Boolean(st.isImport) !== Boolean(isImport) ||
        st.pointerId !== e.pointerId
      )
        return;
      const dx = e.clientX - st.startClientX;
      const dy = e.clientY - st.startClientY;
      const dist = Math.hypot(dx, dy);
      if (dist > Number(st.thresholdPx || 4)) st.moved = true;
      patchViewState(isImport, pageId, (cur) => ({ ...cur, panX: st.startPanX + dx, panY: st.startPanY + dy }));
      e.preventDefault();
    };

    const endDrag = (e) => {
      const st = viewerDragRef.current;
      if (
        !st?.active ||
        st.pageId !== pageId ||
        Boolean(st.isImport) !== Boolean(isImport) ||
        (e && st.pointerId != null && e.pointerId != null && st.pointerId !== e.pointerId)
      )
        return;
      const wasMoved = Boolean(st.moved);
      viewerDragRef.current = {
        ...st,
        active: false,
        pointerId: null,
        moved: false,
        justDraggedAt: wasMoved ? Date.now() : Number(st.justDraggedAt || 0),
      };
      try {
        surfaceRef.current?.releasePointerCapture?.(st.pointerId);
      } catch (_) {}
      e?.preventDefault?.();
    };

    useEffect(() => {
      const onBlur = () => {
        const st = viewerDragRef.current;
        if (!st?.active) return;
        if (st.pageId !== pageId || Boolean(st.isImport) !== Boolean(isImport)) return;
        try {
          surfaceRef.current?.releasePointerCapture?.(st.pointerId);
        } catch (_) {}
        viewerDragRef.current = { ...st, active: false, pointerId: null, moved: false };
      };
      const onUp = (e) => {
        const st = viewerDragRef.current;
        if (!st?.active) return;
        if (st.pageId !== pageId || Boolean(st.isImport) !== Boolean(isImport)) return;
        if (st.pointerId != null && e?.pointerId != null && st.pointerId !== e.pointerId) return;
        viewerDragRef.current = { ...st, active: false, pointerId: null, moved: false };
        try {
          surfaceRef.current?.releasePointerCapture?.(st.pointerId);
        } catch (_) {}
      };
      window.addEventListener("blur", onBlur);
      window.addEventListener("pointerup", onUp);
      window.addEventListener("pointercancel", onUp);
      return () => {
        window.removeEventListener("blur", onBlur);
        window.removeEventListener("pointerup", onUp);
        window.removeEventListener("pointercancel", onUp);
      };
    }, [pageId, isImport]);

    const bumpScale = (factor, clientPoint = null) => {
      if (!pageId) return;
      const el = surfaceRef.current;
      const rect = el?.getBoundingClientRect?.() || null;
      const cx = rect
        ? (clientPoint && Number.isFinite(clientPoint.x) ? clientPoint.x - rect.left : rect.width / 2)
        : 0;
      const cy = rect
        ? (clientPoint && Number.isFinite(clientPoint.y) ? clientPoint.y - rect.top : rect.height / 2)
        : 0;

      patchViewState(isImport, pageId, (cur) => {
        const prevScale = clampScale(cur.scale);
        const nextScale = clampScale(prevScale * factor);
        try {
          console.log("[viewer]", {
            action: "zoom",
            pageId,
            isImport: !!isImport,
            factor,
            cursor: { x: cx, y: cy },
            prev: { scale: prevScale, panX: cur.panX, panY: cur.panY, rotSteps: cur.rotSteps },
            next: { scale: nextScale },
          });
        } catch (_) {}
        if (nextScale === prevScale) return cur;
        const k = nextScale / prevScale;
        const nextPanX = cx - (cx - cur.panX) * k;
        const nextPanY = cy - (cy - cur.panY) * k;
        return { ...cur, scale: nextScale, panX: nextPanX, panY: nextPanY };
      });
    };

    const onWheel = (e) => {
      if (!pageId) return;
      const factor = e.deltaY < 0 ? 1.08 : 1 / 1.08;
      try {
        console.log("[viewer]", { action: "wheelZoom", pageId, isImport: !!isImport, deltaY: e.deltaY, factor });
      } catch (_) {}
      bumpScale(factor, { x: e.clientX, y: e.clientY });
      e.preventDefault(); // prevent page scroll while zooming inside viewer
    };

    const resetView = () => {
      try {
        console.log("[viewer]", { action: "reset", pageId, isImport: !!isImport });
      } catch (_) {}
      return patchViewState(isImport, pageId, { rotSteps: 0, scale: 1, panX: 0, panY: 0 });
    };

    const rotateKeepingCenter = (dir) => {
      if (!pageId) return;
      try {
        console.log("[viewer]", { action: "rotate", pageId, isImport: !!isImport, dir });
      } catch (_) {}
      const el = surfaceRef.current;
      const rect = el?.getBoundingClientRect?.() || null;
      const cx = rect ? rect.width / 2 : 0;
      const cy = rect ? rect.height / 2 : 0;
      const scaleTotal = clampScale(view.scale) * (Number.isFinite(fitScale) && fitScale > 0 ? fitScale : 1);
      if (!(scaleTotal > 0)) {
        patchViewState(isImport, pageId, (cur) => ({ ...cur, rotSteps: (cur.rotSteps + dir + 4) % 4 }));
        return;
      }

      patchViewState(isImport, pageId, (cur) => {
        const rot0 = ((Number(cur.rotSteps) % 4) + 4) % 4;
        const rot1 = (rot0 + dir + 4) % 4;
        const panX0 = Number.isFinite(cur.panX) ? cur.panX : 0;
        const panY0 = Number.isFinite(cur.panY) ? cur.panY : 0;

        const dx = (cx - panX0) / scaleTotal;
        const dy = (cy - panY0) / scaleTotal;

        // Convert screen-center back to content coords by applying inverse rotation (multiples of 90°).
        let px = dx;
        let py = dy;
        if (rot0 === 1) {
          px = dy;
          py = -dx;
        } else if (rot0 === 2) {
          px = -dx;
          py = -dy;
        } else if (rot0 === 3) {
          px = -dy;
          py = dx;
        }

        // Apply new rotation to that same content point, then choose pan so it stays at screen center.
        let rx = px;
        let ry = py;
        if (rot1 === 1) {
          rx = -py;
          ry = px;
        } else if (rot1 === 2) {
          rx = -px;
          ry = -py;
        } else if (rot1 === 3) {
          rx = py;
          ry = -px;
        }

        const nextPanX = cx - rx * scaleTotal;
        const nextPanY = cy - ry * scaleTotal;

        return { ...cur, rotSteps: rot1, panX: nextPanX, panY: nextPanY };
      });
    };

    const rotateLeft = () => rotateKeepingCenter(-1);
    const rotateRight = () => rotateKeepingCenter(1);

    const isDragging = Boolean(viewerDragRef.current?.active && viewerDragRef.current?.pageId === pageId);

    return html`
      <div>
        <div className="viewer-toolbar">
          <div className="flex items-center gap-2">
            <span className="text-sm text-sollers-gray">${title || "Viewer"}</span>
            <span className="mono text-xs text-sollers-gray">${Math.round(clampScale(Number(view.scale || 1)) * 100)}% | ${rotDeg}°</span>
          </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder" onClick=${() => bumpScale(1 / 1.15)}>
              −
            </button>
            <button className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder" onClick=${() => bumpScale(1.15)}>
              +
            </button>
            <button className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder" onClick=${rotateLeft}>
              ⟲ 90°
            </button>
            <button className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder" onClick=${rotateRight}>
              ⟳ 90°
            </button>
            <button className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder" onClick=${resetView}>
              Сброс
            </button>
          </div>
        </div>

        <div
          className=${`viewer-surface ${isDragging ? "is-dragging" : ""}`}
          ref=${surfaceRef}
          onPointerDown=${startDrag}
          onPointerMove=${moveDrag}
          onPointerUp=${endDrag}
          onPointerCancel=${endDrag}
          onWheel=${onWheel}
        >
          ${!hasEverRenderedRef.current && !canRenderStable
            ? html`<div className="viewer-placeholder">Загрузка…</div>`
            : null}
          <div
            className="viewer-transformLayer"
            style=${{
              ...(layerSizeStyle || {}),
              // Keep the layer in DOM to avoid white flashes during first layout.
              ...(!hasEverRenderedRef.current && !canRenderStable ? { opacity: 0 } : null),
              ...transformStyle,
            }}
          >
            <img
              src=${stableImageSrc}
              alt=${imageAlt || "Rendered page"}
              className="viewer-page"
              style=${layerSizeStyle}
              ref=${imageRef}
              onLoad=${(e) => {
                try {
                  onImageLoad?.(e);
                } finally {
                  try {
                    const img = e.currentTarget;
                    const geom = readImageGeom(img);
                    if (pageId && geom) setGeomById((prev) => upsertGeom(prev, pageId, geom));
                  } catch (_) {}
                }
              }}
              draggable="false"
            />

            ${maskUrl && maskOk !== false
              ? html`
                  <img
                    src=${stableMaskSrc}
                    alt="Mask (preload)"
                    className="viewer-maskPreload"
                    style=${layerSizeStyle}
                    onLoad=${onMaskLoad}
                    onError=${onMaskError}
                    draggable="false"
                  />
                `
              : null}

            <svg
              className="viewer-overlay"
              style=${layerSizeStyle}
              viewBox=${viewBox}
              preserveAspectRatio="none"
            >
              ${(regions || []).map((region) => {
                const conf = Number(region.ocr_confidence ?? region.confidence ?? 0);
                const style = getPolygonStyle(conf);
                const isSelected = String(selectedRegion?.region_id || "") === String(region?.region_id || "");
                return html`
                  <polygon
                    key=${region.region_id}
                    points=${pointsToAttr(region.polygon)}
                    className=${`region-polygon ${isSelected ? "is-selected" : ""}`}
                    fill=${style.fill}
                    stroke=${style.stroke}
                    onClick=${() => {
                      const st = viewerDragRef.current;
                      if (st?.active) return;
                      const justDraggedAt = Number(st?.justDraggedAt || 0);
                      if (justDraggedAt > 0 && Date.now() - justDraggedAt < 240) return;
                      onRegionClick?.(region);
                    }}
                  />
                `;
              })}
            </svg>
          </div>
        </div>
      </div>
    `;
  }

  const isBusy = generateInFlight || statusState === "running";

  return html`
    <div className="min-h-[100dvh] bg-sollers-white">
      <header className="border-b border-sollers-grayBorder bg-sollers-white">
        <div className="max-w-[1400px] mx-auto px-4 md:px-8 py-4 flex items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl md:text-4xl tracking-tight text-sollers-graphite font-semibold">
              OCR + перевод (PDF)
            </h1>
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <span className=${`step ${stepMaskActive ? "is-active" : stepMaskDone ? "is-done" : ""}`}>Маска</span>
              <span className="step-sep">→</span>
              <span className=${`step ${stepOcrOrTranslateActive ? "is-active" : stepDone ? "is-done" : ""}`}>OCR+перевод</span>
            </div>
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
              Обработка PDF
            </button>
            <button
              className=${`px-4 py-2 rounded-xl border text-sm transition-transform active:scale-[0.98] ${
                route === "import"
                  ? "bg-sollers-orange text-sollers-white border-sollers-orange"
                  : "bg-sollers-white text-sollers-graphite border-sollers-grayBorder"
              }`}
              onClick=${() => goTo("import")}
            >
              Открыть результат
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-[1400px] mx-auto px-4 md:px-8 py-6 grid grid-cols-1 lg:grid-cols-[360px_minmax(0,1fr)] gap-6">
        ${route === "workspace"
          ? html`
              <section className="space-y-5 min-w-0">
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Файл</h2>
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
                    <div className="text-sm text-sollers-gray">
                      Режим: <span className="mono">eco</span> · Устройство: 
                      <span className="mono">${derivedEffectiveDevice === "cuda" ? "GPU" : "CPU"}</span>
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                      <button
                        onClick=${handleUpload}
                        disabled=${!uploadFile}
                        className="px-4 py-2 rounded-xl border border-sollers-orange bg-sollers-orange text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                      >
                        Загрузить PDF
                      </button>
                      <button
                        onClick=${handleGenerate}
                        disabled=${!projectId || isBusy}
                        className="px-4 py-2 rounded-xl border border-sollers-graphite bg-sollers-graphite text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                      >
                        Запустить OCR
                      </button>
                    </div>
                    <button
                      onClick=${exportBundle}
                      disabled=${!projectId}
                      className="w-full px-4 py-2 rounded-xl border border-sollers-blue text-sollers-blue disabled:opacity-50 transition-transform active:scale-[0.98]"
                    >
                      Скачать результат (.ocpkg)
                    </button>
                  </div>
                </article>

                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <div className="flex items-center justify-between gap-3">
                    <h2 className="text-lg font-semibold">Статус</h2>
                    <button
                      className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder text-sm"
                      onClick=${() => setShowStatus((v) => !v)}
                    >
                      ${showStatus ? "Скрыть" : "Показать"}
                    </button>
                  </div>
                  ${showStatus
                    ? html`
                        <div className="mt-3 space-y-2 text-sm">
                          <p><span className="text-sollers-gray">Состояние:</span> <span className="font-medium">${statusText}</span></p>
                          <p><span className="text-sollers-gray">Этап:</span> <span className="font-medium">${stage}</span></p>
                          <p><span className="text-sollers-gray">Файл:</span> <span className="mono">${filename || "-"}</span></p>
                          <p>
                            <span className="text-sollers-gray">API:</span>
                            <span className=${`ml-2 font-medium ${connectionOk ? "text-sollers-green" : "text-sollers-red"}`}>
                              ${connectionOk ? "online" : "offline"}
                            </span>
                          </p>
                        </div>
                      `
                    : html`<p className="mt-2 text-sm text-sollers-gray">Скрыто (по умолчанию).</p>`}
                </article>
              </section>
            `
          : html`
              <section className="space-y-5 min-w-0">
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5">
                  <h2 className="text-lg font-semibold">Открыть готовый результат</h2>
                  <p className="text-sm text-sollers-gray mt-2">
                    Один файл содержит PDF и результат OCR+перевода.
                  </p>
                  <label className="block space-y-2 mt-4">
                    <span className="text-sm text-sollers-gray">Файл результата (.ocpkg)</span>
                    <input
                      type="file"
                      accept=".ocpkg,application/zip"
                      className="w-full rounded-xl border border-sollers-grayBorder p-2"
                      onChange=${(e) => setImportBundleFile(e.target.files?.[0] || null)}
                    />
                  </label>
                  <button
                    className="mt-4 w-full px-4 py-2 rounded-xl border border-sollers-orange bg-sollers-orange text-sollers-white disabled:opacity-50 transition-transform active:scale-[0.98]"
                    disabled=${!importBundleFile || importOpenInFlight}
                    onClick=${openImportedViewer}
                  >
                    ${importOpenInFlight ? "Загрузка…" : "Открыть"}
                  </button>
                  ${importError
                    ? html`<p className="mt-3 text-sm text-sollers-red">${importError}</p>`
                    : null}
                </article>
              </section>
            `}

        <section className="space-y-5 min-w-0">
          ${route === "workspace"
            ? html`
                ${showProgress
                  ? html`
                      <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5 min-w-0">
                        <div className="flex items-center justify-between mb-4">
                          <h2 className="text-lg font-semibold">Прогресс по страницам</h2>
                          <span className="text-sm text-sollers-gray">ETA приблизительная</span>
                        </div>
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
                                    ETA: ${formatDuration(row.pageEta)} | OCR: ${formatDuration(row.ocrEta)}
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
                                  ${row.translateLabel
                                    ? html`
                                        <div>
                                          <div className="flex justify-between text-xs text-sollers-gray">
                                            <span>Перевод</span><span>${row.translateLabel}</span>
                                          </div>
                                          <div className="h-2 bg-sollers-graySoft rounded mt-1">
                                            <div className="h-2 bg-sollers-blue rounded" style=${{ width: `${row.translatePct}%` }}></div>
                                          </div>
                                        </div>
                                      `
                                    : null}
                                </div>
                              </button>
                            `
                          )}
                        </div>
                      </article>
                    `
                  : null}

                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5 min-w-0">
                  <div className="flex items-center justify-between mb-3">
                    <h2 className="text-lg font-semibold">Просмотр</h2>
                    <div className="flex items-center gap-2">
                      <button
                        className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                        disabled=${pageIndex <= 0}
                        onClick=${() => setPageIndex((prev) => Math.max(0, prev - 1))}
                      >
                        Назад
                      </button>
                      <span className="mono text-sm">${currentPageId || "-"}</span>
                      <button
                        className="px-3 py-1.5 rounded-lg border border-sollers-grayBorder disabled:opacity-50"
                        disabled=${pageIndex >= pages.length - 1}
                        onClick=${() => setPageIndex((prev) => Math.min(pages.length - 1, prev + 1))}
                      >
                        Вперёд
                      </button>
                    </div>
                  </div>

                  ${currentAssets
                    ? html`
                        <${PageViewer}
                          isImport=${false}
                          pageId=${currentPageId}
                          title=${"Viewer"}
                          imageUrl=${currentAssets.image_url}
                          imageAlt=${"Rendered page"}
                          imageRef=${workspaceImageRef}
                          onImageLoad=${(e) => {
                            try {
                              const img = e.currentTarget;
                              workspaceImageRef.current = img;
                              const geom = readImageGeom(img);
                              if (!geom || !currentPageId) return;
                              setPageGeomById((prev) => upsertGeom(prev, currentPageId, geom));
                            } catch (_) {}
                          }}
                          maskUrl=${`${currentAssets.mask_url}?t=${maskNonce}`}
                          maskOk=${maskOkByPage[currentPageId]}
                          maskStyle=${(() => {
                            const g = pageGeomById[currentPageId] || {};
                            return g.cw && g.ch ? { width: `${g.cw}px`, height: `${g.ch}px` } : null;
                          })()}
                          onMaskLoad=${() => {
                            setMaskOkByPage((prev) => {
                              if (prev?.[currentPageId] === true) return prev;
                              return { ...prev, [currentPageId]: true };
                            });
                          }}
                          onMaskError=${() => {
                            setMaskOkByPage((prev) => ({ ...prev, [currentPageId]: false }));
                          }}
                          regions=${currentAssets.regions || []}
                          geomById=${pageGeomById}
                          setGeomById=${setPageGeomById}
                          onRegionClick=${(region) => {
                            setSelectedRegion(region);
                            setRegionTranslation({ statusLabel: "loading", text: "", error: "" });
                          }}
                        />
                      `
                    : html`<p className="text-sm text-sollers-gray">Нет данных страницы.</p>`}
                </article>
              `
            : html`
                <article className="rounded-2xl border border-sollers-grayBorder bg-sollers-white p-5 min-w-0">
                  <h2 className="text-lg font-semibold">Offline report viewer</h2>
                  ${importReport && importProjectId
                    ? html`
                        <p className="text-sm text-sollers-gray mt-1">
                          Страниц: ${importPages.length || 0} | exported at:
                          <span className="mono">${importReport.meta?.exported_at || "-"}</span>
                        </p>
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
                                const imageUrl = `/api/projects/${importProjectId}/pages/${pageId}/image`;
                                return html`
                                  <${PageViewer}
                                    isImport=${true}
                                    pageId=${pageId}
                                    title=${"Offline viewer"}
                                    imageUrl=${imageUrl}
                                    imageAlt=${"Rendered page"}
                                    imageRef=${importImageRef}
                                    onImageLoad=${(e) => {
                                      try {
                                        const img = e.currentTarget;
                                        importImageRef.current = img;
                                        const geom = readImageGeom(img);
                                        if (!geom) return;
                                        setImportPageGeomById((prev) => upsertGeom(prev, pageId, geom));
                                      } catch (_) {}
                                    }}
                                    maskUrl=${null}
                                    maskOk=${true}
                                    regions=${assets.regions || []}
                                    geomById=${importPageGeomById}
                                    setGeomById=${setImportPageGeomById}
                                    onRegionClick=${(region) => {
                                      setSelectedRegion(region);
                                      setRegionTranslation({ statusLabel: "loading", text: "", error: "" });
                                    }}
                                  />
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
                    style=${webEnableRetryOcr ? null : { display: "none" }}
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
