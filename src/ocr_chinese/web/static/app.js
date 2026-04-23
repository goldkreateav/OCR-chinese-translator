let projectId = null;
let pages = [];
let currentPageIndex = 0;
let currentRegion = null;
let currentRegionPolygonEl = null;

const pdfInput = document.getElementById("pdfInput");
const dpiInput = document.getElementById("dpiInput");
const uploadBtn = document.getElementById("uploadBtn");
const generateBtn = document.getElementById("generateBtn");
const statusText = document.getElementById("statusText");
const projectInfo = document.getElementById("projectInfo");
const versionText = document.getElementById("versionText");
const viewerWrap = document.getElementById("viewerWrap");
const pageImage = document.getElementById("pageImage");
const maskImage = document.getElementById("maskImage");
const regionOverlay = document.getElementById("regionOverlay");
const pageText = document.getElementById("pageText");
const prevPageBtn = document.getElementById("prevPageBtn");
const nextPageBtn = document.getElementById("nextPageBtn");

const modalBackdrop = document.getElementById("modalBackdrop");
const regionMeta = document.getElementById("regionMeta");
const regionText = document.getElementById("regionText");
const copyBtn = document.getElementById("copyBtn");
const closeModalBtn = document.getElementById("closeModalBtn");
const retryOcrBtn = document.getElementById("retryOcrBtn");

function setStatus(text) {
  statusText.textContent = text;
}

async function loadVersion() {
  try {
    const resp = await fetch("/api/version");
    if (!resp.ok) return;
    const payload = await resp.json();
    const appVer = payload.app_version || "unknown";
    const bridge = payload.quality_bridge_enabled ? "bridge:on" : "bridge:off";
    versionText.textContent = `${appVer} (${bridge})`;
  } catch (_) {
    // ignore
  }
}

async function pollStatusUntilDone() {
  if (!projectId) return;
  for (;;) {
    const resp = await fetch(`/api/projects/${projectId}/status`);
    if (resp.ok) {
      const s = await resp.json();
      if (s.status === "running") {
        setStatus(`Running (${s.stage || "ocr"})`);
      } else {
        if (s.status === "done") setStatus("Done");
        if (s.status === "error") setStatus("Error");
        return s;
      }
    }
    await new Promise((r) => setTimeout(r, 900));
  }
}

function progressFromStatus(s) {
  return { stage: s.stage || "ocr" };
}

async function uploadPdf() {
  const file = pdfInput.files?.[0];
  if (!file) {
    alert("Select a PDF file first.");
    return;
  }
  const formData = new FormData();
  formData.append("file", file);

  setStatus("Uploading PDF...");
  const response = await fetch("/api/projects", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    setStatus("Upload failed");
    alert(await response.text());
    return;
  }
  const payload = await response.json();
  projectId = payload.project_id;
  projectInfo.textContent = ` Project: ${projectId}`;
  generateBtn.disabled = false;
  setStatus("Uploaded");
}

async function generateMaskAndOcr() {
  if (!projectId) {
    alert("Upload PDF first.");
    return;
  }
  setStatus("Generating mask and OCR...");
  generateBtn.disabled = true;

  // Start polling immediately while the long-running request is in-flight.
  let pollDone = false;
  const pollPromise = (async () => {
    try {
      while (!pollDone) {
        const resp = await fetch(`/api/projects/${projectId}/status`);
        if (resp.ok) {
          const s = await resp.json();
          if (s.status === "running") {
            setStatus(`Running (${s.stage || "ocr"})`);
          } else if (s.status === "done") {
            setStatus("Done");
            return;
          } else if (s.status === "error") {
            setStatus("Error");
            return;
          }
        }
        await new Promise((r) => setTimeout(r, 900));
      }
    } catch (_) {
      // ignore
    }
  })();

  const response = await fetch(`/api/projects/${projectId}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dpi: Number(dpiInput.value || 400) }),
  });

  if (!response.ok) {
    setStatus("Generation failed");
    alert(await response.text());
    generateBtn.disabled = false;
    pollDone = true;
    return;
  }

  // Request finished; stop the background polling loop and await last update.
  pollDone = true;
  await pollPromise;

  await loadPages();
  if (pages.length > 0) {
    currentPageIndex = 0;
    await showPage(pages[currentPageIndex]);
  }
  updatePageButtons();
  setStatus("Done");
  generateBtn.disabled = false;
}

async function loadPages() {
  const response = await fetch(`/api/projects/${projectId}/pages`);
  const payload = await response.json();
  pages = payload.pages || [];
}

function polygonToPoints(points) {
  return points.map((xy) => `${xy[0]},${xy[1]}`).join(" ");
}

function clearOverlay() {
  while (regionOverlay.firstChild) {
    regionOverlay.removeChild(regionOverlay.firstChild);
  }
}

function openModal(region) {
  currentRegion = region;
  const conf = Number(region.ocr_confidence ?? region.confidence ?? 0);
  const score = Number(region.ocr_score ?? conf);
  const variant = region.ocr_variant || "n/a";
  regionMeta.textContent = `ID: ${region.region_id} | confidence: ${conf.toFixed(3)} | score: ${score.toFixed(3)} | variant: ${variant}`;
  regionText.value = region.text || "";
  modalBackdrop.classList.remove("hidden");
}

function confidenceStyle(conf) {
  if (!Number.isFinite(conf) || conf < 0) conf = 0;
  if (conf === 0) {
    return { fill: "rgba(0,0,0,0.18)", stroke: "#000000" }; // black
  }
  if (conf < 0.4) {
    return { fill: "rgba(239,68,68,0.18)", stroke: "#dc2626" }; // red
  }
  if (conf < 0.7) {
    return { fill: "rgba(249,115,22,0.18)", stroke: "#ea580c" }; // orange
  }
  return { fill: "rgba(34,197,94,0.20)", stroke: "#16a34a" }; // green
}

function brighterFill(fillColor, alphaBoost) {
  const match = /^rgba\((\d+),\s*(\d+),\s*(\d+),\s*([0-9.]+)\)$/i.exec(fillColor || "");
  if (!match) return fillColor;
  const r = Number(match[1]);
  const g = Number(match[2]);
  const b = Number(match[3]);
  const a = Math.min(1, Number(match[4]) + alphaBoost);
  return `rgba(${r},${g},${b},${a.toFixed(3)})`;
}

function renderRegions(regions) {
  clearOverlay();
  for (const region of regions) {
    const conf = Number(region.ocr_confidence ?? region.confidence ?? 0);
    const style = confidenceStyle(conf);
    const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
    polygon.setAttribute("points", polygonToPoints(region.polygon));
    polygon.setAttribute("class", "region-polygon");
    polygon.setAttribute("fill", style.fill);
    polygon.setAttribute("stroke", style.stroke);
    polygon.dataset.baseFill = style.fill;
    polygon.dataset.baseStrokeWidth = "1";
    polygon.addEventListener("mouseenter", () => {
      polygon.setAttribute("fill", brighterFill(style.fill, 0.16));
      polygon.setAttribute("stroke-width", "2.5");
    });
    polygon.addEventListener("mouseleave", () => {
      polygon.setAttribute("fill", polygon.dataset.baseFill || style.fill);
      polygon.setAttribute("stroke-width", polygon.dataset.baseStrokeWidth || "1");
    });
    polygon.addEventListener("click", () => {
      currentRegionPolygonEl = polygon;
      openModal(region);
    });
    regionOverlay.appendChild(polygon);
  }
}

async function showPage(pageId) {
  const response = await fetch(`/api/projects/${projectId}/pages/${pageId}/assets`);
  if (!response.ok) {
    setStatus("Failed to load page assets");
    return;
  }
  const assets = await response.json();

  viewerWrap.classList.remove("hidden");
  pageText.textContent = `Page: ${assets.page_id}`;
  pageImage.src = `${assets.image_url}?t=${Date.now()}`;
  maskImage.src = `${assets.mask_url}?t=${Date.now()}`;

  pageImage.onload = () => {
    regionOverlay.setAttribute("viewBox", `0 0 ${pageImage.naturalWidth} ${pageImage.naturalHeight}`);
    renderRegions(assets.regions || []);
  };
}

function updatePageButtons() {
  prevPageBtn.disabled = currentPageIndex <= 0;
  nextPageBtn.disabled = currentPageIndex >= pages.length - 1;
}

uploadBtn.addEventListener("click", () => {
  uploadPdf().catch((error) => {
    setStatus("Upload error");
    alert(String(error));
  });
});

generateBtn.addEventListener("click", () => {
  generateMaskAndOcr().catch((error) => {
    setStatus("Generation error");
    alert(String(error));
  });
});

prevPageBtn.addEventListener("click", async () => {
  if (currentPageIndex > 0) {
    currentPageIndex -= 1;
    await showPage(pages[currentPageIndex]);
    updatePageButtons();
  }
});

nextPageBtn.addEventListener("click", async () => {
  if (currentPageIndex < pages.length - 1) {
    currentPageIndex += 1;
    await showPage(pages[currentPageIndex]);
    updatePageButtons();
  }
});

closeModalBtn.addEventListener("click", () => {
  modalBackdrop.classList.add("hidden");
  currentRegion = null;
  currentRegionPolygonEl = null;
});

copyBtn.addEventListener("click", async () => {
  await navigator.clipboard.writeText(regionText.value || "");
});

retryOcrBtn.addEventListener("click", async () => {
  if (!projectId || !currentRegion?.region_id) return;
  retryOcrBtn.disabled = true;
  const prev = retryOcrBtn.textContent;
  retryOcrBtn.textContent = "Retrying...";
  try {
    const response = await fetch(`/api/projects/${projectId}/regions/${currentRegion.region_id}/retry`, {
      method: "POST",
    });
    if (!response.ok) {
      alert(await response.text());
      return;
    }
    const payload = await response.json();
    currentRegion.text = payload.text;
    currentRegion.ocr_confidence = payload.ocr_confidence;
    currentRegion.ocr_variant = payload.ocr_variant;
    currentRegion.ocr_score = payload.ocr_score;
    openModal(currentRegion);
    if (currentRegionPolygonEl) {
      const style = confidenceStyle(Number(payload.ocr_confidence ?? 0));
      currentRegionPolygonEl.setAttribute("fill", style.fill);
      currentRegionPolygonEl.setAttribute("stroke", style.stroke);
      currentRegionPolygonEl.dataset.baseFill = style.fill;
    }
  } catch (e) {
    alert(String(e));
  } finally {
    retryOcrBtn.textContent = prev;
    retryOcrBtn.disabled = false;
  }
});

// init
loadVersion().catch(() => {});
