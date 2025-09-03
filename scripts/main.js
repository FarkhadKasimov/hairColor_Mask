// Hair Color Demo — iOS-stable offscreen masking + dual loop
import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

/* ===== Detect iOS ===== */
const IS_IOS = /iPad|iPhone|iPod/.test(navigator.userAgent);

/* ===== Performance / Quality ===== */
const SEG_FPS   = 15;       // частота вычисления маски (12–20 хорошо)
const SEG_SCALE = 0.5;      // масштаб входа в сегментацию (0.4–0.6)
const USE_CONF  = !IS_IOS;  // на iOS надёжнее categoryMask

// Тюнинг порогов (на iOS чуть ниже)
const ACC = {
  thresh: IS_IOS ? 0.45 : 0.60, // порог уверенности для confidenceMask
  gain:   IS_IOS ? 1.15 : 1.10, // лёгкое усиление
  emaAlpha: 0.20,               // сглаживание маски (0..1)
  // blur оставлен выключенным ради FPS и стабильности
};

const els = {
  start: document.getElementById("startBtn"),
  stop: document.getElementById("stopBtn"),
  snap: document.getElementById("snapBtn"),
  save: document.getElementById("saveBtn"),
  video: document.getElementById("video"),
  canvas: document.getElementById("canvas"),
  status: document.getElementById("status"),
  mode: document.getElementById("mode"),
  strength: document.getElementById("strength"),
  feather: document.getElementById("feather"),
  solidColor: document.getElementById("solidColor"),
  presets: document.getElementById("presets"),
  topColor: document.getElementById("topColor"),
  bottomColor: document.getElementById("bottomColor"),
  gradientShift: document.getElementById("gradientShift"),
  solidPanel: document.getElementById("solidPanel"),
  ombrePanel: document.getElementById("ombrePanel"),
  hue: document.getElementById("hue"),
  sat: document.getElementById("sat"),
  light: document.getElementById("light"),
  resetHSL: document.getElementById("resetHSL"),
};

const outCtx = els.canvas.getContext("2d", { alpha: true });
let segmenter = null;
let running = false;

/* ===== Downscaled input for segmentation ===== */
const computeCanvas = document.createElement("canvas");
const computeCtx = computeCanvas.getContext("2d", { willReadFrequently: true });

/* ===== Mask canvas (RGBA, compute-res) ===== */
const maskCanvas = document.createElement("canvas");
const maskCtx = maskCanvas.getContext("2d");

/* ===== Colored layer canvas (full-res), masked OFFSCREEN ===== */
const colorCanvas = document.createElement("canvas");
const colorCtx = colorCanvas.getContext("2d");

/* ===== EMA buffer ===== */
let prevAlpha = null;

/* ===== timers ===== */
let drawRaf = 0;
let segTimer = 0;

/* ===== helpers ===== */
const setStatus = (t) => { if (els.status) els.status.textContent = t ?? ""; };
const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const minmax = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

/* ================= INIT ================= */
async function setupCamera() {
  // важно для iOS
  els.video.setAttribute("playsinline", "");
  els.video.playsInline = true;
  els.video.muted = true;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "user", width: { ideal: 960 }, height: { ideal: 720 } },
    audio: false,
  });
  els.video.srcObject = stream;

  await els.video.play();
  await new Promise((r) => (els.video.readyState >= 2 ? r() : (els.video.onloadedmetadata = () => r())));

  const vw = els.video.videoWidth || 640;
  const vh = els.video.videoHeight || 480;

  els.canvas.width = vw;
  els.canvas.height = vh;

  computeCanvas.width  = Math.max(64, Math.round(vw * SEG_SCALE));
  computeCanvas.height = Math.max(64, Math.round(vh * SEG_SCALE));

  maskCanvas.width  = computeCanvas.width;
  maskCanvas.height = computeCanvas.height;

  colorCanvas.width  = vw;
  colorCanvas.height = vh;

  // дефолтная интенсивность
  if (els.strength) els.strength.value = "0.3";

  setStatus(`Камера: ${vw}×${vh}, seg: ${computeCanvas.width}×${computeCanvas.height}`);
  outCtx.drawImage(els.video, 0, 0, vw, vh);
}

async function initSegmenter() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite",
    },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: USE_CONF,
  });
  setStatus("Модель загружена");
}

/* ================= MASK UTILS ================= */
function alphaFromConfidenceMask(confMask) {
  const w = confMask.width ?? confMask.cols ?? confMask.shape?.[1];
  const h = confMask.height ?? confMask.rows ?? confMask.shape?.[0];
  const f32 = typeof confMask.getAsFloat32Array === "function" ? confMask.getAsFloat32Array() : null;
  if (!f32) throw new Error("No Float32 confidence mask");

  const out = new Float32Array(f32.length);
  const t = ACC.thresh, g = ACC.gain;
  for (let i = 0; i < f32.length; i++) {
    const v = clamp01(f32[i] * g);
    out[i] = v >= t ? v : 0;
  }
  return { alpha: out, w, h };
}

function emaAlphaBlend(curr, prev, a) {
  if (!prev || prev.length !== curr.length) return curr.slice();
  const out = new Float32Array(curr.length);
  const b = 1 - a;
  for (let i = 0; i < curr.length; i++) out[i] = a * curr[i] + b * prev[i];
  return out;
}

function alphaToMaskCanvas(alpha, w, h) {
  maskCanvas.width = w; maskCanvas.height = h;
  const id = maskCtx.createImageData(w, h);
  const data = id.data;
  for (let i = 0; i < alpha.length; i++) {
    const a = minmax(Math.round(alpha[i] * 255), 0, 255);
    const off = i * 4;
    data[off + 0] = 255;
    data[off + 1] = 255;
    data[off + 2] = 255;
    data[off + 3] = a;
  }
  maskCtx.putImageData(id, 0, 0);
  return maskCanvas;
}

/* ================= COLOR LAYER (OFFSCREEN) =================
   Собираем цвет в full-res и применяем маску в offscreen канвасе.
   На основной canvas кладём готовую картинку обычным source-over.
*/
function buildColorLayer(maskCanv) {
  const w = colorCanvas.width, h = colorCanvas.height;

  // 1) Сформировать цветовую заливку
  colorCtx.save();
  colorCtx.globalCompositeOperation = "copy"; // очистить и начать заново
  const mode = els.mode.value;
  if (mode === "solid") {
    colorCtx.fillStyle = els.solidColor.value || "#9b4dff";
    colorCtx.fillRect(0, 0, w, h);
  } else {
    const top = els.topColor.value || "#f5d08b";
    const bottom = els.bottomColor.value || "#8e44ad";
    const shift = parseFloat(els.gradientShift.value || "0");
    const grad = colorCtx.createLinearGradient(0, h * (0.2 + shift), 0, h * (0.8 + shift));
    grad.addColorStop(0, top);
    grad.addColorStop(1, bottom);
    colorCtx.fillStyle = grad;
    colorCtx.fillRect(0, 0, w, h);
  }

  // 2) Применить маску внутри offscreen
  colorCtx.globalCompositeOperation = "destination-in";
  colorCtx.imageSmoothingEnabled = true;
  colorCtx.imageSmoothingQuality = "high";
  colorCtx.drawImage(maskCanv, 0, 0, w, h);
  colorCtx.restore();
}

/* ================= HSL (на результате) ================= */
function applyHSLAdjustments(baseCanvas) {
  const w = els.canvas.width, h = els.canvas.height;
  const tmp = document.createElement("canvas");
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext("2d");

  tctx.drawImage(baseCanvas, 0, 0);

  const light = parseInt(els.light?.value || "0", 10);
  if (light !== 0) {
    tctx.globalAlpha = minmax(Math.abs(light) / 100, 0, 1);
    tctx.globalCompositeOperation = light > 0 ? "screen" : "multiply";
    tctx.fillStyle = light > 0 ? "#ffffff" : "#000000";
    tctx.fillRect(0, 0, w, h);
    tctx.globalAlpha = 1;
  }

  const sat = parseInt(els.sat?.value || "0", 10);
  if (sat !== 0) {
    const img = tctx.getImageData(0, 0, w, h);
    const data = img.data;
    if (sat < 0) {
      const k = (100 - Math.min(100, Math.abs(sat))) / 100;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        data[i] = gray + (r - gray) * k;
        data[i + 1] = gray + (g - gray) * k;
        data[i + 2] = gray + (b - gray) * k;
      }
    } else {
      const boost = 1 + sat / 100;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        data[i] = gray + (r - gray) * boost;
        data[i + 1] = gray + (g - gray) * boost;
        data[i + 2] = gray + (b - gray) * boost;
      }
    }
    tctx.putImageData(img, 0, 0);
  }

  return tmp;
}

/* ================= LOOPS ================= */
// Рисуем каждый кадр: видео + готовый цветовой слой
function drawLoop() {
  if (!running) return;

  const w = els.canvas.width, h = els.canvas.height;

  // 0) Чистим основной canvas, чтобы не залипало состояние композитинга
  outCtx.save();
  outCtx.globalCompositeOperation = "source-over";
  outCtx.globalAlpha = 1;
  outCtx.clearRect(0, 0, w, h);

  // 1) База — видео
  outCtx.drawImage(els.video, 0, 0, w, h);

  if (maskCanvas.width > 0 && maskCanvas.height > 0) {
    // 2) Собираем offscreen цвет + маску
    buildColorLayer(maskCanvas);

    // 3) Кладём готовый цветовой слой обычным source-over с альфой
    outCtx.globalCompositeOperation = "source-over";
    outCtx.globalAlpha = minmax(parseFloat(els.strength.value || "0.3"), 0, 1);
    outCtx.drawImage(colorCanvas, 0, 0, w, h);

    // 4) Добавляем текстуру волос (multiply — устойчивее в Safari, чем soft-light)
    outCtx.globalAlpha = 1;
    outCtx.globalCompositeOperation = "multiply";
    outCtx.drawImage(els.video, 0, 0, w, h);

    // 5) Возврат к нормальному режиму
    outCtx.globalCompositeOperation = "source-over";
  }

  outCtx.restore();

  // 6) HSL-коррекция финальной картинки
  const adjusted = applyHSLAdjustments(els.canvas);
  outCtx.drawImage(adjusted, 0, 0);

  setStatus(maskCanvas.width ? "Работает ✔ (iOS offscreen)" : "Ожидание маски…");
  drawRaf = requestAnimationFrame(drawLoop);
}

// Сегментация — реже и на уменьшенном кадре
async function segStep() {
  if (!running) return;
  try {
    computeCtx.drawImage(els.video, 0, 0, computeCanvas.width, computeCanvas.height);
    const res = await segmenter.segmentForVideo(computeCanvas, performance.now());

    let alpha = null, w = computeCanvas.width, h = computeCanvas.height;

    if (USE_CONF && res?.confidenceMasks?.[1]) {
      const got = alphaFromConfidenceMask(res.confidenceMasks[1]); // hair=1
      alpha = got.alpha; w = got.w; h = got.h;
    } else if (res?.categoryMask) {
      const cat = res.categoryMask;
      let mw = cat.width ?? cat.cols ?? cat.shape?.[1];
      let mh = cat.height ?? cat.rows ?? cat.shape?.[0];
      let src = typeof cat.getAsUint8Array === "function" ? cat.getAsUint8Array() : cat.data;
      if (!mw || !mh) { mw = w; mh = h; }
      const a = new Float32Array(mw * mh);
      for (let i = 0; i < mw * mh; i++) a[i] = src[i] === 1 ? 1 : 0;
      alpha = a; w = mw; h = mh;
    }

    if (alpha) {
      alpha = emaAlphaBlend(alpha, prevAlpha, ACC.emaAlpha);
      prevAlpha = alpha;

      // Сборка RGBA маски в maskCanvas (compute-res)
      alphaToMaskCanvas(alpha, w, h);
    }
  } catch (e) {
    console.warn("segStep error:", e);
  }
}

/* ================= UI ================= */
els.start.addEventListener("click", async () => {
  try {
    if (!segmenter) await initSegmenter();
    await setupCamera();
    running = true;

    drawRaf && cancelAnimationFrame(drawRaf);
    drawRaf = requestAnimationFrame(drawLoop);

    segTimer && clearInterval(segTimer);
    segTimer = setInterval(segStep, Math.max(16, Math.round(1000 / SEG_FPS)));
  } catch (e) {
    console.error(e);
    setStatus("Ошибка: " + e.message);
  }
});

els.stop.addEventListener("click", () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
  setStatus("Остановлено");
});

els.snap.addEventListener("click", () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
  setStatus("Снимок (заморожено)");
});

els.save.addEventListener("click", () => {
  const url = els.canvas.toDataURL("image/png");
  const a = document.createElement("a");
  a.href = url;
  a.download = "hair_color_demo.png";
  a.click();
});

els.mode.addEventListener("change", () => {
  const solid = els.mode.value === "solid";
  els.solidPanel.classList.toggle("hidden", !solid);
  els.ombrePanel.classList.toggle("hidden", solid);
});

els.presets?.addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-color]");
  if (btn) els.solidColor.value = btn.dataset.color;
});

els.resetHSL.addEventListener("click", () => {
  els.hue.value = 0; els.sat.value = 0; els.light.value = 0;
});

// iOS: аккуратно останавливаем циклы при уходе со страницы
window.addEventListener("pagehide", () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
});
