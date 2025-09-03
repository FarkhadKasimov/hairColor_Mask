// Hair Color Demo — iOS-safe, dual loop + downscale segmentation (no ImageBitmap)
import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

/* ========= PERFORMANCE TUNING ========= */
const SEG_FPS   = 15;     // how often to run segmentation
const SEG_SCALE = 0.5;    // compute resolution scale (0.4–0.6 good)
const USE_CONF  = true;   // use confidence mask if available
const ACC = {
  thresh: 0.60,
  gain: 1.10,
  emaAlpha: 0.25,
  blur: false,
  rethresh: 0.50
};
/* ===================================== */

const els = {
  start: document.getElementById('startBtn'),
  stop: document.getElementById('stopBtn'),
  snap: document.getElementById('snapBtn'),
  save: document.getElementById('saveBtn'),
  video: document.getElementById('video'),
  canvas: document.getElementById('canvas'),
  status: document.getElementById('status'),
  mode: document.getElementById('mode'),
  strength: document.getElementById('strength'),
  feather: document.getElementById('feather'),
  solidColor: document.getElementById('solidColor'),
  presets: document.getElementById('presets'),
  topColor: document.getElementById('topColor'),
  bottomColor: document.getElementById('bottomColor'),
  gradientShift: document.getElementById('gradientShift'),
  solidPanel: document.getElementById('solidPanel'),
  ombrePanel: document.getElementById('ombrePanel'),
  hue: document.getElementById('hue'),
  sat: document.getElementById('sat'),
  light: document.getElementById('light'),
  resetHSL: document.getElementById('resetHSL'),
};

const ctx = els.canvas.getContext('2d');
let segmenter = null;
let running = false;

// compute-size canvas (downscaled input for segmentation)
const computeCanvas = document.createElement('canvas');
const computeCtx = computeCanvas.getContext('2d', { willReadFrequently: true });

// RGBA mask canvas in compute resolution (iOS-safe storage)
const maskCanvas = document.createElement('canvas');
const maskCtx = maskCanvas.getContext('2d');

// last ready mask (we keep as CANVAS, not ImageBitmap)
let lastMaskCanvas = null;
let prevAlpha = null;

let drawRaf = 0;
let segTimer = 0;

function setStatus(t){ if (els.status) els.status.textContent = t ?? ''; }
function clamp01(v){ return v < 0 ? 0 : v > 1 ? 1 : v; }
function minmax(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

/* ============ INIT ============ */
async function setupCamera() {
  // iOS requirements
  els.video.setAttribute('playsinline', '');
  els.video.playsInline = true;
  els.video.muted = true;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } },
    audio: false
  });
  els.video.srcObject = stream;

  await els.video.play();
  await new Promise(r => {
    if (els.video.readyState >= 2) return r();
    els.video.onloadedmetadata = () => r();
  });

  const vw = els.video.videoWidth || 640;
  const vh = els.video.videoHeight || 480;

  els.canvas.width = vw;
  els.canvas.height = vh;

  computeCanvas.width = Math.max(64, Math.round(vw * SEG_SCALE));
  computeCanvas.height = Math.max(64, Math.round(vh * SEG_SCALE));

  maskCanvas.width = computeCanvas.width;
  maskCanvas.height = computeCanvas.height;

  if (els.strength) els.strength.value = "0.3"; // default intensity

  setStatus(`Камера: ${vw}×${vh}, seg: ${computeCanvas.width}×${computeCanvas.height}`);
  ctx.drawImage(els.video, 0, 0, vw, vh);
}

async function initSegmenter() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite"
    },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: USE_CONF
  });
  setStatus('Модель загружена');
}

/* ======== MASK POSTPROCESS ======== */
function alphaFromConfidenceMask(confMask) {
  const w = confMask.width ?? confMask.cols ?? confMask.shape?.[1];
  const h = confMask.height ?? confMask.rows ?? confMask.shape?.[0];
  const f32 = (typeof confMask.getAsFloat32Array === 'function')
    ? confMask.getAsFloat32Array()
    : null;
  if (!f32) throw new Error('No Float32 confidence mask');

  const out = new Float32Array(f32.length);
  const t = ACC.thresh, g = ACC.gain;
  for (let i=0; i<f32.length; i++){
    const v = clamp01(f32[i] * g);
    out[i] = v >= t ? v : 0;
  }
  return { alpha: out, w, h };
}

function blurAndRethreshold(alpha, w, h, rethresh) {
  if (!ACC.blur) return alpha;
  const dst = new Float32Array(alpha.length);
  for (let y=0; y<h; y++){
    const y0 = y-1, y1 = y, y2 = y+1;
    for (let x=0; x<w; x++){
      const x0 = x-1, x1 = x, x2 = x+1;
      let sum = 0;
      const p = (yy, xx) => (xx<0||yy<0||xx>=w||yy>=h) ? 0 : alpha[yy*w+xx];
      sum += p(y0,x0)+p(y0,x1)+p(y0,x2)+p(y1,x0)+p(y1,x1)+p(y1,x2)+p(y2,x0)+p(y2,x1)+p(y2,x2);
      const m = sum / 9;
      dst[y*w + x] = (m >= rethresh) ? m : 0;
    }
  }
  return dst;
}

function emaAlphaBlend(curr, prev, a){
  if (!prev || prev.length !== curr.length) return curr.slice();
  const out = new Float32Array(curr.length);
  const b = 1 - a;
  for (let i=0; i<curr.length; i++){
    out[i] = a*curr[i] + b*prev[i];
  }
  return out;
}

function alphaToMaskCanvas(alpha, w, h) {
  maskCanvas.width = w; maskCanvas.height = h;
  const id = maskCtx.createImageData(w, h);
  const data = id.data;
  for (let i=0; i<alpha.length; i++){
    const a = minmax(Math.round(alpha[i]*255), 0, 255);
    const off = i*4;
    data[off] = 255; data[off+1] = 255; data[off+2] = 255; data[off+3] = a;
  }
  maskCtx.putImageData(id, 0, 0);
  return maskCanvas; // return CANVAS (iOS-safe)
}

/* ============ RENDER ============ */
function applyHSLAdjustments(baseCanvas) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  tctx.drawImage(baseCanvas, 0, 0);

  const light = parseInt(els.light?.value || "0", 10);
  if (light !== 0) {
    tctx.globalAlpha = minmax(Math.abs(light)/100, 0, 1);
    tctx.globalCompositeOperation = (light > 0) ? 'screen' : 'multiply';
    tctx.fillStyle = (light > 0) ? '#ffffff' : '#000000';
    tctx.fillRect(0, 0, w, h);
    tctx.globalAlpha = 1;
  }

  const sat = parseInt(els.sat?.value || "0", 10);
  if (sat !== 0) {
    const img = tctx.getImageData(0,0,w,h);
    const data = img.data;
    if (sat < 0) {
      const k = (100 - Math.min(100, Math.abs(sat))) / 100;
      for (let i=0; i<data.length; i+=4) {
        const r = data[i], g = data[i+1], b = data[i+2];
        const gray = 0.2126*r + 0.7152*g + 0.0722*b;
        data[i]   = gray + (r - gray)*k;
        data[i+1] = gray + (g - gray)*k;
        data[i+2] = gray + (b - gray)*k;
      }
    } else {
      const boost = 1 + (sat/100);
      for (let i=0; i<data.length; i+=4) {
        const r = data[i], g = data[i+1], b = data[i+2];
        const gray = 0.2126*r + 0.7152*g + 0.0722*b;
        data[i]   = gray + (r - gray)*boost;
        data[i+1] = gray + (g - gray)*boost;
        data[i+2] = gray + (b - gray)*boost;
      }
    }
    tctx.putImageData(img, 0, 0);
  }

  return tmp;
}

function drawCompositeWithMask(maskCanv) {
  const w = els.canvas.width, h = els.canvas.height;

  // 1) Base video
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(els.video, 0, 0, w, h);

  // 2) Color layer
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  const mode = els.mode.value;
  if (mode === 'solid') {
    tctx.fillStyle = els.solidColor.value || '#9b4dff';
  } else {
    const top = els.topColor.value || '#f5d08b';
    const bottom = els.bottomColor.value || '#8e44ad';
    const shift = parseFloat(els.gradientShift.value || "0");
    const grad = tctx.createLinearGradient(0, h*(0.2+shift), 0, h*(0.8+shift));
    grad.addColorStop(0, top);
    grad.addColorStop(1, bottom);
    tctx.fillStyle = grad;
  }
  tctx.fillRect(0, 0, w, h);

  // 3) Apply mask (upscale from compute res)
  tctx.globalCompositeOperation = 'destination-in';
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(maskCanv, 0, 0, w, h);

  // 4) Intensity
  const strength = parseFloat(els.strength.value || "0.3");
  ctx.globalAlpha = minmax(strength, 0, 1);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(tctx.canvas, 0, 0, w, h);

  // 5) Preserve hair texture
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'soft-light';
  ctx.drawImage(els.video, 0, 0, w, h);

  // 6) HSL adjustments
  ctx.globalCompositeOperation = 'source-over';
  const adjusted = applyHSLAdjustments(els.canvas);
  ctx.drawImage(adjusted, 0, 0);
}

/* ===== DRAW LOOP (60 FPS) ===== */
function drawLoop() {
  if (!running) return;
  if (lastMaskCanvas) {
    drawCompositeWithMask(lastMaskCanvas);
    setStatus('Работает ✔ (iOS-safe)');
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.globalAlpha = 1;
    ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);
    setStatus('Ожидание первой маски…');
  }
  drawRaf = requestAnimationFrame(drawLoop);
}

/* ===== SEGMENTATION LOOP (SEG_FPS) ===== */
async function segStep() {
  if (!running) return;
  try {
    computeCtx.drawImage(els.video, 0, 0, computeCanvas.width, computeCanvas.height);
    const res = await segmenter.segmentForVideo(computeCanvas, performance.now());

    let alpha = null, w = computeCanvas.width, h = computeCanvas.height;

    if (USE_CONF && res?.confidenceMasks?.[1]) {
      const got = alphaFromConfidenceMask(res.confidenceMasks[1]);
      alpha = got.alpha; w = got.w; h = got.h;
    } else if (res?.categoryMask) {
      const cat = res.categoryMask;
      let mw = cat.width ?? cat.cols ?? cat.shape?.[1];
      let mh = cat.height ?? cat.rows ?? cat.shape?.[0];
      let src = cat.getAsUint8Array ? cat.getAsUint8Array() : cat.data;
      if (!mw || !mh) { mw = w; mh = h; }
      const a = new Float32Array(mw*mh);
      for (let i=0; i<mw*mh; i++) a[i] = (src[i] === 1) ? 1 : 0;
      alpha = a; w = mw; h = mh;
    }

    if (alpha) {
      alpha = emaAlphaBlend(alpha, prevAlpha, ACC.emaAlpha);
      prevAlpha = alpha;
      if (ACC.blur) alpha = blurAndRethreshold(alpha, w, h, ACC.rethresh);

      // build mask CANVAS (iOS-safe) and store it
      lastMaskCanvas = alphaToMaskCanvas(alpha, w, h);
    }
  } catch (e) {
    console.warn('segStep error:', e);
  }
}

/* --------- UI --------- */
els.start.addEventListener('click', async () => {
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
    setStatus('Ошибка: ' + e.message);
  }
});

els.stop.addEventListener('click', () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
  setStatus('Остановлено');
});

els.snap.addEventListener('click', () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
  setStatus('Снимок (заморожено)');
});

els.save.addEventListener('click', () => {
  const url = els.canvas.toDataURL('image/png');
  const a = document.createElement('a');
  a.href = url;
  a.download = 'hair_color_demo.png';
  a.click();
});

els.mode.addEventListener('change', () => {
  const solid = els.mode.value === 'solid';
  els.solidPanel.classList.toggle('hidden', !solid);
  els.ombrePanel.classList.toggle('hidden', solid);
});

els.presets?.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-color]');
  if (btn) els.solidColor.value = btn.dataset.color;
});

els.resetHSL.addEventListener('click', () => {
  els.hue.value = 0;
  els.sat.value = 0;
  els.light.value = 0;
});

// (опционально) подчистка на выгрузке страницы в iOS
window.addEventListener('pagehide', () => {
  running = false;
  drawRaf && cancelAnimationFrame(drawRaf);
  segTimer && clearInterval(segTimer);
});
