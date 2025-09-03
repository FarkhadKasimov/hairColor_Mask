// Hair Color Demo using MediaPipe Tasks Vision (HairSegmenter) — accuracy tuned
import { ImageSegmenter, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2";

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
let rafId = null;
let lastTick = 0;

// ====== Тонкая настройка точности ======
const ACC = {
  thresh: 0.60,      // порог уверенности (0..1)
  gain: 1.20,        // усиление вероятностей перед порогом
  emaAlpha: 0.35,    // коэффициент EMA (0..1) — чем больше, тем быстрее реагирует
  blur: true,        // вкл. 3×3 blur по альфе
  rethresh: 0.50     // порог после blur (для подчистки)
};

let prevAlpha = null;  // Float32Array для EMA

function setStatus(t){ els.status.textContent = t ?? ''; }
function clamp01(v){ return v < 0 ? 0 : v > 1 ? 1 : v; }
function minmax(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

async function setupCamera() {
  els.video.setAttribute('playsinline', '');
  els.video.muted = true;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 960 }, height: { ideal: 720 } }, // чуть выше разрешение → точнее маска
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

  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';
  setStatus(`Камера готова ${vw}×${vh}`);
  ctx.drawImage(els.video, 0, 0, vw, vh);

  // дефолтная интенсивность 0.3
  if (els.strength) els.strength.value = "0.3";

  console.debug('video readyState:', els.video.readyState, 'size:', vw, vh);
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
    outputConfidenceMasks: true   // важное: включаем confidence masks
  });
  setStatus('Модель загружена');
}

/* ---------- Пост-обработка маски для точности ---------- */
// достаём альфу (0..1) из confidence mask класса hair
function alphaFromConfidenceMask(confMask) {
  // берём «волосы» = индекс 1
  const w = confMask.width ?? confMask.cols ?? confMask.shape?.[1];
  const h = confMask.height ?? confMask.rows ?? confMask.shape?.[0];

  const f32 = (typeof confMask.getAsFloat32Array === 'function')
    ? confMask.getAsFloat32Array()
    : null;
  if (!f32) throw new Error('No Float32 confidence mask');

  // усиление и порог
  const out = new Float32Array(f32.length);
  const t = ACC.thresh;
  const g = ACC.gain;
  for (let i=0; i<f32.length; i++){
    const v = clamp01(f32[i] * g);
    out[i] = v >= t ? v : 0;
  }
  return { alpha: out, w, h };
}

// простенький 3×3 box blur по альфе (в Float32), затем повторный порог
function blurAndRethreshold(alpha, w, h, rethresh) {
  if (!ACC.blur) return alpha;
  const dst = new Float32Array(alpha.length);
  for (let y=0; y<h; y++){
    const y0 = y-1, y1 = y, y2 = y+1;
    for (let x=0; x<w; x++){
      const x0 = x-1, x1 = x, x2 = x+1;
      let sum = 0, cnt = 0;
      const p = (yy, xx) => {
        if (xx<0 || yy<0 || xx>=w || yy>=h) return 0;
        return alpha[yy*w + xx];
      };
      sum += p(y0,x0); cnt++;
      sum += p(y0,x1); cnt++;
      sum += p(y0,x2); cnt++;
      sum += p(y1,x0); cnt++;
      sum += p(y1,x1); cnt++;
      sum += p(y1,x2); cnt++;
      sum += p(y2,x0); cnt++;
      sum += p(y2,x1); cnt++;
      sum += p(y2,x2); cnt++;
      const m = sum / cnt;
      dst[y*w + x] = (m >= rethresh) ? m : 0;
    }
  }
  return dst;
}

// EMA по альфе, чтобы маска меньше «дышала» (вес текущего кадра = emaAlpha)
function emaAlphaBlend(curr, prev, a){
  if (!prev || prev.length !== curr.length) return curr.slice();
  const out = new Float32Array(curr.length);
  const b = 1 - a;
  for (let i=0; i<curr.length; i++){
    out[i] = a*curr[i] + b*prev[i];
  }
  return out;
}

// превращаем Float32 альфу (0..1) в RGBA ImageData маски (белый с альфой)
function alphaToMaskCanvas(alpha, w, h, canvas, ctx2d) {
  canvas.width = w; canvas.height = h;
  const id = ctx2d.createImageData(w, h);
  const data = id.data;
  for (let i=0; i<alpha.length; i++){
    const a = minmax(Math.round(alpha[i]*255), 0, 255);
    const off = i*4;
    data[off+0] = 255;
    data[off+1] = 255;
    data[off+2] = 255;
    data[off+3] = a;
  }
  ctx2d.putImageData(id, 0, 0);
  return canvas;
}

// оффскрин под готовую маску
const offMaskCanvas = document.createElement('canvas');
const offMaskCtx = offMaskCanvas.getContext('2d');

/* ---------- Рендер ---------- */
function applyHSLAdjustments(baseCanvas) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  tctx.drawImage(baseCanvas, 0, 0);

  const light = parseInt(els.light.value || "0", 10);
  if (light !== 0) {
    tctx.globalAlpha = minmax(Math.abs(light)/100, 0, 1);
    tctx.globalCompositeOperation = (light > 0) ? 'screen' : 'multiply';
    tctx.fillStyle = (light > 0) ? '#ffffff' : '#000000';
    tctx.fillRect(0, 0, w, h);
    tctx.globalAlpha = 1;
  }

  const sat = parseInt(els.sat.value || "0", 10);
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

function drawSolidOrGradientMasked(maskCanvas) {
  const w = els.canvas.width, h = els.canvas.height;

  // База — «сырое» видео
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(els.video, 0, 0, w, h);

  // Цветовой слой
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

  // Feather через даун/апскейл
  const feather = parseFloat(els.feather.value || "0.4");
  const fw = Math.max(1, Math.floor(w / (1 + 2*feather)));
  const fh = Math.max(1, Math.floor(h / (1 + 2*feather)));
  const featherCanvas = document.createElement('canvas');
  featherCanvas.width = fw; featherCanvas.height = fh;
  const fctx = featherCanvas.getContext('2d');
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = 'high';
  fctx.drawImage(maskCanvas, 0, 0, fw, fh);

  // Маска → цвет
  tctx.globalCompositeOperation = 'destination-in';
  tctx.drawImage(featherCanvas, 0, 0, fw, fh, 0, 0, w, h);

  // Интенсивность (по умолчанию 0.3 выставлена в setupCamera)
  const strength = parseFloat(els.strength.value || "0.3");
  ctx.globalAlpha = minmax(strength, 0, 1);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(tctx.canvas, 0, 0, w, h);

  // Текстура волос
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'soft-light';
  ctx.drawImage(els.video, 0, 0, w, h);

  // HSL-коррекция
  ctx.globalCompositeOperation = 'source-over';
  const adjusted = applyHSLAdjustments(els.canvas);
  ctx.drawImage(adjusted, 0, 0);
}

async function renderStep(ts) {
  // всегда есть «сырой» кадр
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

  try {
    const res = await segmenter.segmentForVideo(els.video, ts ?? performance.now());
    // берём confidenceMasks[1] — вероятность «волосы»
    const conf = res?.confidenceMasks?.[1];
    const cat = res?.categoryMask;

    if (conf) {
      // от confidence к альфе
      let { alpha, w, h } = alphaFromConfidenceMask(conf);
      // EMA
      alpha = emaAlphaBlend(alpha, prevAlpha, ACC.emaAlpha);
      prevAlpha = alpha;
      // blur + повторный порог
      alpha = blurAndRethreshold(alpha, w, h, ACC.rethresh);
      // альфа → RGBA маска-канвас
      const maskCanvas = alphaToMaskCanvas(alpha, w, h, offMaskCanvas, offMaskCtx);
      drawSolidOrGradientMasked(maskCanvas);
      setStatus('Работает ✔ (confidence)');
      return;
    }

    // fallback: категория (на случай отсутствия confidence mask)
    if (cat) {
      // старый путь — просто бинарная маска
      let mw = cat.width ?? cat.cols ?? cat.shape?.[1];
      let mh = cat.height ?? cat.rows ?? cat.shape?.[0];
      let src = null;
      if (cat.getAsUint8Array) src = cat.getAsUint8Array();
      else if (cat.data) src = cat.data;
      if (!mw || !mh) { mw = els.canvas.width; mh = Math.max(1, Math.round(src.length / mw)); }

      const alpha = new Float32Array(mw*mh);
      for (let i=0; i<mw*mh; i++) alpha[i] = (src[i] === 1) ? 1 : 0;
      // EMA и blur тоже применим
      const sm = emaAlphaBlend(alpha, prevAlpha, ACC.emaAlpha);
      prevAlpha = sm;
      const alpha2 = blurAndRethreshold(sm, mw, mh, ACC.rethresh);
      const maskCanvas = alphaToMaskCanvas(alpha2, mw, mh, offMaskCanvas, offMaskCtx);
      drawSolidOrGradientMasked(maskCanvas);
      setStatus('Работает ✔ (category)');
      return;
    }

    setStatus('Ожидание маски…');
  } catch (e) {
    console.error('segmentForVideo error:', e);
    setStatus('Сегментация недоступна, показываю сырое видео');
  }
}

function loop(ts) {
  if (!running) return;
  if (ts - lastTick > 33) { // ~30 FPS
    renderStep(ts);
    lastTick = ts;
  }
  rafId = requestAnimationFrame(loop);
}

/* --------- UI --------- */

els.start.addEventListener('click', async () => {
  try {
    if (!segmenter) await initSegmenter();
    await setupCamera();
    running = true;
    lastTick = 0;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(loop);
  } catch (e) {
    console.error(e);
    setStatus('Ошибка: ' + e.message);
  }
});

els.stop.addEventListener('click', () => {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  setStatus('Остановлено');
});

els.snap.addEventListener('click', () => {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
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
