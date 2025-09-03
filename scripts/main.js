// Hair Color Demo using MediaPipe Tasks Vision (HairSegmenter)
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

// оффскрин для маски
const offMaskCanvas = document.createElement('canvas');
const offMaskCtx = offMaskCanvas.getContext('2d');

function setStatus(t) { els.status.textContent = t ?? ''; }
function minmax(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

async function setupCamera() {
  // важные атрибуты для iOS/Safari
  els.video.setAttribute('playsinline', '');
  els.video.muted = true;

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } },
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

  // сброс композитинга
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'source-over';

  setStatus(`Камера готова ${vw}×${vh}`);
  // Первый кадр «сырая картинка» на канвас
  ctx.drawImage(els.video, 0, 0, vw, vh);

  // Диагностика, если понадобится
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
    outputConfidenceMasks: false
  });
  setStatus('Модель загружена');
}

function buildMaskFromCategoryMask(catMask) {
  // Ширина/высота могут лежать в разных полях в зависимости от версии
  let mw = catMask.width ?? catMask.cols ?? catMask.shape?.[1];
  let mh = catMask.height ?? catMask.rows ?? catMask.shape?.[0];

  // Достаём плоский массив классов (0=фон, 1=волосы)
  let src = null;
  if (catMask.data && catMask.data.length) {
    // (редкий случай старых билдаов) — уже есть data
    src = catMask.data;
  } else if (typeof catMask.getAsUint8Array === 'function') {
    src = catMask.getAsUint8Array();
  } else if (typeof catMask.getAsFloat32Array === 'function') {
    const f = catMask.getAsFloat32Array();
    // Преобразуем float -> бинарные классы (порог 0.5)
    src = new Uint8Array(f.length);
    for (let i = 0; i < f.length; i++) src[i] = f[i] > 0.5 ? 1 : 0;
  } else {
    throw new Error('Unsupported categoryMask format: no .data/.getAs*Array()');
  }

  // Если ширина/высота не нашлись — попробуем вывести из длины
  if (!mw || !mh) {
    // на крайний случай: примем ширину равной ширине канваса
    mw = mw ?? els.canvas.width;
    mh = mh ?? Math.max(1, Math.round(src.length / mw));
  }

  // Собираем альфа-маску (белый RGBA c альфой 255 для волос, 0 — фон)
  offMaskCanvas.width = mw;
  offMaskCanvas.height = mh;
  const id = offMaskCtx.createImageData(mw, mh);
  const data = id.data;

  for (let i = 0; i < mw * mh; i++) {
    const isHair = (src[i] === 1) || (src[i] > 0.5);
    const off = i * 4;
    data[off + 0] = 255;
    data[off + 1] = 255;
    data[off + 2] = 255;
    data[off + 3] = isHair ? 255 : 0;
  }

  offMaskCtx.putImageData(id, 0, 0);
  return offMaskCanvas;
}


function applyHSLAdjustments(baseCanvas) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  // Копируем текущую картинку
  tctx.drawImage(baseCanvas, 0, 0);

  // Lightness: добавим белый/чёрный через screen/multiply
  const light = parseInt(els.light.value || "0", 10); // -100..100
  if (light !== 0) {
    tctx.globalAlpha = minmax(Math.abs(light) / 100, 0, 1);
    tctx.globalCompositeOperation = (light > 0) ? 'screen' : 'multiply';
    tctx.fillStyle = (light > 0) ? '#ffffff' : '#000000';
    tctx.fillRect(0, 0, w, h);
    tctx.globalAlpha = 1;
  }

  // Saturation: ручной микс к/от серого
  const sat = parseInt(els.sat.value || "0", 10); // -100..100
  if (sat !== 0) {
    const img = tctx.getImageData(0, 0, w, h);
    const data = img.data;
    if (sat < 0) {
      // десат: тянем к серому
      const k = (100 - minmax(Math.abs(sat), 0, 100)) / 100;
      for (let i = 0; i < data.length; i += 4) {
        const r = data[i], g = data[i + 1], b = data[i + 2];
        const gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        data[i] = gray + (r - gray) * k;
        data[i + 1] = gray + (g - gray) * k;
        data[i + 2] = gray + (b - gray) * k;
      }
    } else {
      // небольшой буст насыщенности
      const boost = 1 + (sat / 100);
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

  // Hue-rotate (точный) опущен для скорости — для продакшена лучше шейдер/WebGL.
  return tmp;
}

function drawSolidOrGradientMasked(maskCanvas) {
  const w = els.canvas.width, h = els.canvas.height;

  // 1) База: рисуем «сырое» видео
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(els.video, 0, 0, w, h);

  // 2) Подготавливаем цветовой слой
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  const mode = els.mode.value;
  if (mode === 'solid') {
    tctx.fillStyle = els.solidColor.value || '#9b4dff';
  } else {
    const top = els.topColor.value || '#f5d08b';
    const bottom = els.bottomColor.value || '#8e44ad';
    const shift = parseFloat(els.gradientShift.value || "0"); // -0.5..0.5
    const grad = tctx.createLinearGradient(0, h * (0.2 + shift), 0, h * (0.8 + shift));
    grad.addColorStop(0, top);
    grad.addColorStop(1, bottom);
    tctx.fillStyle = grad;
  }
  tctx.fillRect(0, 0, w, h);

  // 3) Feather маски (дешёвое размытие масштабом)
  const feather = parseFloat(els.feather.value || "0.4");
  const fw = Math.max(1, Math.floor(w / (1 + 2 * feather)));
  const fh = Math.max(1, Math.floor(h / (1 + 2 * feather)));
  const featherCanvas = document.createElement('canvas');
  featherCanvas.width = fw; featherCanvas.height = fh;
  const fctx = featherCanvas.getContext('2d');
  fctx.imageSmoothingEnabled = true;
  fctx.imageSmoothingQuality = 'high';
  fctx.drawImage(maskCanvas, 0, 0, fw, fh);

  // 4) Применяем маску волос
  tctx.globalCompositeOperation = 'destination-in';
  tctx.drawImage(featherCanvas, 0, 0, fw, fh, 0, 0, w, h);

  // 5) Накладываем цветовой слой на видео с интенсивностью
  const strength = parseFloat(els.strength.value || "0.65");
  ctx.globalAlpha = minmax(strength, 0, 1);
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(tctx.canvas, 0, 0, w, h);

  // 6) Сохраняем текстуру волос (soft-light с исходным видео)
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'soft-light';
  ctx.drawImage(els.video, 0, 0, w, h);

  // 7) HSL-коррекция на результате
  ctx.globalCompositeOperation = 'source-over';
  const adjusted = applyHSLAdjustments(els.canvas);
  ctx.drawImage(adjusted, 0, 0);
}

async function renderStep(ts) {
  // Всегда рисуем сырое видео (если по какой-то причине сегментация подвисла)
  ctx.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = 1;
  ctx.drawImage(els.video, 0, 0, els.canvas.width, els.canvas.height);

  try {
    const res = await segmenter.segmentForVideo(els.video, ts ?? performance.now());
    if (res && res.categoryMask) {
      const maskCanvas = buildMaskFromCategoryMask(res.categoryMask);
      drawSolidOrGradientMasked(maskCanvas);
      setStatus('Работает ✔');
    } else {
      setStatus('Ожидание маски…');
    }
  } catch (e) {
    console.error('segmentForVideo error:', e);
    setStatus('Сегментация недоступна, показываю сырое видео');
  }
}

function loop(ts) {
  if (!running) return;
  // троттлинг до ~30 FPS
  if (ts - lastTick > 33) {
    renderStep(ts);
    lastTick = ts;
  }
  rafId = requestAnimationFrame(loop);
}

/* ----------------- UI bindings ----------------- */

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
