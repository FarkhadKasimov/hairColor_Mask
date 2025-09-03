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
let offMaskCanvas = document.createElement('canvas');
let offMaskCtx = offMaskCanvas.getContext('2d');

function setStatus(t) { els.status.textContent = t; }

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
  els.video.srcObject = stream;
  await els.video.play();
  await new Promise(r => els.video.onloadedmetadata = r);
  els.canvas.width = els.video.videoWidth;
  els.canvas.height = els.video.videoHeight;
  setStatus('Камера готова');
}

async function initSegmenter() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
  );
  segmenter = await ImageSegmenter.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/hair_segmenter/float32/1/hair_segmenter.tflite"
    },
    runningMode: "VIDEO",
    outputCategoryMask: true,
    outputConfidenceMasks: false
  });
  setStatus('Модель загружена');
}

function applyHSLAdjustments(baseCanvas) {
  const w = els.canvas.width;
  const h = els.canvas.height;
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  // Copy current canvas (already masked & colored) -> tmp
  tctx.drawImage(baseCanvas, 0, 0);

  // Lightness: add white or black overlay
  const light = parseInt(els.light.value, 10); // -100..100
  if (light !== 0) {
    tctx.globalAlpha = Math.min(1, Math.abs(light)/100);
    tctx.globalCompositeOperation = (light > 0) ? 'screen' : 'multiply';
    tctx.fillStyle = (light > 0) ? '#ffffff' : '#000000';
    tctx.fillRect(0, 0, w, h);
    tctx.globalAlpha = 1;
  }

  // Saturation: lerp towards gray or boost
  const sat = parseInt(els.sat.value, 10); // -100..100
  if (sat !== 0) {
    const img = tctx.getImageData(0,0,w,h);
    const data = img.data;
    const k = (100 - minmax(Math.abs(sat), 0, 100)) / 100;
    for (let i=0; i<data.length; i+=4) {
      const r = data[i], g = data[i+1], b = data[i+2];
      const gray = 0.2126*r + 0.7152*g + 0.0722*b;
      if (sat < 0) {
        data[i]   = gray + (r - gray)*k;
        data[i+1] = gray + (g - gray)*k;
        data[i+2] = gray + (b - gray)*k;
      } else {
        const boost = 1 + (sat/100);
        data[i]   = gray + (r - gray)*boost;
        data[i+1] = gray + (g - gray)*boost;
        data[i+2] = gray + (b - gray)*boost;
      }
    }
    tctx.putImageData(img, 0, 0);
  }

  // Hue rotation is skipped for performance in 2D canvas demo.
  return tmp;
}

function minmax(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

function drawSolidOrGradientMasked(maskCanvas) {
  const w = els.canvas.width, h = els.canvas.height;
  // Draw base video
  ctx.drawImage(els.video, 0, 0, w, h);

  // Prepare colored layer
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');

  // Fill with color/gradient
  const mode = els.mode.value;
  if (mode === 'solid') {
    tctx.fillStyle = els.solidColor.value;
  } else {
    const top = els.topColor.value;
    const bottom = els.bottomColor.value;
    const shift = parseFloat(els.gradientShift.value); // -0.5..0.5
    const grad = tctx.createLinearGradient(0, h*(0.2+shift), 0, h*(0.8+shift));
    grad.addColorStop(0, top);
    grad.addColorStop(1, bottom);
    tctx.fillStyle = grad;
  }
  tctx.fillRect(0, 0, w, h);

  // Feather edges by drawing mask scaled down & up (cheap blur)
  const feather = parseFloat(els.feather.value);
  const featherCanvas = document.createElement('canvas');
  const fw = Math.max(1, Math.floor(w / (1 + 2*feather)));
  const fh = Math.max(1, Math.floor(h / (1 + 2*feather)));
  featherCanvas.width = fw; featherCanvas.height = fh;
  const fctx = featherCanvas.getContext('2d');
  fctx.drawImage(maskCanvas, 0, 0, fw, fh);

  // Mask into hair with feathered mask
  tctx.globalCompositeOperation = 'destination-in';
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  tctx.drawImage(featherCanvas, 0, 0, fw, fh, 0, 0, w, h);

  // Place over the base video with intensity
  const strength = parseFloat(els.strength.value);
  ctx.globalAlpha = strength;
  ctx.globalCompositeOperation = 'source-over';
  ctx.drawImage(tctx.canvas, 0, 0, w, h);

  // Soft-light original to preserve texture
  ctx.globalAlpha = 1;
  ctx.globalCompositeOperation = 'soft-light';
  ctx.drawImage(els.video, 0, 0, w, h);

  // HSL adjustments on result
  ctx.globalCompositeOperation = 'source-over';
  const adjusted = applyHSLAdjustments(els.canvas);
  ctx.drawImage(adjusted, 0, 0);
}

function buildMaskFromCategoryMask(catMask) {
  const mw = catMask.width, mh = catMask.height;
  offMaskCanvas.width = mw; offMaskCanvas.height = mh;
  const mctx = offMaskCtx;
  const id = mctx.createImageData(mw, mh);
  const data = id.data;
  const src = catMask.data;
  for (let i=0; i<mw*mh; i++) {
    const isHair = src[i] === 1;
    const off = i*4;
    data[off] = 255; data[off+1] = 255; data[off+2] = 255; data[off+3] = isHair ? 255 : 0;
  }
  mctx.putImageData(id, 0, 0);
  return offMaskCanvas;
}

async function renderLoop() {
  if (!running) return;
  const t = performance.now();
  const result = await segmenter.segmentForVideo(els.video, t);
  const maskCanvas = buildMaskFromCategoryMask(result.categoryMask);

  drawSolidOrGradientMasked(maskCanvas);
  rafId = requestAnimationFrame(renderLoop);
}

els.start.addEventListener('click', async () => {
  try {
    if (!segmenter) await initSegmenter();
    await setupCamera();
    running = true;
    setStatus('Работает ✔');
    renderLoop();
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