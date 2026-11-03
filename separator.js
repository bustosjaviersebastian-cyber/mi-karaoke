// src/demucs/constants.js
var CONSTANTS = {
  SAMPLE_RATE: 44100,
  FFT_SIZE: 4096,
  HOP_SIZE: 1024,
  TRAINING_SAMPLES: 343980,
  MODEL_SPEC_BINS: 2048,
  MODEL_SPEC_FRAMES: 336,
  SEGMENT_OVERLAP: 0.25,
  TRACKS: ["drums", "bass", "other", "vocals"],
  // Default model URL (Hugging Face Hub)
  DEFAULT_MODEL_URL: "https://huggingface.co/timcsy/demucs-web-onnx/resolve/main/htdemucs_embedded.onnx"
};

// src/demucs/fft.js
var fftTwiddles = /* @__PURE__ */ new Map();
var ifftTwiddles = /* @__PURE__ */ new Map();
var hannWindows = /* @__PURE__ */ new Map();
function getFFTTwiddles(n) {
  if (fftTwiddles.has(n)) return fftTwiddles.get(n);
  const real = new Float32Array(n / 2);
  const imag = new Float32Array(n / 2);
  for (let k = 0; k < n / 2; k++) {
    const angle = -2 * Math.PI * k / n;
    real[k] = Math.cos(angle);
    imag[k] = Math.sin(angle);
  }
  const twiddles = { real, imag };
  fftTwiddles.set(n, twiddles);
  return twiddles;
}
function getIFFTTwiddles(n) {
  if (ifftTwiddles.has(n)) return ifftTwiddles.get(n);
  const real = new Float32Array(n / 2);
  const imag = new Float32Array(n / 2);
  for (let k = 0; k < n / 2; k++) {
    const angle = 2 * Math.PI * k / n;
    real[k] = Math.cos(angle);
    imag[k] = Math.sin(angle);
  }
  const twiddles = { real, imag };
  ifftTwiddles.set(n, twiddles);
  return twiddles;
}
function getHannWindow(size) {
  if (hannWindows.has(size)) return hannWindows.get(size);
  const window = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / size));
  }
  hannWindows.set(size, window);
  return window;
}
function bitReverse(n, bits) {
  let result = 0;
  for (let i = 0; i < bits; i++) {
    result = result << 1 | n & 1;
    n >>= 1;
  }
  return result;
}
function fft(realOut, imagOut, realIn, n) {
  const bits = Math.log2(n) | 0;
  const twiddles = getFFTTwiddles(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    realOut[i] = realIn[j];
    imagOut[i] = 0;
  }
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const k = j * step;
        const tReal = twiddles.real[k];
        const tImag = twiddles.imag[k];
        const idx1 = i + j;
        const idx2 = i + j + halfSize;
        const eReal = realOut[idx1];
        const eImag = imagOut[idx1];
        const oReal = realOut[idx2] * tReal - imagOut[idx2] * tImag;
        const oImag = realOut[idx2] * tImag + imagOut[idx2] * tReal;
        realOut[idx1] = eReal + oReal;
        imagOut[idx1] = eImag + oImag;
        realOut[idx2] = eReal - oReal;
        imagOut[idx2] = eImag - oImag;
      }
    }
  }
}
function ifft(realOut, imagOut, realIn, imagIn, n) {
  const bits = Math.log2(n) | 0;
  const twiddles = getIFFTTwiddles(n);
  for (let i = 0; i < n; i++) {
    const j = bitReverse(i, bits);
    realOut[i] = realIn[j];
    imagOut[i] = imagIn[j];
  }
  for (let size = 2; size <= n; size *= 2) {
    const halfSize = size / 2;
    const step = n / size;
    for (let i = 0; i < n; i += size) {
      for (let j = 0; j < halfSize; j++) {
        const k = j * step;
        const tReal = twiddles.real[k];
        const tImag = twiddles.imag[k];
        const idx1 = i + j;
        const idx2 = i + j + halfSize;
        const eReal = realOut[idx1];
        const eImag = imagOut[idx1];
        const oReal = realOut[idx2] * tReal - imagOut[idx2] * tImag;
        const oImag = realOut[idx2] * tImag + imagOut[idx2] * tReal;
        realOut[idx1] = eReal + oReal;
        imagOut[idx1] = eImag + oImag;
        realOut[idx2] = eReal - oReal;
        imagOut[idx2] = eImag - oImag;
      }
    }
  }
  for (let i = 0; i < n; i++) {
    realOut[i] /= n;
    imagOut[i] /= n;
  }
}
function stft(signal, fftSize, hopSize) {
  const numFrames = Math.floor((signal.length - fftSize) / hopSize) + 1;
  const numBins = fftSize / 2 + 1;
  const window = getHannWindow(fftSize);
  const scale = 1 / Math.sqrt(fftSize);
  const specReal = new Float32Array(numFrames * numBins);
  const specImag = new Float32Array(numFrames * numBins);
  const frameReal = new Float32Array(fftSize);
  const frameImag = new Float32Array(fftSize);
  const windowedFrame = new Float32Array(fftSize);
  for (let frame = 0; frame < numFrames; frame++) {
    const start = frame * hopSize;
    for (let i = 0; i < fftSize; i++) {
      windowedFrame[i] = signal[start + i] * window[i];
    }
    fft(frameReal, frameImag, windowedFrame, fftSize);
    const outOffset = frame * numBins;
    for (let k = 0; k < numBins; k++) {
      specReal[outOffset + k] = frameReal[k] * scale;
      specImag[outOffset + k] = frameImag[k] * scale;
    }
  }
  return { real: specReal, imag: specImag, numFrames, numBins };
}
function istft(specReal, specImag, numFrames, numBins, fftSize, hopSize, length) {
  const outputLength = length || (numFrames - 1) * hopSize + fftSize;
  const output = new Float32Array(outputLength);
  const windowSum = new Float32Array(outputLength);
  const window = getHannWindow(fftSize);
  const scale = Math.sqrt(fftSize);
  const fullReal = new Float32Array(fftSize);
  const fullImag = new Float32Array(fftSize);
  const outReal = new Float32Array(fftSize);
  const outImag = new Float32Array(fftSize);
  for (let frame = 0; frame < numFrames; frame++) {
    fullReal.fill(0);
    fullImag.fill(0);
    for (let k = 0; k < numBins; k++) {
      fullReal[k] = specReal[frame * numBins + k];
      fullImag[k] = specImag[frame * numBins + k];
    }
    for (let k = 1; k < numBins - 1; k++) {
      fullReal[fftSize - k] = fullReal[k];
      fullImag[fftSize - k] = -fullImag[k];
    }
    ifft(outReal, outImag, fullReal, fullImag, fftSize);
    const start = frame * hopSize;
    for (let i = 0; i < fftSize && start + i < outputLength; i++) {
      output[start + i] += outReal[i] * window[i] * scale;
      windowSum[start + i] += window[i] * window[i];
    }
  }
  for (let i = 0; i < outputLength; i++) {
    if (windowSum[i] > 1e-8) {
      output[i] /= windowSum[i];
    }
  }
  return output;
}
function reflectPad(signal, padLeft, padRight) {
  const length = signal.length;
  const output = new Float32Array(padLeft + length + padRight);
  for (let i = 0; i < padLeft; i++) {
    const srcIdx = Math.min(padLeft - i, length - 1);
    output[i] = signal[srcIdx];
  }
  output.set(signal, padLeft);
  for (let i = 0; i < padRight; i++) {
    const srcIdx = Math.max(0, length - 2 - i);
    output[padLeft + length + i] = signal[srcIdx];
  }
  return output;
}

// src/demucs/processor.js
var { SAMPLE_RATE, FFT_SIZE, HOP_SIZE, TRAINING_SAMPLES, MODEL_SPEC_BINS, MODEL_SPEC_FRAMES, SEGMENT_OVERLAP, TRACKS } = CONSTANTS;
function standaloneMask(freqOutput) {
  const numTracks = 4;
  const numChannels = 4;
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const result = [];
  for (let t = 0; t < numTracks; t++) {
    const trackSpec = {
      leftReal: new Float32Array(numBins * numFrames),
      leftImag: new Float32Array(numBins * numFrames),
      rightReal: new Float32Array(numBins * numFrames),
      rightImag: new Float32Array(numBins * numFrames)
    };
    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const baseIdx = t * numChannels * numBins * numFrames;
        const outIdx = b * numFrames + f;
        trackSpec.leftReal[outIdx] = freqOutput[baseIdx + 0 * numBins * numFrames + b * numFrames + f];
        trackSpec.leftImag[outIdx] = freqOutput[baseIdx + 1 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightReal[outIdx] = freqOutput[baseIdx + 2 * numBins * numFrames + b * numFrames + f];
        trackSpec.rightImag[outIdx] = freqOutput[baseIdx + 3 * numBins * numFrames + b * numFrames + f];
      }
    }
    result.push(trackSpec);
  }
  return result;
}
function standaloneIspec(trackSpec, targetLength) {
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const hopLength = HOP_SIZE;
  const paddedBins = numBins + 1;
  const paddedFrames = numFrames + 4;
  const padChannel = (real, imag) => {
    const paddedReal = new Float32Array(paddedFrames * paddedBins);
    const paddedImag = new Float32Array(paddedFrames * paddedBins);
    for (let f = 0; f < numFrames; f++) {
      for (let b = 0; b < numBins; b++) {
        const srcIdx = b * numFrames + f;
        const dstFrame = f + 2;
        const dstIdx = dstFrame * paddedBins + b;
        paddedReal[dstIdx] = real[srcIdx];
        paddedImag[dstIdx] = imag[srcIdx];
      }
    }
    return { real: paddedReal, imag: paddedImag };
  };
  const leftPadded = padChannel(trackSpec.leftReal, trackSpec.leftImag);
  const rightPadded = padChannel(trackSpec.rightReal, trackSpec.rightImag);
  const centerPad = FFT_SIZE / 2;
  const pad = Math.floor(hopLength / 2) * 3;
  const istftLength = (paddedFrames - 1) * hopLength + FFT_SIZE;
  const leftOut = istft(leftPadded.real, leftPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);
  const rightOut = istft(rightPadded.real, rightPadded.imag, paddedFrames, paddedBins, FFT_SIZE, hopLength, istftLength);
  const totalOffset = centerPad + pad;
  const left = leftOut.subarray(totalOffset, totalOffset + targetLength);
  const right = rightOut.subarray(totalOffset, totalOffset + targetLength);
  return { left: new Float32Array(left), right: new Float32Array(right) };
}
function prepareModelInput(leftChannel, rightChannel) {
  const inputLength = TRAINING_SAMPLES;
  const paddedLeft = new Float32Array(inputLength);
  const paddedRight = new Float32Array(inputLength);
  const copyLen = Math.min(leftChannel.length, inputLength);
  paddedLeft.set(leftChannel.subarray(0, copyLen));
  paddedRight.set(rightChannel.subarray(0, copyLen));
  const le = Math.ceil(inputLength / HOP_SIZE);
  const pad = Math.floor(HOP_SIZE / 2) * 3;
  const padRight = pad + le * HOP_SIZE - inputLength;
  const stftInputLeft = reflectPad(paddedLeft, pad, padRight);
  const stftInputRight = reflectPad(paddedRight, pad, padRight);
  const centerPad = FFT_SIZE / 2;
  const centeredLeft = reflectPad(stftInputLeft, centerPad, centerPad);
  const centeredRight = reflectPad(stftInputRight, centerPad, centerPad);
  const stftLeft = stft(centeredLeft, FFT_SIZE, HOP_SIZE);
  const stftRight = stft(centeredRight, FFT_SIZE, HOP_SIZE);
  const numBins = MODEL_SPEC_BINS;
  const numFrames = MODEL_SPEC_FRAMES;
  const frameOffset = 2;
  const magSpec = new Float32Array(4 * numBins * numFrames);
  for (let f = 0; f < numFrames; f++) {
    const srcFrame = f + frameOffset;
    for (let b = 0; b < numBins; b++) {
      const srcIdx = srcFrame * stftLeft.numBins + b;
      magSpec[0 * numBins * numFrames + b * numFrames + f] = stftLeft.real[srcIdx];
      magSpec[1 * numBins * numFrames + b * numFrames + f] = stftLeft.imag[srcIdx];
      magSpec[2 * numBins * numFrames + b * numFrames + f] = stftRight.real[srcIdx];
      magSpec[3 * numBins * numFrames + b * numFrames + f] = stftRight.imag[srcIdx];
    }
  }
  const waveform = new Float32Array(2 * inputLength);
  waveform.set(paddedLeft, 0);
  waveform.set(paddedRight, inputLength);
  return { waveform, magSpec, numBins, numFrames, originalLength: leftChannel.length };
}

// src/separator.src.js
import { Mp3Encoder } from "./lame.js";
var ORT_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
var MODEL_URL = CONSTANTS.DEFAULT_MODEL_URL;
var CACHE = "mk-model-v1";
var { TRAINING_SAMPLES: N, SEGMENT_OVERLAP: SEGMENT_OVERLAP2, SAMPLE_RATE: SR, MODEL_SPEC_BINS: B, MODEL_SPEC_FRAMES: F } = CONSTANTS;
var TEST_SECONDS = 30;
var pause = () => new Promise((r) => setTimeout(r));
var cancelado = () => new DOMException("Cancelado", "AbortError");
var motor = null;
async function hasWebGPU() {
  try {
    return !!(navigator.gpu && await navigator.gpu.requestAdapter());
  } catch {
    return false;
  }
}
async function modeloListo() {
  try {
    return !!await (await caches.open(CACHE)).match(MODEL_URL);
  } catch {
    return false;
  }
}
async function obtenerModelo(onEstado, signal) {
  const cache = await caches.open(CACHE), hit = await cache.match(MODEL_URL);
  if (hit) return hit.arrayBuffer();
  if (!navigator.onLine) throw new Error("Hace falta internet para descargar el modelo la primera vez.");
  const res = await fetch(MODEL_URL, { signal });
  if (!res.ok) throw new Error("No se pudo descargar el modelo (HTTP " + res.status + ").");
  const total = +res.headers.get("Content-Length") || 0;
  let buf;
  if (total && res.body) {
    buf = new Uint8Array(total);
    let got = 0;
    const rd = res.body.getReader();
    for (; ; ) {
      const { done, value } = await rd.read();
      if (done) break;
      buf.set(value, got);
      got += value.length;
      onEstado({ fase: "descarga", frac: got / total, mb: got / 1048576, totalMb: total / 1048576 });
    }
    buf = buf.buffer;
  } else buf = await res.arrayBuffer();
  try {
    await cache.put(MODEL_URL, new Response(buf.slice(0), { headers: { "Content-Type": "application/octet-stream" } }));
  } catch {
  }
  return buf;
}
async function cargarMotor(onEstado, signal) {
  if (motor) return motor;
  onEstado({ fase: "motor" });
  const ort = await import(ORT_BASE + "ort.webgpu.min.mjs");
  ort.env.wasm.wasmPaths = ORT_BASE;
  const buf = await obtenerModelo(onEstado, signal);
  onEstado({ fase: "cargando" });
  const gpu = await hasWebGPU();
  const sess = await ort.InferenceSession.create(buf, { executionProviders: gpu ? ["webgpu", "wasm"] : ["wasm"], graphOptimizationLevel: "basic" });
  return motor = { sess, Tensor: ort.Tensor, gpu };
}
async function decodificar(file, prueba) {
  const ac = new OfflineAudioContext(2, SR, SR);
  const ab = await ac.decodeAudioData(await file.arrayBuffer());
  let ini = 0, n = ab.length;
  if (prueba) {
    n = Math.min(TEST_SECONDS * SR, ab.length);
    ini = Math.floor(Math.max(0, Math.min(ab.length * 0.3, ab.length - n)));
  }
  const L = ab.getChannelData(0).slice(ini, ini + n);
  const R = (ab.numberOfChannels > 1 ? ab.getChannelData(1) : ab.getChannelData(0)).slice(ini, ini + n);
  return { L, R };
}
async function separarInstrumental(L, R, { sess, Tensor, onProg = () => {
}, signal } = {}) {
  const total = L.length, stride = Math.floor(N * (1 - SEGMENT_OVERLAP2)), half = stride * 0.5;
  const nSeg = Math.max(1, Math.ceil((total - N) / stride) + 1);
  const oL = new Float32Array(total), oR = new Float32Array(total), w = new Float32Array(total);
  const t0 = performance.now();
  let k = 0;
  for (let start = 0; start < total; start += stride) {
    if (signal && signal.aborted) throw cancelado();
    const end = Math.min(start + N, total), len = end - start;
    const sL = new Float32Array(N), sR = new Float32Array(N);
    sL.set(L.subarray(start, end));
    sR.set(R.subarray(start, end));
    const inp = prepareModelInput(sL, sR), feeds = {};
    feeds[sess.inputNames[0]] = new Tensor("float32", inp.waveform, [1, 2, N]);
    if (sess.inputNames.length > 1) feeds[sess.inputNames[1]] = new Tensor("float32", inp.magSpec, [1, 4, B, F]);
    const res = await sess.run(feeds);
    let td = null, ts = null, fd = null;
    for (const name of sess.outputNames) {
      const t = res[name];
      if (t.dims.length === 4 && t.dims[2] === 2) {
        td = t.data;
        ts = t.dims;
      } else if (t.dims.length === 5 && t.dims[2] === 4) fd = t.data;
    }
    if (!td) throw new Error("La salida del modelo no tiene la forma esperada.");
    const ch = ts[2], sm = ts[3], specs = fd ? standaloneMask(fd) : null;
    for (let t = 0; t < 3; t++) {
      const f = specs ? standaloneIspec(specs[t], N) : null, bl = t * ch * sm, br = bl + sm;
      for (let i = 0; i < len; i++) {
        const wv = Math.min(i / half, 1, (len - i) / half);
        oL[start + i] += (td[bl + i] + (f ? f.left[i] : 0)) * wv;
        oR[start + i] += (td[br + i] + (f ? f.right[i] : 0)) * wv;
      }
    }
    for (let i = 0; i < len; i++) w[start + i] += Math.min(i / half, 1, (len - i) / half);
    k++;
    const elapsed = (performance.now() - t0) / 1e3;
    onProg({ frac: k / nSeg, seg: k, nSeg, elapsed, eta: elapsed / k * (nSeg - k) });
    await pause();
  }
  let peak = 0;
  for (let i = 0; i < total; i++) {
    if (w[i] > 0) {
      oL[i] /= w[i];
      oR[i] /= w[i];
    }
    peak = Math.max(peak, Math.abs(oL[i]), Math.abs(oR[i]));
  }
  if (peak > 0.98) {
    const g = 0.98 / peak;
    for (let i = 0; i < total; i++) {
      oL[i] *= g;
      oR[i] *= g;
    }
  }
  return { L: oL, R: oR, separacion: (performance.now() - t0) / 1e3 };
}
var aI16 = (a) => {
  const o = new Int16Array(a.length);
  for (let i = 0; i < a.length; i++) {
    const v = Math.max(-1, Math.min(1, a[i]));
    o[i] = v < 0 ? v * 32768 : v * 32767;
  }
  return o;
};
async function aMp3(L, R, signal) {
  const enc = new Mp3Encoder(2, SR, 160), bloque = 1152 * 40, partes = [];
  for (let i = 0, c = 0; i < L.length; i += bloque, c++) {
    if (signal && signal.aborted) throw cancelado();
    const b = enc.encodeBuffer(aI16(L.subarray(i, i + bloque)), aI16(R.subarray(i, i + bloque)));
    if (b.length) partes.push(new Uint8Array(b));
    if (c % 10 === 0) await pause();
  }
  const f = enc.flush();
  if (f.length) partes.push(new Uint8Array(f));
  return new Blob(partes, { type: "audio/mpeg" });
}
async function crearInstrumental(file, { prueba = false, onEstado = () => {
}, signal } = {}) {
  onEstado({ fase: "decodificando" });
  const { L, R } = await decodificar(file, prueba);
  const m = await cargarMotor(onEstado, signal);
  const out = await separarInstrumental(L, R, { sess: m.sess, Tensor: m.Tensor, signal, onProg: (p) => onEstado({ fase: "separando", ...p }) });
  onEstado({ fase: "comprimiendo" });
  const blob = await aMp3(out.L, out.R, signal);
  return { blob, segundos: L.length / SR, separacion: out.separacion, gpu: m.gpu };
}
export {
  aMp3,
  crearInstrumental,
  hasWebGPU,
  modeloListo,
  separarInstrumental
};
