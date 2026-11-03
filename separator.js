// src/dsp.src.js
import { Mp3Encoder } from "./lame.js";
var SR = 44100;
var NF = 4096;
var HOP = 1024;
var NB = NF / 2 + 1;
var LOG = 12;
var TEST_SECONDS = 30;
var PRESETS = { 1: [0.65, 0.95], 2: [0.5, 0.9], 3: [0.3, 0.75] };
var pause = () => new Promise((r) => setTimeout(r));
var cancelado = () => new DOMException("Cancelado", "AbortError");
var ss = (x, a, b) => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};
var rev = new Uint16Array(NF);
var cosT = new Float32Array(NF / 2);
var sinT = new Float32Array(NF / 2);
for (let i = 0; i < NF; i++) {
  let r = 0;
  for (let b = 0; b < LOG; b++) if (i & 1 << b) r |= 1 << LOG - 1 - b;
  rev[i] = r;
}
for (let i = 0; i < NF / 2; i++) {
  cosT[i] = Math.cos(2 * Math.PI * i / NF);
  sinT[i] = -Math.sin(2 * Math.PI * i / NF);
}
function fft(re, im, inv) {
  for (let i = 0; i < NF; i++) {
    const j = rev[i];
    if (j > i) {
      let t = re[i];
      re[i] = re[j];
      re[j] = t;
      t = im[i];
      im[i] = im[j];
      im[j] = t;
    }
  }
  for (let size = 2; size <= NF; size <<= 1) {
    const h = size >> 1, step = NF / size;
    for (let i = 0; i < NF; i += size) for (let j = 0, k = 0; j < h; j++, k += step) {
      const wr = cosT[k], wi = inv ? -sinT[k] : sinT[k], a = i + j, b = a + h;
      const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
      re[b] = re[a] - tr;
      im[b] = im[a] - ti;
      re[a] += tr;
      im[a] += ti;
    }
  }
}
var hann = new Float32Array(NF);
for (let i = 0; i < NF; i++) hann[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / NF);
var bandG = new Float32Array(NB);
for (let k = 0; k < NB; k++) {
  const f = k * SR / NF;
  bandG[k] = ss(f, 120, 220) * (1 - ss(f, 6e3, 8e3));
}
function esMono(L, R) {
  let s = 0, m = 0;
  for (let i = 0; i < L.length; i += 3) {
    const d = L[i] - R[i], p = L[i] + R[i];
    s += d * d;
    m += p * p;
  }
  return m > 0 && s / m < 0.02;
}
async function quitarCentro(L, R, { fuerza = 2, onProg = () => {
}, signal } = {}) {
  const n = L.length, [lo, hi] = PRESETS[fuerza] || PRESETS[2], t0 = performance.now();
  const oL = new Float32Array(n), oR = new Float32Array(n);
  const re = new Float32Array(NF), im = new Float32Array(NF), prev = new Float32Array(NB).fill(1);
  const m = new Float32Array(NB), ms = new Float32Array(NB);
  const total = Math.ceil((n + NF - HOP) / HOP);
  let fr = 0;
  for (let s = -(NF - HOP); s < n; s += HOP, fr++) {
    if (signal && signal.aborted) throw cancelado();
    for (let j = 0; j < NF; j++) {
      const i = s + j;
      if (i >= 0 && i < n) {
        re[j] = L[i] * hann[j];
        im[j] = R[i] * hann[j];
      } else {
        re[j] = 0;
        im[j] = 0;
      }
    }
    fft(re, im, false);
    for (let k = 0; k < NB; k++) {
      const kk = (NF - k) % NF;
      const lr = (re[k] + re[kk]) / 2, li = (im[k] - im[kk]) / 2, rr = (im[k] + im[kk]) / 2, ri = (re[kk] - re[k]) / 2;
      const ml = Math.hypot(lr, li), mr = Math.hypot(rr, ri), mx = Math.max(ml, mr), mn = Math.min(ml, mr);
      const cosd = (lr * rr + li * ri) / (ml * mr + 1e-12);
      const sim = mx > 1e-6 ? mn / mx * Math.max(0, cosd) : 0;
      const m0 = 1 - bandG[k] * ss(sim, lo, hi) * 0.98;
      m[k] = 0.5 * m0 + 0.5 * prev[k];
      prev[k] = m[k];
    }
    for (let k = 0; k < NB; k++) {
      let a = 0, c = 0;
      for (let d = -2; d <= 2; d++) {
        const q = k + d;
        if (q >= 0 && q < NB) {
          a += m[q];
          c++;
        }
      }
      ms[k] = a / c;
    }
    for (let k = 0; k < NB; k++) {
      const kk = (NF - k) % NF, g = ms[k];
      const lr = g * (re[k] + re[kk]) / 2, li = g * (im[k] - im[kk]) / 2, rr = g * (im[k] + im[kk]) / 2, ri = g * (re[kk] - re[k]) / 2;
      re[k] = lr - ri;
      im[k] = li + rr;
      if (k > 0 && k < NF / 2) {
        re[kk] = lr + ri;
        im[kk] = -li + rr;
      }
    }
    fft(re, im, true);
    for (let j = 0; j < NF; j++) {
      const i = s + j;
      if (i >= 0 && i < n) {
        const w = hann[j] / (NF * 1.5);
        oL[i] += re[j] * w;
        oR[i] += im[j] * w;
      }
    }
    if (fr % 150 === 0) {
      onProg({ frac: fr / total });
      await pause();
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
async function decodificar(file, prueba) {
  const ab = await new OfflineAudioContext(2, SR, SR).decodeAudioData(await file.arrayBuffer());
  let ini = 0, n = ab.length;
  if (prueba) {
    n = Math.min(TEST_SECONDS * SR, ab.length);
    ini = Math.floor(Math.max(0, Math.min(ab.length * 0.3, ab.length - n)));
  }
  const L = ab.getChannelData(0).slice(ini, ini + n);
  const R = (ab.numberOfChannels > 1 ? ab.getChannelData(1) : ab.getChannelData(0)).slice(ini, ini + n);
  return { L, R };
}
async function crearInstrumental(file, { prueba = false, fuerza = 2, onEstado = () => {
}, signal } = {}) {
  onEstado({ fase: "decodificando" });
  const { L, R } = await decodificar(file, prueba);
  if (esMono(L, R)) return { mono: true };
  const out = await quitarCentro(L, R, { fuerza, signal, onProg: (p) => onEstado({ fase: "procesando", ...p }) });
  onEstado({ fase: "comprimiendo" });
  const blob = await aMp3(out.L, out.R, signal);
  return { blob, segundos: L.length / SR, separacion: out.separacion, mono: false };
}
export {
  aMp3,
  crearInstrumental,
  esMono,
  quitarCentro
};
