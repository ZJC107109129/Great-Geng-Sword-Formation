/* =====================================================================
 * 大庚剑阵 · 万剑归宗
 * 手势剑阵：双指=剑群 / 三指=散剑 / 五指=剑环 / 握拳=剑球
 * 技术：MediaPipe HandLandmarker（本地 vendor）+ Canvas 2D 伪 3D 投影
 * 视觉：玄夜、剑光青、符金；所有剑形与光晕均为程序化生成的离屏贴图
 * ===================================================================== */

/* ---------------- 可调参数 ---------------- */
const CONFIG = {
  swordCount: 300,
  streamSwordCount: 1000,  // 双指剑群形态的剑数：飞行时拉长成带，要够密才成流
  mp: {
    bundle:    "./vendor/mediapipe/vision_bundle.mjs",
    bundleAlt: "./vendor/mediapipe/vision_bundle.js",
    wasm:      "./vendor/mediapipe/wasm",
    model:     "./vendor/models/hand_landmarker.task",
  },
  camera: { width: { ideal: 1280 }, height: { ideal: 720 } },
  hands: { swap: false },
  projection: { f: 2.4 },
  gesture: {
    hold: 0.20,        // 手势需稳定多久才切换形态（秒）
    tipRatio: 1.05,    // 指尖相对指根远离手腕的倍数，判定手指伸直
    ballSpread: 0.62,  // 指尖内收程度低于此值 → 握拳
  },
  motion: {
    easePos: 6.5,
    easeDir: 8.0,
    easeAlpha: 2.6,
    easeCenter: 6.0,
  },
};

/* ---------------- 小工具 ---------------- */
const TAU = Math.PI * 2;
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const clamp01 = v => clamp(v, 0, 1);
const lerp = (a, b, k) => a + (b - a) * k;
const rgba = (c, a) => `rgba(${c[0] | 0},${c[1] | 0},${c[2] | 0},${a})`;
const rnd = i => { const x = Math.sin(i * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const smooth = v => { v = clamp01(v); return v * v * (3 - 2 * v); };

/* 最近几帧的中位数：压住 MediaPipe 的指头抖动 */
function pushHist(arr, v, max) {
  arr.push(v);
  while (arr.length > max) arr.shift();
}
function median(arr) {
  if (!arr.length) return -1;
  const s = arr.slice().sort((a, b) => a - b);
  return s[(s.length - 1) >> 1];
}

/* ---------------- 参数化入口 ---------------- */
const params = new URLSearchParams(location.search);
const PREVIEW = params.get("still") === "1" || params.has("mode");
const DEMO_START = params.get("demo") === "1";
const AUTO_SHOW = params.get("mode") === "auto";
const MODES = ["stream", "scatter", "ring", "ball"];
const MODE_FORMATION = { stream: 0.30, scatter: 0.52, ring: 1.00, ball: 0.78 };

/* ---------------- 全局状态 ---------------- */
const state = {
  t: 0,
  mode: "ring",
  cand: null,
  candAt: 0,
  input: "none",                 // "camera" | "pointer"
  center: { x: 0, y: 0 },
  targetCenter: { x: 0, y: 0 },
  R: 0,
  targetR: 0,
  axis: { x: 1, y: 0 },
  targetAxis: { x: 1, y: 0 },
  roll: 0,
  targetRoll: 0,
  formationA: 1,
  formationTarget: 1,
  hand: null,
  handSeen: { left: 0, right: 0 },
  noHandSince: 0,
  counts: [],
  spreads: [],
  flow: { pts: [], last: 0 },
  shock: [],
  flash: 0,
  dust: [],
  stars: [],
  autoAt: 0,
  demo: false,
};

/* ---------------- DOM ---------------- */
const stage   = document.getElementById("stage");
const ctx     = stage.getContext("2d");
const video   = document.getElementById("cam");
const skel    = document.getElementById("skel");
const skelCtx = skel.getContext("2d");
const el = {
  camWrap:   document.getElementById("camWrap"),
  toast:     document.getElementById("toast"),
  gate:      document.getElementById("gate"),
  startBtn:  document.getElementById("startBtn"),
  reveal:    document.getElementById("reveal"),
  blackout:  document.getElementById("blackout"),
  errDetail: document.getElementById("errDetail"),
  tagLeft:   document.getElementById("tagLeft"),
  tagRight:  document.getElementById("tagRight"),
  chips:     Array.from(document.querySelectorAll("#modeChips .chip")),
};

/* ---------------- 运行环境自检：IDE 内置预览会静默屏蔽摄像头 ---------------- */
const ENV_NOTE = (() => {
  const ua = navigator.userAgent || "";
  if (/Electron/i.test(ua)) return "当前在 IDE 内置窗口（Electron）中运行，摄像头会被宿主静默拦截；请用系统浏览器打开本页。";
  if (window.chrome && window.chrome.webview) return "当前在 IDE 内置预览（WebView2）中运行，摄像头会被宿主静默拦截；请用系统浏览器打开本页。";
  if (window.self !== window.top) return "页面被嵌在预览框里，摄像头可能被权限策略拦截；请在新窗口或系统浏览器中打开。";
  if (!window.isSecureContext) return "当前不是安全上下文，摄像头不可用；请通过 http://127.0.0.1 或 https 访问。";
  return "";
})();
(function showEnvNote() {
  const node = document.getElementById("envNote");
  if (!node || !ENV_NOTE) return;
  node.hidden = false;
  node.textContent = ENV_NOTE + "地址：" + location.origin + "/";
})();

/* ---------------- 画布与缓存 ---------------- */
let W = 0, H = 0, DPR = 1, minWH = 0;
let bgGrad = null, vigGrad = null;

function buildBackdrops() {
  bgGrad = ctx.createLinearGradient(0, 0, 0, H);
  bgGrad.addColorStop(0, "#060a12");
  bgGrad.addColorStop(0.55, "#0a1220");
  bgGrad.addColorStop(1, "#04060b");

  const cx = W * 0.5, cy = H * 0.48;
  vigGrad = ctx.createRadialGradient(
    cx, cy, Math.min(W, H) * 0.28,
    cx, cy, Math.max(W, H) * 0.82
  );
  vigGrad.addColorStop(0, "rgba(0,0,0,0)");
  vigGrad.addColorStop(0.70, "rgba(0,0,0,0.30)");
  vigGrad.addColorStop(1, "rgba(0,0,0,0.76)");
}

function buildStars() {
  state.stars.length = 0;
  const n = 96;
  for (let i = 0; i < n; i++) {
    state.stars.push({
      x: rnd(i * 3.1), y: rnd(i * 7.7) * 0.92,
      r: 0.4 + rnd(i * 11.3) * 1.2,
      ph: rnd(i * 5.9) * TAU,
      hue: rnd(i * 13.7) > 0.8 ? "gold" : "cold",
    });
  }
}

function buildDust() {
  state.dust.length = 0;
  for (let i = 0; i < 46; i++) {
    state.dust.push({
      x: rnd(i * 6.1), y: rnd(i * 9.3),
      vy: 0.012 + rnd(i * 4.7) * 0.024,
      s: 0.8 + rnd(i * 8.9) * 1.7,
      ph: rnd(i * 12.1) * TAU,
      gold: rnd(i * 15.7) > 0.72,
    });
  }
}

function resize() {
  const prevMin = minWH;
  DPR = Math.min(2, window.devicePixelRatio || 1);
  W = window.innerWidth;
  H = window.innerHeight;
  minWH = Math.min(W, H);
  stage.width = W * DPR;
  stage.height = H * DPR;
  stage.style.width = W + "px";
  stage.style.height = H + "px";
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.imageSmoothingQuality = "high";
  buildBackdrops();
  if (!state.center.x) {
    state.center.x = state.targetCenter.x = W * 0.5;
    state.center.y = state.targetCenter.y = H * 0.48;
    state.R = state.targetR = minWH * 0.33;
  } else if (prevMin > 0) {
    // 星屑/尘光都是归一化坐标，无需重建；只让阵径随短边等比缩放
    const k = minWH / prevMin;
    state.R *= k;
    state.targetR *= k;
  }
}
window.addEventListener("resize", () => { resize(); clampCamWrap(); });
resize();
buildStars();
buildDust();

/* =====================================================================
 * 离屏剑形贴图：剑身 / 光尾 / 光点
 * =================================================================== */
function createCanvas(w, h) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  return c;
}

/* 剑身配色：jade=荧绿主体，gold=流金，ember=焰红点缀（复刻参考视频的绿底金焰星点） */
const BLADE_PAL = {
  jade:  { col: "124,255,96",  soft: "222,255,190", glow: "92,236,60" },
  gold:  { col: "255,214,118", soft: "255,240,200", glow: "255,192,86" },
  ember: { col: "255,148,84",  soft: "255,216,170", glow: "255,110,44" },
};

/* 光梭贴图：头白亮、尾渐隐的荧光短剑（视频里的“剑”就是这种细小光条） */
function makeBlade(key) {
  const p = BLADE_PAL[key] || BLADE_PAL.jade;
  const w = 64, h = 420;
  const c = createCanvas(w, h);
  const g = c.getContext("2d");
  const cx = w / 2, tip = 14, tail = h - 10;
  const col = p.col;
  const colSoft = p.soft;
  const hw = 9;                        // 光梭最宽处

  // 外层剑光：一次烘焙，避免每帧 shadowBlur
  g.save();
  g.shadowColor = `rgba(${p.glow},0.9)`;
  g.shadowBlur = 16;
  const body = g.createLinearGradient(0, tip, 0, tail);
  body.addColorStop(0, "rgba(255,255,255,0.98)");
  body.addColorStop(0.10, `rgba(${colSoft},0.95)`);
  body.addColorStop(0.38, `rgba(${col},0.88)`);
  body.addColorStop(0.75, `rgba(${col},0.40)`);
  body.addColorStop(1, `rgba(${col},0)`);
  g.fillStyle = body;
  g.beginPath();
  g.moveTo(cx, tip);
  g.quadraticCurveTo(cx + hw, h * 0.30, cx + hw * 0.62, h * 0.66);
  g.quadraticCurveTo(cx + hw * 0.18, h * 0.90, cx, tail);
  g.quadraticCurveTo(cx - hw * 0.18, h * 0.90, cx - hw * 0.62, h * 0.66);
  g.quadraticCurveTo(cx - hw, h * 0.30, cx, tip);
  g.closePath();
  g.fill();
  g.restore();

  // 白亮剑芯：头亮尾隐，读作“光”而不是“铁”
  const core = g.createLinearGradient(0, tip, 0, tail);
  core.addColorStop(0, "rgba(255,255,255,1)");
  core.addColorStop(0.30, "rgba(255,255,255,0.85)");
  core.addColorStop(0.68, "rgba(255,255,255,0.22)");
  core.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = core;
  g.beginPath();
  g.moveTo(cx, tip);
  g.quadraticCurveTo(cx + 2.6, h * 0.30, cx + 1.6, h * 0.62);
  g.lineTo(cx, tail);
  g.lineTo(cx - 1.6, h * 0.62);
  g.quadraticCurveTo(cx - 2.6, h * 0.30, cx, tip);
  g.closePath();
  g.fill();

  // 剑尖一点寒芒
  const flare = g.createRadialGradient(cx, tip + 14, 0, cx, tip + 14, 40);
  flare.addColorStop(0, "rgba(255,255,255,0.9)");
  flare.addColorStop(0.30, `rgba(${colSoft},0.34)`);
  flare.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = flare;
  g.fillRect(0, 0, w, 92);
  return c;
}

function makeStreak(key) {
  const p = BLADE_PAL[key] || BLADE_PAL.jade;
  const w = 40, h = 320;
  const c = createCanvas(w, h);
  const g = c.getContext("2d");
  const col = p.col;
  g.save();
  g.shadowColor = `rgba(${col},0.9)`;
  g.shadowBlur = 10;
  const grd = g.createLinearGradient(0, 0, 0, h);
  grd.addColorStop(0, `rgba(${col},0)`);
  grd.addColorStop(0.45, `rgba(${col},0.52)`);
  grd.addColorStop(0.62, `rgba(255,255,255,0.42)`);
  grd.addColorStop(1, `rgba(${col},0)`);
  g.fillStyle = grd;
  g.fillRect(w / 2 - 1.8, 0, 3.6, h);
  g.restore();

  const halo = g.createLinearGradient(0, 0, 0, h);
  halo.addColorStop(0, `rgba(${col},0)`);
  halo.addColorStop(0.5, `rgba(${col},0.22)`);
  halo.addColorStop(1, `rgba(${col},0)`);
  g.fillStyle = halo;
  g.fillRect(w / 2 - 8, 0, 16, h);
  return c;
}

function makeGlowDot(inner, mid, outer) {
  const s = 128;
  const c = createCanvas(s, s);
  const g = c.getContext("2d");
  const r = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  r.addColorStop(0, "rgba(255,255,255,0.95)");
  r.addColorStop(0.18, `rgba(${inner || "222,255,190"},0.62)`);
  r.addColorStop(0.45, `rgba(${mid || "120,235,90"},0.18)`);
  r.addColorStop(1, `rgba(${outer || "40,180,80"},0)`);
  g.fillStyle = r;
  g.fillRect(0, 0, s, s);
  return c;
}

const SPR = {
  blade: makeBlade("jade"),
  bladeGold: makeBlade("gold"),
  bladeEmber: makeBlade("ember"),
  streak: makeStreak("jade"),
  streakGold: makeStreak("gold"),
  streakEmber: makeStreak("ember"),
  dot: makeGlowDot(),
};
const BLADE_SPR = [SPR.blade, SPR.bladeGold, SPR.bladeEmber];
const STREAK_SPR = [SPR.streak, SPR.streakGold, SPR.streakEmber];

/* ---------------- 纸纹 / 颗粒 ---------------- */
let grainPat = null;
(function makeGrain() {
  const g = createCanvas(128, 128);
  const gc = g.getContext("2d");
  const id = gc.createImageData(128, 128);
  for (let i = 0; i < id.data.length; i += 4) {
    const v = (Math.random() * 255) | 0;
    id.data[i] = id.data[i + 1] = id.data[i + 2] = v;
    id.data[i + 3] = 255;
  }
  gc.putImageData(id, 0, 0);
  grainPat = ctx.createPattern(g, "repeat");
})();

/* =====================================================================
 * 剑阵：三维分布 + 二维投影
 * =================================================================== */
const RINGS = [
  { n: 96, r: 1.00, z: -0.10, spd:  0.50, tilt: 0.34 },
  { n: 72, r: 0.74, z:  0.10, spd: -0.72, tilt: 0.52 },
  { n: 48, r: 0.44, z:  0.26, spd:  1.00, tilt: 0.20 },
];
const N = Math.max(CONFIG.swordCount, CONFIG.streamSwordCount);
const swords = [];

(function buildSwords() {
  const ringSlots = [];
  RINGS.forEach((rg, gi) => {
    for (let k = 0; k < rg.n; k++) ringSlots.push({ gi, k, n: rg.n });
  });
  for (let i = 0; i < N; i++) {
    const s = {
      seed: rnd(i * 3.77),
      phase: rnd(i * 91.37),
      helix: i / N + rnd(i * 13.7) * 0.02,
      // tint：0=荧绿主体，1=流金，2=焰红点缀（参考视频里绿群中星点的金红剑）
      tint: i % 11 === 3 ? 1 : (i % 11 === 7 ? 2 : 0),
      ring: ringSlots[i % ringSlots.length],
      fib: { x: 0, y: 0, z: 0 },
      rdir: { x: 0, y: 0, z: 0 },
      x: 0, y: 0, z: 0,
      dx: 0, dy: 1, dz: 0,
      a: 0, size: 1,
      sx: 0, sy: 0, px: null, py: null,
      prevF: undefined,
    };
    // 球面 Fibonacci 分布（剑球形态只活跃前 swordCount 把，按它均匀布满全球面）
    const nb = CONFIG.swordCount;
    const yv = 1 - 2 * ((i % nb) + 0.5) / nb;
    const rr = Math.sqrt(Math.max(0, 1 - yv * yv));
    const th = (i % nb) * 2.399963;
    s.fib.x = Math.cos(th) * rr;
    s.fib.y = yv;
    s.fib.z = Math.sin(th) * rr;
    // 散剑用的随机方向
    const t1 = rnd(i * 5.11) * TAU;
    const cz = rnd(i * 7.19) * 2 - 1;
    const sz = Math.sqrt(Math.max(0, 1 - cz * cz));
    s.rdir.x = Math.cos(t1) * sz;
    s.rdir.y = cz;
    s.rdir.z = Math.sin(t1) * sz;
    // 初始位置在阵心附近，开场时从虚空中聚拢
    s.x = s.fib.x * 0.18;
    s.y = s.fib.y * 0.18;
    s.z = s.fib.z * 0.18;
    swords.push(s);
  }
})();

/* 目标槽（避免每帧新建对象） */
const T = { x: 0, y: 0, z: 0, dx: 0, dy: 1, dz: 0, alpha: 0, size: 1, snap: false };

/* ---- 剑环：中空涡旋盘。剑贴盘切向差速旋转，内快外慢绞出涡纹（参考视频的五指大盘） ---- */
function targetRing(s, t) {
  // 连续中空盘面：内缘 0.34、外缘 1.0，sqrt 分布让盘面密度均匀
  const rr = 0.34 + 0.66 * Math.sqrt(s.seed);
  const a0 = (s.ring.k / RINGS[s.ring.gi].n) * TAU + rr * 4.8;   // 螺旋相位 → 涡纹
  const a = a0 + t * (0.55 + 0.85 * (1 - rr));     // 差速：内快外慢
  const tilt = 0.18;                                // 盘面近乎正对镜头
  T.x = Math.cos(a) * rr;
  T.y = Math.sin(a) * rr * Math.cos(tilt) + 0.018 * Math.sin(t * 1.1 + s.seed * 4);
  T.z = Math.sin(a) * rr * Math.sin(tilt);
  // 切向为主、略向外张：既绕盘而行，又让剑锋微朝外
  let ox = -Math.sin(a) + Math.cos(a) * 0.30;
  let oy = Math.cos(a) * Math.cos(tilt) + Math.sin(a) * Math.cos(tilt) * 0.30;
  let oz = Math.cos(a) * Math.sin(tilt) + 0.02;
  const ol = Math.hypot(ox, oy, oz) || 1;
  T.dx = ox / ol; T.dy = oy / ol; T.dz = oz / ol;
  T.alpha = 1;
  T.size = 0.86 + 0.30 * rnd(s.seed * 97);
  T.snap = false;
}

/* ---- 剑球：球面涡旋。剑贴着球面沿纬线切向流动，绞成毛线球（参考视频的握拳剑球） ---- */
function targetBall(s, t) {
  const p = s.fib;
  const cy = Math.cos(t * 0.5), sy = Math.sin(t * 0.5);
  const x0 = p.x * cy + p.z * sy;
  const z0 = -p.x * sy + p.z * cy;
  const ca = Math.cos(0.14), sa = Math.sin(0.14);
  const y2 = p.y * ca - z0 * sa;
  const z2 = p.y * sa + z0 * ca;
  const rad = 0.72 + 0.022 * Math.sin(t * 1.1 + s.seed * 5);
  T.x = x0 * rad; T.y = y2 * rad; T.z = z2 * rad;
  // 纬线切向（绕倾斜自转轴）+ 少量朝外分量：流动绵密、极点不塌
  const lat = Math.sqrt(Math.max(0, 1 - y2 * y2)) || 1e-4;
  let dx = -z2 / lat + x0 * 0.22;
  let dy = -0.22 + 0.10 * Math.sin(s.seed * 9);
  let dz = x0 / lat + z2 * 0.22;
  const l = Math.hypot(dx, dy, dz) || 1;
  T.dx = dx / l; T.dy = dy / l; T.dz = dz / l;
  T.alpha = 1;
  T.size = 0.82 + 0.30 * rnd(s.seed * 31);
  T.snap = false;
}

/* ---- 剑流：双指形态的龙卷基底。静止=指尖上的剑轮自转，移动=能量注入 ---- */
const FLOW = {
  ox: 0, oy: 0,      // 云团头锚点（屏幕坐标，含一次识别周期的速度外推）
  ax: 1, ay: 0,      // 指锋方向（屏幕单位向量）
  dx: 1, dy: 0,      // 运动方向（屏幕单位向量，缓动过）：拉彗尾、顺剑锋用
  speedK: 0,         // 0..1：0 = 静止云球，1 = 全速剑带
  init: false,       // 锚点是否已初始化（首帧直接落位，之后缓动）
};

function buildFlow(t, dt) {
  const src = state.flow.pts;
  const R = Math.max(state.R, 1);
  const ax = state.axis.x, ay = state.axis.y;

  // 1) 锚点目标：最新指尖位置 + 按识别周期外推（30fps 手势靠外推补成 60fps 连续）。
  //    没有新鲜轨迹时锚点原地悬浮，云团不会跳回阵心
  let tx = FLOW.ox, ty = FLOW.oy;
  if (src.length) {
    const p0 = src[0];
    tx = p0.x; ty = p0.y;
    if (src.length >= 2) {
      const p1 = src[1];
      const dt0 = (p0.ts || 0) - (p1.ts || 0);
      const ahead = clamp((state.t - (p0.ts || 0)) / Math.max(dt0, 0.016), 0, 0.55);
      if (ahead > 0.02) {
        tx += (p0.x - p1.x) * ahead;
        ty += (p0.y - p1.y) * ahead;
      }
    }
    // 锚点略微越过指尖，让剑云看起来是“托”在指锋上
    tx += ax * R * 0.06;
    ty += ay * R * 0.06;
  } else if (!FLOW.init) {
    tx = state.center.x + ax * R * 0.16;
    ty = state.center.y + ay * R * 0.16;
  }
  // 锚点自身缓动：手出现/消失/换位时云团是“流”过去的，不是跳过去
  if (!FLOW.init) {
    FLOW.ox = tx; FLOW.oy = ty; FLOW.init = true;
  } else {
    const kA = 1 - Math.exp(-dt * 10);
    FLOW.ox += (tx - FLOW.ox) * kA;
    FLOW.oy += (ty - FLOW.oy) * kA;
  }

  // 2) 移动方向与速度：取一个稍旧的轨迹点做基线，压住采样噪声。
  //    速度在剑群形态里只作“能量”：拖尾越长、旋速、亮度随之升高
  let spd = 0;
  if (src.length >= 3) {
    const p0 = src[0];
    const ref = src.find(p => (p0.ts || 0) - (p.ts || 0) > 0.06) || src[src.length - 1];
    const dts = (p0.ts || 0) - (ref.ts || 0);
    const dx = p0.x - ref.x, dy = p0.y - ref.y;
    const d = Math.hypot(dx, dy);
    if (dts > 0.008) spd = d / dts / Math.max(1, minWH);
    // 指针停住后不再有新事件，历史速度会“冻”在最后一刻；按采样年龄衰减回 0
    const idle = state.t - (p0.ts || 0);
    if (idle > 0.10) spd *= clamp01(1 - (idle - 0.10) / 0.22);
    if (d > 3 && idle < 0.30) {
      const k = 1 - Math.exp(-dt * 10);
      FLOW.dx += (dx / d - FLOW.dx) * k;
      FLOW.dy += (dy / d - FLOW.dy) * k;
    }
  }
  const ml = Math.hypot(FLOW.dx, FLOW.dy) || 1;
  FLOW.dx /= ml; FLOW.dy /= ml;

  // 速度 → 剑群能量：动得快尾拉得长、转得快、亮得猛
  const target = clamp01(spd / 0.62);
  FLOW.speedK += (target - FLOW.speedK) * (1 - Math.exp(-dt * (target > FLOW.speedK ? 12 : 6)));

  FLOW.ax = ax; FLOW.ay = ay;
}

/* 取轨迹在时刻 tQ 的位置（pts 新在前；越界夹到端点）。供剑群做时差锚点 */
function sampleTrail(pts, tQ) {
  if (!pts.length) return null;
  if (tQ >= (pts[0].ts || 0)) return pts[0];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (tQ >= (b.ts || 0)) {
      const span = (a.ts || 0) - (b.ts || 0);
      const f = span > 1e-4 ? (tQ - (b.ts || 0)) / span : 0;
      return { x: lerp(b.x, a.x, f), y: lerp(b.y, a.y, f) };
    }
  }
  return pts[pts.length - 1];
}

/* 为一把剑掷一次剑群参数（生命周期到点后重掷，产生细碎的闪光感） */
function seedFlowSword(s, initial) {
  s.gen = (s.gen | 0) + 1;
  const g = s.gen;
  s.life = 0.70 + 1.10 * rnd(s.seed * 57.1 + g * 13.7);
  s.spin = rnd(s.seed * 11.9 + g * 7.3) * TAU;
  s.lag = rnd(s.seed * 71.3 + g * 3.7);                       // 时差锚点：0=云头 1=云尾
  // 涡盘归一化半径：幂分布让盘心密、盘缘疏（涡旋星系的盘面密度）
  s.discR = Math.pow(rnd(s.seed * 23.9 + g * 5.1), 0.62);
  s.discA = rnd(s.seed * 29.3 + g * 8.9) * TAU;               // 盘内初始角
  s.radK = 0.88 + 0.36 * rnd(s.seed * 31.7 + g * 2.2);        // 离轴距离抖散 → 盘/带的蓬松度
  s.sizeBase = 0.70 + 0.45 * rnd(s.seed * 17.3 + g * 9.1);    // 小飞剑，密才像群
  s.jx = (rnd(s.seed * 41.3 + g * 1.7) - 0.5) * minWH * 0.016;  // 每把剑的固定偏移
  s.jy = (rnd(s.seed * 43.9 + g * 4.1) - 0.5) * minWH * 0.016;
  // 首次进入时按随机年龄错开，一显形就是完整的一团剑云，而不是从指尖冒出来
  s.age = initial ? rnd(s.seed * 61.7) * s.life : 0;
}

/* 双指形态：悬在指尖上方的荧绿剑群。
 * 静止：涡旋星盘 —— 剑贴着盘面沿切线差速旋转，内快外慢绞出涡纹（参考视频的旋涡盘）；
 * 移动：每把剑按各自的“时差”锚定在指尖轨迹上 → 盘被拉成波浪剑带，
 *       带内剑随波起伏、离散成鸟群；尾部剑顺着流向摆正；停手后收拢回涡盘。 */
function targetStream(s, t, dt) {
  if (!s.gen) seedFlowSword(s, true);

  s.age += dt;
  let snapped = false;
  if (s.age >= s.life) {
    seedFlowSword(s, false);
    s.a = 0;                               // 重生瞬间压暗，不叠出一个亮斑
    snapped = true;
  }

  const u = clamp01(s.age / s.life);
  const sk = FLOW.speedK;                  // 0..1 移动能量
  const R = Math.max(state.R, 1);
  const F = CONFIG.projection.f;
  const cx = state.center.x, cy = state.center.y;

  // 时差锚点：lag 越大锚得越靠后，移动时涡盘沿轨迹拉长成剑带
  const delay = s.lag * (0.10 + 0.70 * sk);
  const anchor = sampleTrail(state.flow.pts, state.t - delay) || FLOW;
  const bx = anchor.x + s.jx;
  const by = anchor.y + s.jy;

  // 涡盘几何：盘面极坐标，差速旋转（内快外慢）绞出涡纹
  const dr = s.discR;
  const a = s.discA + t * (0.9 + 2.1 * (1 - dr)) * (1 + 0.55 * sk) + s.spin * 0.05;
  // 飞行时盘面收拢成带；静止时是舒展的圆盘
  const spread = (1 - 0.58 * sk) * (1 + 0.05 * Math.sin(t * 2.1 + s.spin));
  const r = R * (0.10 + 0.46 * dr) * s.radK * spread;
  const ey = 0.94;                          // 盘面近乎正对镜头，略带透视
  // 盘内翻涌的细碎抖动：飞行时收拢，剑带更致密成流
  const churn = R * 0.014 * (0.6 + s.lag) * (1 - 0.30 * sk);
  // 波浪剑带：飞行时沿带起伏，复刻视频里绸带般的波纹
  const wave = R * 0.11 * sk;
  const wx = Math.sin(s.lag * 9.4 + t * 3.1 + s.spin) * wave;
  const wy = Math.sin(s.lag * 7.1 - t * 2.4 + s.spin * 2) * wave;
  const px = bx + Math.cos(a) * r + Math.sin(t * 3.1 + s.spin * 5) * churn + wx;
  const py = by - R * 0.12 + Math.sin(a) * r * ey + Math.cos(t * 2.7 + s.spin * 4) * churn + wy;

  const z = 0.10 + 0.12 * (s.helix - 0.5) + 0.06 * Math.sin(a * 2 + t) + 0.05 * Math.cos(t * 2 + s.spin);
  const sc = F / (F - z);
  T.x = (px - cx) / (R * sc);
  T.y = (py - cy) / (R * sc);
  T.z = z;

  // 剑锋：贴盘切向绕旋 + 微仰；移动时整体渐顺流向，尾部剑拉得最直
  const sw = 0.14 * Math.sin(u * 5.0 + s.spin * 2 - t * 2.6);
  let tx2 = -Math.sin(a) * ey;
  let ty2 = Math.cos(a);
  const radial = 0.20 * (1 - sk);           // 静止时略向外张，涡感更强
  tx2 += Math.cos(a) * radial;
  ty2 += Math.sin(a) * radial * ey;
  const mo = clamp01(sk * (0.30 + s.lag) * 1.8);
  if (mo > 0.01) {
    tx2 = lerp(tx2, FLOW.dx, mo);
    ty2 = lerp(ty2, FLOW.dy, mo);
  }
  const dxs = tx2 - ty2 * sw;
  const dys = ty2 + tx2 * sw;
  const dzs = 0.10 * Math.sin(u * 4.0 + s.spin);
  const dl2 = Math.hypot(dxs, dys, dzs) || 1;
  T.dx = dxs / dl2; T.dy = dys / dl2; T.dz = dzs / dl2;

  // 慢淡入、长尾淡出：整团剑云始终在闪，重生时才明显压暗
  const fadeIn = smooth(clamp01(u / 0.12));
  const fadeOut = 1 - 0.82 * smooth(clamp01((u - 0.55) / 0.45));
  const flick = 0.86 + 0.18 * Math.sin(u * 9.0 + s.spin * 3.0 - t * 4.0);
  T.alpha = clamp01(fadeIn * fadeOut * flick * (0.92 + 0.08 * sk) * (0.94 + 0.20 * rnd(s.seed * 19.3)));
  T.size = s.sizeBase * (1 - 0.10 * u) * (1 + 0.10 * sk);
  T.snap = snapped;
}

/* ---- 散剑：向外爆散，边旋边散，回收于阵心 ---- */
function targetScatter(s, t) {
  const f = ((t * 0.38 + s.phase) % 1 + 1) % 1;
  const rad = 0.34 + 1.30 * f;
  const ca = Math.cos(t * 0.25), sa = Math.sin(t * 0.25);
  const rx = s.rdir.x * ca + s.rdir.z * sa;
  const ry = s.rdir.y;
  const rz = -s.rdir.x * sa + s.rdir.z * ca;
  T.x = rx * rad; T.y = ry * rad; T.z = rz * rad;
  T.dx = rx; T.dy = ry; T.dz = rz;
  T.alpha = 0.26 + 0.74 * Math.sin(Math.PI * f);
  T.size = 0.78 + 0.36 * f;
  T.snap = s.prevF !== undefined && f < s.prevF - 0.5;
  s.prevF = f;
}

function updateSwords(dt, t) {
  if (state.mode === "stream") buildFlow(t, dt);
  // 剑流：目标点本身就在漩涡上平滑移动，这里留一点惯性才有甩尾的拖拽感
  const snappy = state.mode === "stream";
  const kp = 1 - Math.exp(-dt * CONFIG.motion.easePos * (snappy ? 1.45 : 1));
  const kd = 1 - Math.exp(-dt * CONFIG.motion.easeDir * (snappy ? 2.2 : 1));
  const ka = 1 - Math.exp(-dt * CONFIG.motion.easeAlpha);
  // 龙卷专属的额外剑：其它形态下原地淡出，退场不跳变
  const extraOff = state.mode !== "stream";
  for (let i = 0; i < swords.length; i++) {
    const s = swords[i];
    if (extraOff && i >= CONFIG.swordCount) {
      T.x = s.x; T.y = s.y; T.z = s.z;
      T.dx = s.dx; T.dy = s.dy; T.dz = s.dz;
      T.alpha = 0; T.size = s.size; T.snap = false;
    } else if (state.mode === "stream") targetStream(s, t, dt);
    else if (state.mode === "ball") targetBall(s, t);
    else if (state.mode === "scatter") targetScatter(s, t);
    else targetRing(s, t);

    if (T.snap) {
      s.x = T.x; s.y = T.y; s.z = T.z;
      s.dx = T.dx; s.dy = T.dy; s.dz = T.dz;
      s.px = null; s.py = null;              // 重生瞬移不画光尾，避免拉出一道假残影
    } else {
      s.x += (T.x - s.x) * kp;
      s.y += (T.y - s.y) * kp;
      s.z += (T.z - s.z) * kp;
      const l = Math.hypot(T.dx, T.dy, T.dz) || 1;
      s.dx += (T.dx / l - s.dx) * kd;
      s.dy += (T.dy / l - s.dy) * kd;
      s.dz += (T.dz / l - s.dz) * kd;
    }
    s.a += (T.alpha - s.a) * ka;
    s.size += (T.size - s.size) * kp;

    const dl = Math.hypot(s.dx, s.dy, s.dz) || 1;
    s.dx /= dl; s.dy /= dl; s.dz /= dl;
  }
}

/* =====================================================================
 * 手势识别：MediaPipe HandLandmarker
 * =================================================================== */
let landmarker = null, camReady = false, lastDetect = 0, camStream = null;
let visionLib = null, visionFileset = null;

const HAND_LINKS = [
  [0,1],[1,2],[2,3],[3,4], [0,5],[5,6],[6,7],[7,8], [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16], [13,17],[17,18],[18,19],[19,20], [0,17],
];

async function loadVisionLibrary() {
  const candidates = [CONFIG.mp.bundle, CONFIG.mp.bundleAlt].filter(Boolean);
  const problems = [];
  let lib = null;
  for (const url of candidates) {
    try {
      lib = await import(/* @vite-ignore */ url);
      break;
    } catch (e) {
      const line = `import("${url}") → ${e.name}: ${e.message}`;
      problems.push(line);
      console.error("[手势库]", line, e);
    }
  }
  if (!lib) {
    state.libraryError = problems.join(" ｜ ");
    showErrorDetail(state.libraryError);
    return false;
  }
  if (!lib.FilesetResolver) {
    state.libraryError = "模块已加载，但缺少 FilesetResolver 导出（bundle 入口可能有误）";
    showErrorDetail(state.libraryError);
    return false;
  }
  visionLib = lib;
  try {
    visionFileset = await lib.FilesetResolver.forVisionTasks(CONFIG.mp.wasm);
  } catch (e) {
    visionLib = null; visionFileset = null;
    state.libraryError = `FilesetResolver.forVisionTasks("${CONFIG.mp.wasm}") → ${e.name}: ${e.message}`;
    console.error("[手势库 wasm]", state.libraryError, e);
    showErrorDetail(state.libraryError);
    return false;
  }
  hideErrorDetail();
  return true;
}

async function loadHandModel() {
  try {
    if (!visionLib || !visionFileset) return false;
    landmarker = await visionLib.HandLandmarker.createFromOptions(visionFileset, {
      baseOptions: { modelAssetPath: CONFIG.mp.model, delegate: "GPU" },
      runningMode: "VIDEO",
      numHands: 2,
    });
    hideErrorDetail();
    return true;
  } catch (e) {
    const line = `HandLandmarker.createFromOptions("${CONFIG.mp.model}") → ${e.name}: ${e.message}`;
    state.libraryError = line;
    showErrorDetail(line);
    console.error("[模型]", line, e);
    landmarker = null;
    return false;
  }
}

async function initCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return "unsupported";
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: CONFIG.camera.width, height: CONFIG.camera.height, facingMode: "user" },
        audio: false,
      });
    } catch (e) {
      // 虚拟摄像头/老设备不接受 ideal 尺寸约束时，退回最宽松的请求
      if (e && (e.name === "OverconstrainedError" || e.name === "NotFoundError")) {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      } else throw e;
    }
    video.srcObject = stream;
    camStream = stream;
    // play() 在个别设备上可能一直不 resolve，不能让它卡住整条流程
    await Promise.race([
      video.play().catch(() => {}),
      new Promise(r => setTimeout(r, 1200)),
    ]);
    camReady = true;
    // 之前若走过降级流程，这里要恢复视频与小窗显示
    video.style.display = "";
    skel.style.display = "";
    el.camWrap.classList.remove("hidden");
    clampCamWrap();
    return "ok";
  } catch (e) {
    console.warn("摄像头不可用：", e);
    const n = e && e.name;
    if (n === "NotAllowedError" || n === "SecurityError") return "denied";
    if (n === "NotFoundError" || n === "OverconstrainedError") return "notfound";
    return "error";
  }
}

function stopCamera() {
  if (camStream) { camStream.getTracks().forEach(t => t.stop()); camStream = null; }
  video.srcObject = null;
  camReady = false;
  state.hand = null;
  if (el.camWrap) el.camWrap.classList.add("hidden");
}

async function resumeCamera() {
  if (state.input !== "camera" || document.hidden) return;
  if (!camReady) {
    showToast("正在重新请求摄像头…", 0);
    const r = await initCamera();
    if (r !== "ok") { showToast(camFailMessage(r).text, 5200); return; }
  }
  if (!landmarker) {
    // 之前模型没加载成功（或授权超时被降级），这次一并补上
    showToast("正在加载手势模型…", 0);
    if (!visionFileset && !(await withTimeout(loadVisionLibrary(), 20000))) {
      stopCamera(); showToast("手势库加载失败 · 见下方诊断", 5200); return;
    }
    if (!(await withTimeout(loadHandModel(), 30000))) {
      stopCamera(); showToast("模型加载失败 · 请检查 vendor/models", 5200); return;
    }
  }
  showToast("把手伸进画面", 4200);
  hideErrorDetail();
}

function withTimeout(promise, ms) {
  return Promise.race([promise, new Promise(res => setTimeout(() => res(false), ms))]);
}

const camFailMessage = reason => ({
  denied:      { text: "摄像头不可用/被拒绝", tip: "请在浏览器地址栏左侧的摄像头图标里改为「允许」，然后刷新页面。" },
  notfound:    { text: "摄像头不可用 · 未找到设备", tip: "请确认摄像头已连接且未被其他程序独占，然后重试。" },
  unsupported: { text: "摄像头不可用 · 当前环境不支持", tip: "请用 localhost 或 HTTPS 访问，并换用新版 Chrome。" },
  error:       { text: "摄像头不可用 · 取流失败", tip: "请关闭占用摄像头的其他程序后按 C 重新开启。" },
}[reason] || { text: "摄像头不可用", tip: "请检查设备与浏览器权限后重试。" });

/* 屏幕坐标：摄像头画面左右镜像 */
const toScreen = p => ({ x: (1 - p.x) * W, y: p.y * H });

/* 含深度的距离：指尖朝向镜头（正反手都常见）时，2D 投影会压缩，只有算上 z 才数得准 */
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, (a.z - b.z) * 0.9);

function countFingers(lm) {
  const wrist = lm[0];
  let n = 0;
  for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) {
    if (d3(lm[tip], wrist) > d3(lm[pip], wrist) * CONFIG.gesture.tipRatio) n++;
  }
  return n;
}

function spreadOf(lm, handSize) {
  let sum = 0;
  for (const tip of [8, 12, 16, 20]) sum += dist(lm[tip], lm[9]);
  return (sum / 4) / (handSize + 1e-6);
}

function analyzeHand(lm, now, label) {
  const wrist = lm[0];
  const mcp = lm[9];
  const handSize = dist(wrist, mcp) + 1e-6;
  const pos = toScreen(mcp);
  const wpt = toScreen(wrist);
  let ax = pos.x - wpt.x, ay = pos.y - wpt.y;
  const al = Math.hypot(ax, ay) || 1;
  ax /= al; ay /= al;

  const a = toScreen(lm[5]), b = toScreen(lm[17]);
  const roll = Math.atan2(b.y - a.y, b.x - a.x);

  const tipsS = [4, 8, 12, 16, 20].map(k => toScreen(lm[k]));

  // 双指操控专用：食指+中指指尖中点、指根→指尖的朝向、指长
  // 只取手指上的点，不掺掌部位置；正手/反手、指尖朝向镜头都按同一套算
  const fpos = { x: (tipsS[1].x + tipsS[2].x) * 0.5, y: (tipsS[1].y + tipsS[2].y) * 0.5 };
  const p5 = toScreen(lm[5]), p9 = toScreen(lm[9]);
  const fbase = { x: (p5.x + p9.x) * 0.5, y: (p5.y + p9.y) * 0.5 };
  let fx = fpos.x - fbase.x, fy = fpos.y - fbase.y;
  const fl = Math.hypot(fx, fy);
  if (fl < Math.min(W, H) * 0.05) {
    // 指尖正对镜头：2D 投影被压扁，方向不可信，退回手掌轴向
    fx = ax; fy = ay;
  } else {
    fx /= fl; fy /= fl;
  }

  return {
    pos, axis: { x: ax, y: ay }, roll,
    count: countFingers(lm),
    spread: spreadOf(lm, handSize),
    size: handSize,
    tips: tipsS,
    fingerPos: fpos,
    fingerAxis: { x: fx, y: fy },
    fingerLen: dist(wpt, fpos) + 1e-6,
    t: now, label,
  };
}

function modeFromHand(h, count, spread) {
  const c = count === undefined ? h.count : count;
  const sp = spread === undefined ? h.spread : spread;
  if (c === 0 || (c <= 1 && sp < CONFIG.gesture.ballSpread)) return "ball";
  if (c <= 2) return "stream";
  if (c === 3) return "scatter";
  return "ring";
}

function drawSkeleton(lm, color) {
  const w = skel.width, h = skel.height;
  skelCtx.strokeStyle = color; skelCtx.fillStyle = color;
  skelCtx.lineWidth = 4; skelCtx.lineCap = "round"; skelCtx.globalAlpha = 0.9;
  for (const [a, b] of HAND_LINKS) {
    skelCtx.beginPath();
    skelCtx.moveTo((1 - lm[a].x) * w, lm[a].y * h);
    skelCtx.lineTo((1 - lm[b].x) * w, lm[b].y * h);
    skelCtx.stroke();
  }
  for (const p of lm) {
    skelCtx.beginPath();
    skelCtx.arc((1 - p.x) * w, p.y * h, 6, 0, TAU);
    skelCtx.fill();
  }
  skelCtx.globalAlpha = 1;
}

function detect(now) {
  // 30fps 推理足够：手指判定还有中位数与缓动平滑，不必每帧都跑模型
  if (!landmarker || !camReady || video.readyState < 2 || now - lastDetect < 33) return;
  lastDetect = now;
  let res;
  try { res = landmarker.detectForVideo(video, now); } catch (e) { return; }

  skelCtx.clearRect(0, 0, skel.width, skel.height);
  const lms = res.landmarks || [], hds = res.handedness || [];
  const labelOf = i => {
    let l = hds[i] && hds[i][0] ? hds[i][0].categoryName : "Right";
    if (CONFIG.hands.swap) l = l === "Left" ? "Right" : "Left";
    return l;
  };

  let chosen = null;
  for (let i = 0; i < lms.length; i++) {
    const lm = lms[i];
    const label = labelOf(i);
    const hand = analyzeHand(lm, now, label);
    if (label === "Left") {
      state.handSeen.left = now;
      drawSkeleton(lm, "#63d8ff");
    } else {
      state.handSeen.right = now;
      drawSkeleton(lm, "#ffd58a");
      chosen = hand;               // 双手时以右手为主
    }
    if (!chosen) chosen = hand;
  }
  state.hand = chosen;
}

/* =====================================================================
 * 手势 → 形态：稳定 0.2s 才切换
 * =================================================================== */
function updateGesture(dt, now) {
  const h = state.hand;
  const fresh = h && now - h.t < 400;

  if (fresh) {
    state.noHandSince = 0;
    // 指头抖动很大：用最近几帧的中位数做判定，否则 2/3/4 来回跳会让形态永远切不过去
    pushHist(state.counts, h.count, 5);
    pushHist(state.spreads, h.spread, 5);
    const cand = modeFromHand(h, median(state.counts), median(state.spreads));
    if (state.cand !== cand) {
      state.cand = cand;
      state.candAt = state.t;
    } else if (state.t - state.candAt > (cand === "stream" ? 0.12 : CONFIG.gesture.hold) && state.mode !== cand) {
      setMode(cand);
    }
    // 双指（剑流）时完全由手指驱动：阵心=两指尖中点，主向=指根→指尖
    // 不再跟随掌部位置/掌宽，手腕原地转动或掌部晃动都不会拖慢剑流
    const fingerMode = state.mode === "stream" || cand === "stream";
    if (fingerMode) {
      state.targetCenter.x = h.fingerPos.x;
      state.targetCenter.y = h.fingerPos.y;
      state.targetAxis.x = h.fingerAxis.x;
      state.targetAxis.y = h.fingerAxis.y;
      state.targetR = clamp(h.fingerLen * minWH * 1.15, minWH * 0.17, minWH * 0.44);
    } else {
      state.targetCenter.x = h.pos.x;
      state.targetCenter.y = h.pos.y;
      state.targetAxis.x = h.axis.x;
      state.targetAxis.y = h.axis.y;
      state.targetR = clamp(h.size * minWH * 2.2, minWH * 0.17, minWH * 0.44);
    }
    state.targetRoll = h.roll;

    // 食指 + 中指指尖中点 = 剑流头部；按“位移 ≥5px 或 50ms”采样轨迹
    const tipA = h.tips[1], tipB = h.tips[2];
    const mx = (tipA.x + tipB.x) * 0.5, my = (tipA.y + tipB.y) * 0.5;
    const p0 = state.flow.pts[0];
    if (!p0 || Math.hypot(mx - p0.x, my - p0.y) > 4 || state.t - state.flow.last > 0.045) {
      state.flow.pts.unshift({ x: mx, y: my, ts: state.t });
      if (state.flow.pts.length > 160) state.flow.pts.pop();
      state.flow.last = state.t;
    }
  } else {
    state.cand = null;
    state.counts.length = 0;
    state.spreads.length = 0;
    if (!state.noHandSince) state.noHandSince = now;
    // 指针输入时阵心由鼠标控制，不能被无手空闲逻辑回收
    if (state.input !== "pointer" && (now - state.noHandSince) / 1000 > 1.0) {
      // 无手：阵自悬于画面中央，缓慢呼吸
      state.targetCenter.x = W * 0.5 + Math.sin(state.t * 0.11) * W * 0.018;
      state.targetCenter.y = H * 0.48 + Math.sin(state.t * 0.09 + 1.3) * H * 0.02;
      state.targetR = minWH * 0.33;
      const a = 1.02 + 0.22 * Math.sin(state.t * 0.12);
      state.targetAxis.x = Math.cos(a);
      state.targetAxis.y = Math.sin(a);
      state.targetRoll = 0;
    }
  }

  const kc = 1 - Math.exp(-dt * (state.mode === "stream" ? 18 : CONFIG.motion.easeCenter));
  state.center.x += (state.targetCenter.x - state.center.x) * kc;
  state.center.y += (state.targetCenter.y - state.center.y) * kc;

  const kr = 1 - Math.exp(-dt * (state.mode === "stream" ? 8.0 : 4.2));
  state.R += (state.targetR - state.R) * kr;

  const ka = 1 - Math.exp(-dt * (state.mode === "stream" ? 12.5 : 7.5));
  state.axis.x += (state.targetAxis.x - state.axis.x) * ka;
  state.axis.y += (state.targetAxis.y - state.axis.y) * ka;
  const al = Math.hypot(state.axis.x, state.axis.y) || 1;
  state.axis.x /= al; state.axis.y /= al;

  let dr = state.targetRoll - state.roll;
  while (dr > Math.PI) dr -= TAU;
  while (dr < -Math.PI) dr += TAU;
  state.roll += dr * (1 - Math.exp(-dt * 5));

  state.formationA += (state.formationTarget - state.formationA) * (1 - Math.exp(-dt * 3.4));
}

function setMode(m) {
  if (state.mode === m) return;
  state.mode = m;
  state.formationTarget = MODE_FORMATION[m] ?? 1;
  state.flow.pts.length = 0;   // 换印后清空旧轨迹，避免剑流拖着上一种形态的残影
  state.flow.last = 0;
  addShock();
  updateChips();
}

function updateChips() {
  for (const chip of el.chips) chip.classList.toggle("on", chip.dataset.mode === state.mode);
}

/* 手印条可点击：触屏 / 无键盘时也能换印 */
for (const chip of el.chips) {
  chip.addEventListener("click", () => setMode(chip.dataset.mode));
}

function addShock() {
  state.shock.push({ r: minWH * 0.06, a: 1, v: minWH * 0.55 });
  state.shock.push({ r: minWH * 0.02, a: 0.8, v: minWH * 0.75 });
  if (state.shock.length > 10) state.shock.splice(0, state.shock.length - 10);
  state.flash = 1;
}

/* =====================================================================
 * 背景 / 地面符阵 / 剑阵 / 电弧 / 尘光 / 暗角
 * =================================================================== */
/* 云气贴图按颜色缓存：避免每帧重复创建径向渐变 */
const cloudSprites = new Map();
function cloudSprite(color) {
  const key = color.join(",");
  let c = cloudSprites.get(key);
  if (c) return c;
  const s = 256;
  c = createCanvas(s, s);
  const g = c.getContext("2d");
  const rg = g.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
  rg.addColorStop(0, rgba(color, 1));
  rg.addColorStop(0.5, rgba(color, 0.36));
  rg.addColorStop(1, rgba(color, 0));
  g.fillStyle = rg;
  g.fillRect(0, 0, s, s);
  cloudSprites.set(key, c);
  return c;
}

function softEllipse(cx, cy, rx, ry, color, alpha) {
  if (alpha <= 0.001 || rx <= 0 || ry <= 0) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.drawImage(cloudSprite(color), cx - rx, cy - ry, rx * 2, ry * 2);
  ctx.restore();
}

function drawBackground(t) {
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  // 云气
  softEllipse(W * (0.30 + Math.sin(t * 0.05) * 0.03), H * 0.30,
    W * 0.34, H * 0.20, [22, 86, 60], 0.10);
  softEllipse(W * (0.74 + Math.cos(t * 0.04) * 0.03), H * 0.68,
    W * 0.40, H * 0.22, [34, 72, 92], 0.09);
  softEllipse(W * 0.52, H * 0.48, W * 0.26, H * 0.24, [28, 118, 84], 0.08);

  // 星屑
  for (const s of state.stars) {
    const tw = 0.35 + 0.65 * (0.5 + 0.5 * Math.sin(t * 1.2 + s.ph));
    ctx.fillStyle = s.hue === "gold"
      ? `rgba(255,222,164,${0.30 * tw})`
      : `rgba(196,244,196,${0.34 * tw})`;
    ctx.beginPath();
    ctx.arc(s.x * W, s.y * H, s.r, 0, TAU);
    ctx.fill();
  }
  ctx.restore();
}

function drawFormation(t) {
  const cx = state.center.x, cy = state.center.y, R = state.R;
  const a = state.formationA;
  if (a <= 0.015 || R <= 1) return;

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  // 外圈：断续符环
  ctx.strokeStyle = `rgba(150,232,160,${0.055 * a})`;
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(cx, cy, R * 1.66, 0, TAU); ctx.stroke();

  ctx.strokeStyle = `rgba(176,244,176,${0.11 * a})`;
  ctx.lineWidth = 1.2;
  ctx.setLineDash([R * 0.035, R * 0.075]);
  ctx.lineDashOffset = -t * R * 0.05;
  ctx.beginPath(); ctx.arc(cx, cy, R * 1.74, 0, TAU); ctx.stroke();
  ctx.setLineDash([]);

  // 八枚符字，绕阵缓行
  const chars = ["庚", "剑", "阵", "罡", "乾", "坤", "震", "巽"];
  ctx.font = `${Math.max(11, Math.round(R * 0.072))}px "Kaiti SC","STKaiti","KaiTi",serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (let i = 0; i < chars.length; i++) {
    const ang = (i / chars.length) * TAU + t * 0.05 + Math.PI / 8;
    const x = cx + Math.cos(ang) * R * 1.86;
    const y = cy + Math.sin(ang) * R * 1.86;
    ctx.fillStyle = `rgba(206,255,206,${0.10 * a})`;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(ang + Math.PI / 2);
    ctx.fillText(chars[i], 0, 0);
    ctx.restore();
  }

  // 地面符环（压扁的椭圆，表示阵盘）
  ctx.save();
  ctx.translate(cx, cy + R * 0.80);
  ctx.scale(1, 0.30);
  ctx.lineWidth = 1;
  const rings = [
    { r: R * 1.16, col: "rgba(150,232,160,", alpha: 0.10, dash: [R * 0.05, R * 0.08], dir: 1 },
    { r: R * 1.48, col: "rgba(255,206,132,", alpha: 0.055, dash: [R * 0.03, R * 0.11], dir: -1 },
    { r: R * 1.86, col: "rgba(150,232,160,", alpha: 0.045, dash: [R * 0.02, R * 0.14], dir: 1 },
  ];
  for (const rg of rings) {
    ctx.strokeStyle = `${rg.col}${rg.alpha * a})`;
    ctx.setLineDash(rg.dash);
    ctx.lineDashOffset = t * R * 0.05 * rg.dir;
    ctx.beginPath(); ctx.arc(0, 0, rg.r, 0, TAU); ctx.stroke();
  }
  ctx.setLineDash([]);
  ctx.restore();

  ctx.restore();
}

function drawSwords(dt) {
  const cx = state.center.x, cy = state.center.y, R = state.R;
  const F = CONFIG.projection.f;
  const isStream = state.mode === "stream";
  // 剑群里是小而细密的飞剑：细才透出流动的丝缕感
  const baseLen = minWH * (isStream ? 0.034 : 0.055);

  ctx.save();
  ctx.globalCompositeOperation = "lighter";

  // 阵心的一点青光：剑群里压暗缩小，只作指尖微光，不抢涡盘的戏
  const pulse = 0.10 + 0.035 * Math.sin(state.t * 1.6);
  ctx.globalAlpha = pulse * (isStream ? 0.5 : 1);
  const dotR = isStream ? R * 1.4 : R * 2.5;
  ctx.drawImage(SPR.dot, cx - dotR, cy - dotR, dotR * 2, dotR * 2);

  for (let i = 0; i < swords.length; i++) {
    const s = swords[i];
    const sc = F / (F - s.z);
    const sx = cx + s.x * R * sc;
    const sy = cy + s.y * R * sc;
    s.sx = sx; s.sy = sy;

    if (sx < -90 || sx > W + 90 || sy < -130 || sy > H + 130) { s.px = sx; s.py = sy; continue; }

    const alpha = s.a * (isStream ? 0.36 + 0.10 * FLOW.speedK : 0.34) + s.a * 0.48 * ((s.z + 1) / 2);
    if (alpha <= 0.012) { s.px = sx; s.py = sy; continue; }

    const speed = s.px === null ? 0 : Math.hypot(sx - s.px, sy - s.py) / Math.max(dt, 1e-4);
    const fore = clamp(Math.hypot(s.dx, s.dy), 0.18, 1);
    const len = baseLen * s.size * sc * (0.62 + 0.72 * s.seed) * (0.42 + 0.58 * fore);

    // 光尾：速度越快越长（剑群里压低门槛，飞起来就是一缕缕流光）
    if (speed > (isStream ? 40 : 70)) {
      const spdK = clamp(speed / 900, 0, 1);
      const tl = clamp(speed * (isStream ? 0.14 : 0.14), len * 0.5, len * (isStream ? 2.4 : 2.6));
      const tw = len * (isStream ? 0.13 : 0.22);
      const va = Math.atan2(sy - s.py, sx - s.px) - Math.PI / 2;
      ctx.globalAlpha = alpha * ((isStream ? 0.10 : 0.07) + (isStream ? 0.42 : 0.32) * spdK);
      ctx.save();
      ctx.translate(sx, sy);
      ctx.rotate(va);
      ctx.drawImage(STREAK_SPR[s.tint], -tw / 2, -tl / 2, tw, tl);
      ctx.restore();
    }

    // 剑身周围的一层柔光：剑群里就是一颗颗亮斑，密而碎才有云团的感觉
    ctx.globalAlpha = alpha * (isStream ? 0.16 + 0.14 * FLOW.speedK : 0.065);
    ctx.drawImage(SPR.dot, sx - len * 1.0, sy - len * 1.0, len * 2.0, len * 2.0);

    // 剑身（剑群里用细条：整体更纤细，丝缕感优先于剑的轮廓感）
    const ang = Math.atan2(s.dy, s.dx);
    const bw = len * (isStream ? 0.15 : 0.18);
    ctx.globalAlpha = alpha;
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(ang + Math.PI / 2);
    ctx.drawImage(BLADE_SPR[s.tint], -bw / 2, -len / 2, bw, len);
    ctx.restore();

    s.px = sx; s.py = sy;
  }
  ctx.restore();
}

function drawArcs(t) {
  const a = state.formationA;
  if (a < 0.2) return;
  const seed = Math.floor(t * 7);
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  ctx.lineCap = "round";
  for (let k = 0; k < 6; k++) {
    const i = (rnd(seed * 17.3 + k * 91.7) * swords.length) | 0;
    const A = swords[i];
    if (!A || A.a < 0.25) continue;
    // 随机抽样找最近邻，避免 O(n²)
    let best = null, bd = Infinity;
    for (let j = 0; j < 8; j++) {
      const idx = (rnd(seed * 7.7 + k * 31.1 + j * 5.3) * swords.length) | 0;
      const B = swords[idx];
      if (!B || B.a < 0.25) continue;
      const d = Math.hypot(A.sx - B.sx, A.sy - B.sy);
      if (d > 4 && d < bd) { bd = d; best = B; }
    }
    if (!best || bd > minWH * 0.5) continue;
    const flick = 0.5 + 0.5 * rnd(seed + k * 3.3);
    const alpha = (0.05 + 0.10 * flick) * a;
    ctx.strokeStyle = k % 3 === 0
      ? `rgba(255,214,140,${alpha})`
      : `rgba(160,240,150,${alpha})`;
    ctx.lineWidth = 0.6 + 1.2 * flick;
    ctx.beginPath();
    const steps = 6;
    for (let m = 0; m <= steps; m++) {
      const u = m / steps;
      const jx = (rnd(seed * 3.1 + k * 11 + m * 7.7) - 0.5) * 16;
      const jy = (rnd(seed * 5.9 + k * 13 + m * 9.1) - 0.5) * 16;
      const x = lerp(A.sx, best.sx, u) + (m === 0 || m === steps ? 0 : jx);
      const y = lerp(A.sy, best.sy, u) + (m === 0 || m === steps ? 0 : jy);
      m === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  ctx.restore();
}

function drawFingertips() {
  const h = state.hand;
  if (!h || state.demo) return;
  if (performance.now() - h.t > 450) return;
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  const pulse = 0.55 + 0.25 * Math.sin(state.t * 5.4);
  for (const p of h.tips) {
    ctx.globalAlpha = 0.42 * pulse;
    ctx.drawImage(SPR.dot, p.x - 18, p.y - 18, 36, 36);
    ctx.globalAlpha = 0.8 * pulse;
    ctx.fillStyle = "rgba(230,252,255,0.95)";
    ctx.beginPath();
    ctx.arc(p.x, p.y, 1.6, 0, TAU);
    ctx.fill();
  }
  // 掌心微光
  ctx.globalAlpha = 0.16;
  ctx.drawImage(SPR.dot, h.pos.x - 46, h.pos.y - 46, 92, 92);
  ctx.restore();
}

function drawDust(dt) {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const p of state.dust) {
    p.y -= p.vy * dt;
    if (p.y < -0.05) { p.y = 1.05; p.x = Math.random(); }
    const x = p.x * W + Math.sin(state.t * 0.6 + p.ph) * 10;
    const y = p.y * H;
    const a = 0.10 + 0.12 * (0.5 + 0.5 * Math.sin(state.t * 1.8 + p.ph));
    ctx.globalAlpha = a;
    ctx.fillStyle = p.gold ? "rgba(255,216,148,1)" : "rgba(170,240,170,1)";
    ctx.beginPath();
    ctx.arc(x, y, p.s, 0, TAU);
    ctx.fill();
  }
  ctx.restore();
}

function updateShock(dt) {
  for (let i = state.shock.length - 1; i >= 0; i--) {
    const s = state.shock[i];
    s.r += s.v * dt;
    s.a -= dt * 1.25;
    if (s.a <= 0) state.shock.splice(i, 1);
  }
  state.flash = Math.max(0, state.flash - dt * 2.2);
}

function drawShock() {
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const s of state.shock) {
    const a = Math.max(0, s.a);
    ctx.strokeStyle = `rgba(170,244,170,${a * 0.5})`;
    ctx.lineWidth = 1 + a * 2;
    ctx.beginPath(); ctx.arc(state.center.x, state.center.y, s.r, 0, TAU); ctx.stroke();

    ctx.save();
    ctx.translate(state.center.x, state.center.y + state.R * 0.6);
    ctx.scale(1, 0.32);
    ctx.strokeStyle = `rgba(255,214,140,${a * 0.34})`;
    ctx.beginPath(); ctx.arc(0, 0, s.r, 0, TAU); ctx.stroke();
    ctx.restore();
  }
  if (state.flash > 0.01) {
    ctx.fillStyle = `rgba(150,235,140,${state.flash * 0.045})`;
    ctx.fillRect(0, 0, W, H);
  }
  ctx.restore();
}

function drawVignette() {
  ctx.fillStyle = vigGrad;
  ctx.fillRect(0, 0, W, H);
}

function drawGrain() {
  if (!grainPat) return;
  const ox = (Math.random() * 128) | 0;
  const oy = (Math.random() * 128) | 0;
  ctx.save();
  ctx.globalCompositeOperation = "overlay";
  ctx.globalAlpha = 0.028;
  ctx.translate(-ox, -oy);
  ctx.fillStyle = grainPat;
  ctx.fillRect(0, 0, W + 128, H + 128);
  ctx.restore();
}

/* =====================================================================
 * 主循环
 * =================================================================== */
function render(dt) {
  drawBackground(state.t);
  drawFormation(state.t);
  drawSwords(dt);
  drawArcs(state.t);
  drawFingertips();
  drawDust(dt);
  drawShock();
  drawVignette();
  drawGrain();
}

/* ---------------- 手势调试读数（?debug=1） ---------------- */
const DEBUG_ON = params.get("debug") === "1";
const dbgHud = DEBUG_ON ? (() => {
  const d = document.createElement("div");
  d.style.cssText = "position:fixed;left:12px;top:10px;z-index:9;font:11px/1.75 ui-monospace,Consolas,monospace;color:#8fe3ff;background:rgba(4,8,14,.74);border:1px solid rgba(124,228,255,.22);border-radius:4px;padding:7px 10px;white-space:pre;pointer-events:none";
  document.body.appendChild(d);
  return d;
})() : null;

function updateDebug(now) {
  if (!dbgHud) return;
  const h = state.hand;
  const fresh = h && now - h.t < 400;
  const spMed = median(state.spreads);
  dbgHud.textContent =
    "cam=" + (camReady ? "ok" : "no") + "  model=" + (landmarker ? "ok" : "no") + "  input=" + state.input + "\n" +
    "hand=" + (fresh ? h.label : "-") + "  count=" + (fresh ? h.count : "-") + "  med=" + median(state.counts) +
    "  spread=" + (fresh ? h.spread.toFixed(2) : "-") + "  spMed=" + (spMed < 0 ? "-" : spMed.toFixed(2)) + "\n" +
    "mode=" + state.mode + "  cand=" + (state.cand || "-") + "  flow=" + (FLOW.speedK * 100).toFixed(0) + "%/" + state.flow.pts.length + "\n" +
    "center=" + (state.center.x | 0) + "," + (state.center.y | 0) + "  R=" + (state.R / (minWH || 1)).toFixed(2);
}

let last = performance.now();
function loop(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  state.t += dt;

  detect(now);
  updateGesture(dt, now);
  updateSwords(dt, state.t);
  updateShock(dt);
  updateDebug(now);

  /* 左右手标签防抖 */
  if (el.tagLeft)  el.tagLeft.classList.toggle("on",  now - state.handSeen.left  < 400);
  if (el.tagRight) el.tagRight.classList.toggle("on", now - state.handSeen.right < 400);

  /* 预览模式：auto 轮播 */
  if (AUTO_SHOW && state.t * 1000 > state.autoAt + 5200) {
    state.autoAt += 5200;
    const i = MODES.indexOf(state.mode);
    setMode(MODES[(i + 1) % MODES.length]);
  }

  render(dt);
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* =====================================================================
 * 提示 / 错误
 * =================================================================== */
let toastTimer = null;
function showToast(text, ms = 4200) {
  if (!el.toast || state.demo) return;
  el.toast.textContent = text;
  el.toast.classList.add("show");
  clearTimeout(toastTimer);
  if (ms > 0) toastTimer = setTimeout(() => el.toast.classList.remove("show"), ms);
}

function showErrorDetail(txt) {
  if (!el.errDetail || !txt) return;
  el.errDetail.textContent = `[加载诊断] ${txt}`;
  el.errDetail.classList.remove("hidden");
}
function hideErrorDetail() {
  if (!el.errDetail) return;
  el.errDetail.classList.add("hidden");
  el.errDetail.textContent = "";
}

function enterFallback(msg, tip) {
  state.input = "pointer";
  el.camWrap.classList.add("hidden");
  video.style.display = "none";
  skel.style.display = "none";
  showToast(msg || "鼠标引阵 · 数字键 1–4 换印", 7000);
  const fullTip = [tip, ENV_NOTE].filter(Boolean).join("\n");
  if (fullTip) showErrorDetail(fullTip);
}

/* =====================================================================
 * 键盘 / 鼠标降级
 * =================================================================== */
const KEY_MODE = { Digit1: "stream", Digit2: "scatter", Digit3: "ring", Digit4: "ball" };

window.addEventListener("keydown", e => {
  // 入口页未关闭时不吃快捷键，避免改到看不见的形态
  if (!PREVIEW && !el.gate.classList.contains("hide")) return;
  if (KEY_MODE[e.code]) { setMode(KEY_MODE[e.code]); return; }
  if (e.code === "Space") { e.preventDefault(); addShock(); }
  else if (e.code === "KeyC") {
    if (camReady) { stopCamera(); state.input = "pointer"; showToast("摄像头已关闭 · 按 C 重新开启", 4200); }
    else { state.input = "camera"; resumeCamera(); }
  }
  else if (e.code === "KeyD") { setDemo(!state.demo); }
  else if (e.code === "Escape" && state.demo) { setDemo(false); }
});

window.addEventListener("pointermove", e => {
  if (state.input === "camera" && camReady) return;
  state.targetCenter.x = e.clientX;
  state.targetCenter.y = e.clientY;

  // 鼠标/触控同样能引剑流：把指针轨迹当作河道，方向由移动方向决定
  const src = state.flow.pts;
  const head = src[0];
  const near = head ? Math.hypot(e.clientX - head.x, e.clientY - head.y) : Infinity;
  if (near > 5 || state.t - state.flow.last > 0.05) {
    if (head) {
      // 取一个稍远的点算方向，避免逐个事件抖动导致朝向乱跳
      const ref = src.find(p => Math.hypot(e.clientX - p.x, e.clientY - p.y) > 24) || head;
      const vx = e.clientX - ref.x, vy = e.clientY - ref.y;
      const vl = Math.hypot(vx, vy);
      if (vl > 6) { state.targetAxis.x = vx / vl; state.targetAxis.y = vy / vl; }
    }
    src.unshift({ x: e.clientX, y: e.clientY, ts: state.t });
    if (src.length > 160) src.pop();
    state.flow.last = state.t;
  }
});

/* ---------------- 演示模式 ---------------- */
let blackoutTimer = null;
function setDemo(on) {
  if (state.demo === on) return;
  el.blackout.classList.add("on");
  clearTimeout(blackoutTimer);
  blackoutTimer = setTimeout(() => {
    state.demo = on;
    document.body.classList.toggle("demo", on);
    el.blackout.classList.remove("on");
    if (on) { el.toast.classList.remove("show"); hideErrorDetail(); }
  }, 180);
}

/* ---------------- 开场揭幕 ---------------- */
function playReveal() {
  const rv = el.reveal;
  if (!rv) return;
  rv.classList.remove("hidden", "play", "fade");
  void rv.offsetWidth;
  rv.classList.add("play");
  setTimeout(() => rv.classList.add("fade"), 1100);
  setTimeout(() => { rv.classList.add("hidden"); rv.classList.remove("play", "fade"); }, 1500);
}

/* ---------------- 回看小窗：拖动 + 记住位置 ---------------- */
/* 窗口变小后把小窗夹回可视范围，避免停在屏幕外找不回来 */
function clampCamWrap() {
  const w = el.camWrap;
  if (!w || !w.offsetWidth || w.classList.contains("hidden") || !w.style.left) return;
  const l = parseFloat(w.style.left), t = parseFloat(w.style.top);
  if (!isFinite(l) || !isFinite(t)) return;
  w.style.left = clamp(l, 0, Math.max(0, window.innerWidth - w.offsetWidth)) + "px";
  w.style.top = clamp(t, 0, Math.max(0, window.innerHeight - w.offsetHeight)) + "px";
}

skel.width = 1280; skel.height = 720;
(function setupCamDrag() {
  const w = el.camWrap;
  if (!w) return;
  try {
    const saved = JSON.parse(localStorage.getItem("jianzhenCamPos") || "null");
    if (saved && typeof saved.x === "number") {
      w.style.right = "auto"; w.style.bottom = "auto";
      w.style.left = (saved.x * 100) + "vw";
      w.style.top = (saved.y * 100) + "vh";
    }
  } catch (e) {}
  let dragging = false, ox = 0, oy = 0;
  const onDown = e => {
    dragging = true; w.classList.add("dragging");
    const r = w.getBoundingClientRect();
    ox = e.clientX - r.left; oy = e.clientY - r.top;
    e.preventDefault();
  };
  const onMove = e => {
    if (!dragging) return;
    w.style.right = "auto"; w.style.bottom = "auto";
    const nx = Math.max(0, Math.min(window.innerWidth - w.offsetWidth, e.clientX - ox));
    const ny = Math.max(0, Math.min(window.innerHeight - w.offsetHeight, e.clientY - oy));
    w.style.left = nx + "px"; w.style.top = ny + "px";
    try {
      localStorage.setItem("jianzhenCamPos",
        JSON.stringify({ x: nx / window.innerWidth, y: ny / window.innerHeight }));
    } catch (e) {}
  };
  const onUp = () => { dragging = false; w.classList.remove("dragging"); };
  w.addEventListener("pointerdown", onDown);
  window.addEventListener("pointermove", onMove);
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", onUp);
})();

/* ---------------- 后台 / 失焦的卫生处理 ---------------- */
document.addEventListener("visibilitychange", () => {
  if (document.hidden) stopCamera();
  else resumeCamera();
});
window.addEventListener("blur", () => { stopCamera(); });
window.addEventListener("focus", () => { if (!document.hidden) resumeCamera(); });

/* =====================================================================
 * 入口：点击进入（请求摄像头 + 加载模型）
 * =================================================================== */
el.startBtn.addEventListener("click", async () => {
  el.gate.classList.add("hide");
  playReveal();
  state.input = "camera";

  // 先请求摄像头：仍然处在 click 手势内，授权弹窗不会因后续的长耗时任务而挂起
  showToast("正在请求摄像头权限…", 0);
  const camP = initCamera();
  const hint = setTimeout(() => {
    if (!camReady) showToast("没看到授权弹窗？点地址栏左侧「摄像头」图标 → 允许；或按 C 重试", 8000);
  }, 6000);

  // 等用户的真实决定；等待期间不加载大文件，页面保持可响应
  const camRes = await withTimeout(camP, 35000);
  clearTimeout(hint);
  if (camRes !== "ok") {
    if (camRes === false) {
      // 35 秒还没回音：提示排查，同时保留“迟到的授权”，稍后允许也能自动接回
      camP.then(r => {
        if (r !== "ok") return;
        if (camReady) el.camWrap.classList.remove("hidden");
        state.input = "camera";
        resumeCamera();
      });
      enterFallback("摄像头授权无响应",
        "若系统或浏览器拦住了弹窗：Windows「设置 → 隐私和安全性 → 摄像头」里允许桌面应用访问；再点地址栏左侧的摄像头图标改成「允许」，然后按 C 重试。");
    } else {
      const m = camFailMessage(camRes);
      enterFallback(m.text, m.tip);
    }
    return;
  }

  showToast("正在加载手势库…", 0);
  if (!(await withTimeout(loadVisionLibrary(), 20000))) {
    stopCamera();
    enterFallback("手势库加载失败 · 见下方诊断",
      "请确认 vendor/mediapipe 目录完整；也可直接用鼠标 + 数字键体验。");
    return;
  }

  showToast("正在加载手势模型…", 0);
  if (!(await withTimeout(loadHandModel(), 30000))) {
    stopCamera();
    enterFallback("模型加载失败",
      "请确认 vendor/models/hand_landmarker.task 存在且完整（约 7.6 MB）。");
    return;
  }

  hideErrorDetail();
  showToast("起手结印：双指剑流 · 三指散剑 · 五指剑环 · 握拳剑球", 6000);
});

/* =====================================================================
 * 预览 / 演示模式：?mode=ring|ball|stream|scatter|auto&still=1&demo=1
 * =================================================================== */
if (PREVIEW) {
  document.body.classList.add("preview");
  el.gate.classList.add("hidden");
  state.input = "pointer";
  const m = params.get("mode");
  if (m && MODES.includes(m)) {
    state.mode = m;
    state.formationA = state.formationTarget = MODE_FORMATION[m];
  }
  state.autoAt = state.t * 1000;
  updateChips();
}
if (DEMO_START) {
  state.demo = true;
  document.body.classList.add("demo");
}
updateChips();
