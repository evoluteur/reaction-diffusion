/*
  Reaction-Diffusion
  The Gray-Scott model on the GPU (WebGL2): two chemicals, A and B, spread
  and react (A + 2B -> 3B), A is fed in, B is removed. Depending on the feed
  and kill rates, the same two lines of math grow spots, stripes, coral,
  worms or waves.
  https://github.com/evoluteur/reaction-diffusion
  (c) 2026 Olivier Giulieri
*/

const SIM = 512; // simulation grid (SIM x SIM cells, wrapping at the edges)

// feed (f) and kill (k) rates for the patterns named in the literature
// (Pearson 1993, Munafo's xmorphia, Karl Sims' tutorial)
const presets = [
  { id: "coral", name: "Coral", f: 0.0545, k: 0.062, text: "Branching growth that fills every gap it finds, like coral or lichen." },
  { id: "mitosis", name: "Mitosis", f: 0.0367, k: 0.0649, text: "Spots that grow, stretch and split in two, like dividing cells." },
  { id: "spots", name: "Spots", f: 0.03, k: 0.062, text: "Stable round spots, evenly spaced: the leopard and the cheetah." },
  { id: "stripes", name: "Labyrinth", f: 0.029, k: 0.057, text: "Stripes that wind into a maze: the zebra, the angelfish, the brain coral." },
  { id: "fingerprint", name: "Fingerprint", f: 0.037, k: 0.06, text: "Parallel ridges that curve and fork, like the skin of a fingertip." },
  { id: "worms", name: "Worms", f: 0.078, k: 0.061, text: "Short worms that crawl apart and stop, each at its own length." },
  { id: "holes", name: "Holes", f: 0.039, k: 0.058, text: "The negative of spots: a sheet of B pierced with round holes." },
  { id: "waves", name: "Waves", f: 0.014, k: 0.054, text: "Pulsing rings that collide and cancel, never settling." },
  { id: "chaos", name: "Chaos", f: 0.026, k: 0.051, text: "Turbulence: patches form and break apart forever." },
  { id: "uskate", name: "U-skate", f: 0.062, k: 0.0609, text: "Rare little U-shapes that glide across the dish (seed a few and wait)." },
];

const palettes = [
  { id: "ink", name: "Ink on paper", stops: ["#f4eddd", "#d9cdb4", "#7a6f5c", "#2a2621", "#0e0d0c"] },
  { id: "coral", name: "Coral reef", stops: ["#06223a", "#0d5c75", "#f2a65a", "#ff6f59", "#fff1d6"] },
  { id: "ember", name: "Ember", stops: ["#050203", "#3b0a0a", "#c1272d", "#f7931e", "#fff3b0"] },
  { id: "ocean", name: "Ocean", stops: ["#001219", "#005f73", "#0a9396", "#94d2bd", "#e9d8a6"] },
  { id: "moss", name: "Moss", stops: ["#0f160e", "#2d4a22", "#6a8d3a", "#c9d27c", "#f6f2d4"] },
  { id: "mono", name: "Black & white", stops: ["#000000", "#2b2b2b", "#808080", "#d4d4d4", "#ffffff"] },
];

// limits of the feed / kill sliders and of the map
const F_MIN = 0.01, F_MAX = 0.1, K_MIN = 0.045, K_MAX = 0.07;

const state = {
  f: 0.0545,
  k: 0.062,
  preset: "coral",
  palette: "coral",
  speed: 12, // simulation steps per frame
  brush: 10,
  relief: true,
  map: false, // parameter map: f varies along y and k along x
  paused: false,
  steps: 0,
  settled: false, // nothing has moved for a while: stop computing
  faded: false, // and nothing is left
};

let gl, canvas, progSim, progDraw, progBrush, progProbe;
let probeTex, probeFbo;
const PROBE = 64; // the dish is sampled on a 64 x 64 grid to tell if it still moves
const CHECK_EVERY = 600; // steps between two looks
const STILL = 0.1; // largest change of the shown color level between two looks, for "nothing moves visibly"
const STILL_LOOKS = 3; // looks in a row
let lastProbe = null, lastCheck = 0, stillCount = 0;
let tex = [], fbo = [], cur = 0, quad;
let strokeSeed = 0; // changes with each stroke, so the speckles differ
let pointer = null; // {x, y, erase} in simulation cells while painting

// ------------------------------------------------------------ shaders

const VS = `#version 300 es
in vec2 pos;
out vec2 uv;
void main() {
  uv = pos * 0.5 + 0.5;
  gl_Position = vec4(pos, 0.0, 1.0);
}`;

const FS_SIM = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 outColor;
uniform sampler2D state;
uniform vec2 px;
uniform float f;
uniform float k;
uniform bool map;
void main() {
  vec2 c = texture(state, uv).rg;
  // 3x3 Laplacian: 0.2 to the sides, 0.05 to the corners, -1 at the center
  vec2 lap = -c
    + 0.2 * (texture(state, uv + vec2(px.x, 0.0)).rg + texture(state, uv - vec2(px.x, 0.0)).rg
           + texture(state, uv + vec2(0.0, px.y)).rg + texture(state, uv - vec2(0.0, px.y)).rg)
    + 0.05 * (texture(state, uv + px).rg + texture(state, uv - px).rg
            + texture(state, uv + vec2(px.x, -px.y)).rg + texture(state, uv + vec2(-px.x, px.y)).rg);
  float F = f, K = k;
  if (map) {
    K = mix(${K_MIN.toFixed(3)}, ${K_MAX.toFixed(3)}, uv.x);
    F = mix(${F_MIN.toFixed(3)}, ${F_MAX.toFixed(3)}, uv.y);
  }
  float a = c.r, b = c.g;
  float abb = a * b * b;
  // Gray-Scott, with Da = 1, Db = 0.5, dt = 1
  float na = a + (1.0 * lap.r - abb + F * (1.0 - a));
  float nb = b + (0.5 * lap.g + abb - (K + F) * b);
  outColor = vec4(clamp(na, 0.0, 1.0), clamp(nb, 0.0, 1.0), 0.0, 1.0);
}`;

const FS_BRUSH = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 outColor;
uniform sampler2D state;
uniform vec2 center;
uniform float radius;
uniform bool erase;
uniform float size;
uniform float seed;
float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7)) + seed) * 43758.5453);
}
void main() {
  vec2 c = texture(state, uv).rg;
  vec2 d = uv * size - center;
  d = d - size * floor(d / size + 0.5); // wrap around the edges
  float r = length(d);
  float m = 1.0 - smoothstep(radius * 0.7, radius, r);
  if (erase) {
    c = mix(c, vec2(1.0, 0.0), m);
  } else {
    // speckles of B (3x3 cells, 15% of them) rather than a solid blot:
    // big uniform patches die out in the spot regimes, small seeds grow
    float speck = step(hash(floor(uv * size / 3.0)), 0.15);
    c = mix(c, vec2(0.5, 0.5), m * speck);
  }
  outColor = vec4(c, 0.0, 1.0);
}`;

const FS_DRAW = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 outColor;
uniform sampler2D state;
uniform vec2 px;
uniform vec3 stops[5];
uniform bool relief;
float level(vec2 p) {
  return clamp(texture(state, p).g * 2.6, 0.0, 1.0);
}
vec3 ramp(float t) {
  float s = t * 4.0;
  if (s < 1.0) return mix(stops[0], stops[1], s);
  if (s < 2.0) return mix(stops[1], stops[2], s - 1.0);
  if (s < 3.0) return mix(stops[2], stops[3], s - 2.0);
  return mix(stops[3], stops[4], clamp(s - 3.0, 0.0, 1.0));
}
void main() {
  float t = level(uv);
  vec3 col = ramp(smoothstep(0.02, 0.98, t));
  if (relief) {
    float dx = level(uv + vec2(px.x, 0.0)) - level(uv - vec2(px.x, 0.0));
    float dy = level(uv + vec2(0.0, px.y)) - level(uv - vec2(0.0, px.y));
    vec3 n = normalize(vec3(-dx * 3.0, -dy * 3.0, 1.0));
    vec3 l = normalize(vec3(-0.5, 0.6, 0.65));
    float diff = dot(n, l);
    float spec = pow(max(dot(reflect(-l, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
    col = col * (0.62 + 0.42 * diff) + spec * 0.22;
  }
  outColor = vec4(col, 1.0);
}`;

// B at the center of each probe cell (the dish is read at 64 x 64 points),
// as the color level it is drawn with (see level() in FS_DRAW)
const FS_PROBE = `#version 300 es
precision highp float;
in vec2 uv;
out vec4 outColor;
uniform sampler2D state;
void main() {
  float b = texture(state, uv).g;
  outColor = vec4(clamp(b * 2.6, 0.0, 1.0), b, 0.0, 1.0);
}`;

// ------------------------------------------------------------ WebGL setup

const compile = (type, src) => {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s));
  }
  return s;
};

const program = (fs) => {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.bindAttribLocation(p, 0, "pos");
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    throw new Error(gl.getProgramInfoLog(p));
  }
  p.u = {};
  const n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) {
    const name = gl.getActiveUniform(p, i).name.replace("[0]", "");
    p.u[name] = gl.getUniformLocation(p, name);
  }
  return p;
};

const makeTexture = (data) => {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, SIM, SIM, 0, gl.RGBA, gl.FLOAT, data);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
  return t;
};

const initGL = () => {
  canvas = document.getElementById("rd-canvas");
  gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true, antialias: false });
  if (!gl || !gl.getExtension("EXT_color_buffer_float")) {
    return false;
  }
  canvas.width = SIM;
  canvas.height = SIM;
  progSim = program(FS_SIM);
  progDraw = program(FS_DRAW);
  progBrush = program(FS_BRUSH);
  progProbe = program(FS_PROBE);
  quad = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  for (let i = 0; i < 2; i++) {
    tex[i] = makeTexture(null);
    fbo[i] = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[i]);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex[i], 0);
  }
  if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
    return false;
  }
  probeTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, probeTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, PROBE, PROBE, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  probeFbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, probeFbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, probeTex, 0);
  return true;
};

// read B at the probe points
const probe = () => {
  gl.useProgram(progProbe);
  gl.bindFramebuffer(gl.FRAMEBUFFER, probeFbo);
  gl.viewport(0, 0, PROBE, PROBE);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex[cur]);
  gl.uniform1i(progProbe.u.state, 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  const out = new Float32Array(PROBE * PROBE * 4);
  gl.readPixels(0, 0, PROBE, PROBE, gl.RGBA, gl.FLOAT, out);
  const level = new Float32Array(PROBE * PROBE);
  let most = 0; // the most B anywhere, to tell when it has died out
  for (let i = 0; i < level.length; i++) {
    level[i] = out[i * 4];
    most = Math.max(most, out[i * 4 + 1]);
  }
  return { level, most };
};

// has the dish stopped changing, as far as the eye can tell? Holes, spots and
// mitosis go still; coral and labyrinths mostly do, though a stray defect may
// still creep; waves, chaos and worms never do; and a dish where B died out is
// still too
const checkSettled = () => {
  lastCheck = state.steps;
  const p = probe();
  if (lastProbe) {
    let change = 0;
    for (let i = 0; i < p.level.length; i++) change = Math.max(change, Math.abs(p.level[i] - lastProbe.level[i]));
    stillCount = change < STILL ? stillCount + 1 : 0;
    if (stillCount >= STILL_LOOKS) {
      state.settled = true;
      state.faded = p.most < 0.02;
      lastShown = -1;
    }
  }
  lastProbe = p;
};

// something changed: compute again
const wake = () => {
  if (state.settled) lastShown = -1;
  state.settled = false;
  state.faded = false;
  lastProbe = null;
  stillCount = 0;
  lastCheck = state.steps;
};

// one pass of a full-screen shader, reading the current state, writing the other buffer
const pass = (prog, setUniforms) => {
  gl.useProgram(prog);
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo[1 - cur]);
  gl.viewport(0, 0, SIM, SIM);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex[cur]);
  gl.uniform1i(prog.u.state, 0);
  setUniforms(prog.u);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  cur = 1 - cur;
};

const step = () => {
  pass(progSim, (u) => {
    gl.uniform2f(u.px, 1 / SIM, 1 / SIM);
    gl.uniform1f(u.f, state.f);
    gl.uniform1f(u.k, state.k);
    gl.uniform1i(u.map, state.map ? 1 : 0);
  });
  state.steps++;
};

const brush = (x, y, erase) => {
  pass(progBrush, (u) => {
    gl.uniform2f(u.center, x, y);
    gl.uniform1f(u.radius, state.brush);
    gl.uniform1i(u.erase, erase ? 1 : 0);
    gl.uniform1f(u.size, SIM);
    gl.uniform1f(u.seed, strokeSeed);
  });
};

const hexToRgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16) / 255);

const draw = () => {
  gl.useProgram(progDraw);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex[cur]);
  gl.uniform1i(progDraw.u.state, 0);
  gl.uniform2f(progDraw.u.px, 1 / SIM, 1 / SIM);
  gl.uniform1i(progDraw.u.relief, state.relief ? 1 : 0);
  const pal = palettes.find((p) => p.id === state.palette) || palettes[0];
  gl.uniform3fv(progDraw.u.stops, new Float32Array(pal.stops.flatMap(hexToRgb)));
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
};

// ------------------------------------------------------------ seeding

// a fresh dish: all A, with a few drops of B to start the reaction
const seed = (how = "drops") => {
  const data = new Float32Array(SIM * SIM * 4);
  for (let i = 0; i < SIM * SIM; i++) {
    data[i * 4] = 1;
    data[i * 4 + 3] = 1;
  }
  const dot = (cx, cy, r) => {
    for (let y = -r; y <= r; y++) {
      for (let x = -r; x <= r; x++) {
        if (x * x + y * y > r * r) continue;
        const px = (Math.round(cx + x) + SIM) % SIM;
        const py = (Math.round(cy + y) + SIM) % SIM;
        const i = (py * SIM + px) * 4;
        data[i] = 0.5;
        data[i + 1] = 0.25 + Math.random() * 0.25;
      }
    }
  };
  if (how === "noise") {
    for (let n = 0; n < 900; n++) dot(Math.random() * SIM, Math.random() * SIM, 2);
  } else if (how !== "none") {
    // patches of small speckles (3 x 3 cells, 15% of them): a large solid
    // blot can die out in some regimes (waves, spots), small seeds always grow
    const cell = 3;
    const cells = Math.ceil(SIM / cell);
    const speck = new Uint8Array(cells * cells).map(() => (Math.random() < 0.15 ? 1 : 0));
    const n = 8 + Math.floor(Math.random() * 6);
    for (let d = 0; d < n; d++) {
      const cx = Math.random() * SIM, cy = Math.random() * SIM;
      const r = 20 + Math.random() * 40;
      for (let y = -r; y <= r; y++) {
        for (let x = -r; x <= r; x++) {
          if (x * x + y * y > r * r) continue;
          const px = (Math.round(cx + x) + SIM) % SIM;
          const py = (Math.round(cy + y) + SIM) % SIM;
          if (!speck[Math.floor(py / cell) * cells + Math.floor(px / cell)]) continue;
          const i = (py * SIM + px) * 4;
          data[i] = 0.5;
          data[i + 1] = 0.5;
        }
      }
    }
  }
  for (let i = 0; i < 2; i++) {
    gl.bindTexture(gl.TEXTURE_2D, tex[i]);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, SIM, SIM, 0, gl.RGBA, gl.FLOAT, data);
  }
  state.steps = 0;
  wake();
};

// ------------------------------------------------------------ the loop

const loop = () => {
  if (pointer) {
    // dabs along the path since the last frame, so fast strokes stay continuous
    const pts = pointer.trail.length ? pointer.trail : [{ x: pointer.x, y: pointer.y }];
    pts.slice(-60).forEach((p) => brush(p.x, p.y, pointer.erase));
    pointer.trail = [];
  }
  if (!state.paused && !state.settled) {
    for (let i = 0; i < state.speed; i++) step();
    if (state.steps - lastCheck >= CHECK_EVERY) checkSettled();
  }
  draw();
  updateStepCount();
  requestAnimationFrame(loop);
};

let lastShown = -1;
const updateStepCount = () => {
  if (state.steps === lastShown) return;
  lastShown = state.steps;
  const el = document.getElementById("step-info");
  if (!el) return;
  const n = state.steps.toLocaleString();
  el.textContent = state.faded
    ? `Faded away at step ${n}: nothing is left. Reseed or paint to start again.`
    : state.settled
      ? `Settled at step ${n}: nothing moves anymore.`
      : `Step ${n}.`;
};

// ------------------------------------------------------------ painting

const toCell = (e) => {
  const r = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - r.left) / r.width) * SIM,
    y: (1 - (e.clientY - r.top) / r.height) * SIM, // textures have y up
  };
};

const setupPainting = () => {
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const c = toCell(e);
    strokeSeed = Math.random() * 1000;
    pointer = { ...c, trail: [c], erase: e.button === 2 || e.shiftKey };
    wake();
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointer) return;
    const c = toCell(e);
    const d = Math.hypot(c.x - pointer.x, c.y - pointer.y);
    const n = Math.min(40, Math.ceil(d / Math.max(1, state.brush * 0.4)));
    for (let i = 1; i <= n; i++) {
      pointer.trail.push({
        x: pointer.x + ((c.x - pointer.x) * i) / n,
        y: pointer.y + ((c.y - pointer.y) * i) / n,
      });
    }
    pointer.x = c.x;
    pointer.y = c.y;
  });
  const stop = () => (pointer = null);
  canvas.addEventListener("pointerup", stop);
  canvas.addEventListener("pointercancel", stop);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
};

// ------------------------------------------------------------ feed / kill map

let hoverId = null; // pattern under the mouse on the map
const mapCanvas = () => document.getElementById("fk-map");
const kToX = (k, w) => ((k - K_MIN) / (K_MAX - K_MIN)) * w;
const fToY = (f, h) => h - ((f - F_MIN) / (F_MAX - F_MIN)) * h;

const drawMap = () => {
  const c = mapCanvas();
  if (!c) return;
  const ratio = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * ratio)) {
    c.width = Math.round(w * ratio);
    c.height = Math.round(h * ratio);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(c);
  const fg = css.color;
  ctx.strokeStyle = fg;
  ctx.globalAlpha = 0.15;
  for (let i = 1; i < 5; i++) {
    ctx.beginPath();
    ctx.moveTo((w * i) / 5, 0);
    ctx.lineTo((w * i) / 5, h);
    ctx.moveTo(0, (h * i) / 5);
    ctx.lineTo(w, (h * i) / 5);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
  ctx.font = "11px Overpass, sans-serif";
  ctx.textBaseline = "middle";
  presets.forEach((p) => {
    const x = kToX(p.k, w), y = fToY(p.f, h);
    ctx.fillStyle = p.id === state.preset ? "#ff6f59" : fg;
    ctx.globalAlpha = p.id === state.preset ? 1 : 0.7;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
    if (p.id === state.preset || p.id === hoverId) {
      const right = x < w - 80;
      ctx.textAlign = right ? "left" : "right";
      ctx.fillText(p.name, x + (right ? 10 : -10), y);
    }
  });
  ctx.globalAlpha = 1;
  // the current feed / kill
  const x = kToX(state.k, w), y = fToY(state.f, h);
  ctx.strokeStyle = "#ff6f59";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.lineWidth = 1;
};

const setupMap = () => {
  const c = mapCanvas();
  if (!c) return;
  let dragging = false;
  const pick = (e) => {
    const r = c.getBoundingClientRect();
    const kx = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const fy = Math.min(1, Math.max(0, 1 - (e.clientY - r.top) / r.height));
    let k = K_MIN + kx * (K_MAX - K_MIN);
    let f = F_MIN + fy * (F_MAX - F_MIN);
    // snap to a named pattern when close to it
    const near = presets.find(
      (p) => Math.abs(kToX(p.k, r.width) - kToX(k, r.width)) < 6 && Math.abs(fToY(p.f, r.height) - fToY(f, r.height)) < 6,
    );
    if (near) {
      setPreset(near.id, false);
    } else {
      setRates(f, k);
    }
  };
  c.addEventListener("pointerdown", (e) => {
    dragging = true;
    c.setPointerCapture(e.pointerId);
    pick(e);
  });
  c.addEventListener("pointermove", (e) => {
    if (dragging) return pick(e);
    const r = c.getBoundingClientRect();
    const mx = e.clientX - r.left, my = e.clientY - r.top;
    const near = presets.find(
      (p) => Math.hypot(kToX(p.k, r.width) - mx, fToY(p.f, r.height) - my) < 7,
    );
    const id = near ? near.id : null;
    if (id !== hoverId) {
      hoverId = id;
      c.title = near ? near.name : "";
      drawMap();
    }
  });
  c.addEventListener("pointerleave", () => {
    hoverId = null;
    drawMap();
  });
  c.addEventListener("pointerup", () => (dragging = false));
  window.addEventListener("resize", drawMap);
};

// ------------------------------------------------------------ controls

const $ = (id) => document.getElementById(id);

const showRates = () => {
  $("feed").value = state.f;
  $("kill").value = state.k;
  $("feed-val").textContent = state.f.toFixed(4);
  $("kill-val").textContent = state.k.toFixed(4);
  const p = presets.find((p) => p.id === state.preset);
  $("preset").value = p ? p.id : "";
  $("preset-text").textContent = state.map
    ? "The parameter map: feed grows from bottom to top, kill from left to right. Every pattern at once."
    : p
      ? p.text
      : "Your own mix. Patterns live along a thin band of the map, outside it everything dies out or turns solid.";
  drawMap();
  saveHash();
};

const setRates = (f, k) => {
  wake();
  state.f = f;
  state.k = k;
  const p = presets.find((p) => Math.abs(p.f - f) < 1e-6 && Math.abs(p.k - k) < 1e-6);
  state.preset = p ? p.id : "";
  showRates();
};

const setPreset = (id, reseed = true) => {
  const p = presets.find((p) => p.id === id);
  if (!p) return;
  wake();
  state.preset = p.id;
  state.f = p.f;
  state.k = p.k;
  if (state.map) {
    state.map = false;
    $("map").checked = false;
  }
  if (reseed) seed("drops");
  showRates();
};

let hashTimer;
const saveHash = () => {
  clearTimeout(hashTimer);
  hashTimer = setTimeout(() => {
    const h = state.preset
      ? `p=${state.preset}`
      : `f=${state.f.toFixed(4)}&k=${state.k.toFixed(4)}`;
    const pal = state.palette !== "coral" ? `&c=${state.palette}` : "";
    const map = state.map ? "&map=1" : "";
    history.replaceState(null, "", "#" + h + pal + map);
  }, 250);
};

const readHash = () => {
  const q = new URLSearchParams(location.hash.slice(1));
  const p = presets.find((p) => p.id === q.get("p"));
  if (p) {
    state.preset = p.id;
    state.f = p.f;
    state.k = p.k;
  } else if (q.get("f") && q.get("k")) {
    const f = parseFloat(q.get("f")), k = parseFloat(q.get("k"));
    if (f >= F_MIN && f <= F_MAX && k >= K_MIN && k <= K_MAX) {
      state.f = f;
      state.k = k;
      state.preset = "";
    }
  }
  if (palettes.some((p) => p.id === q.get("c"))) state.palette = q.get("c");
  state.map = q.get("map") === "1";
};

const download = () => {
  draw();
  canvas.toBlob((blob) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `reaction-diffusion-${state.preset || "f" + state.f.toFixed(4) + "-k" + state.k.toFixed(4)}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
};

const setupControls = () => {
  $("preset").innerHTML =
    presets.map((p) => `<option value="${p.id}">${p.name}</option>`).join("") +
    '<option value="" disabled>Custom</option>';
  $("palette").innerHTML = palettes.map((p) => `<option value="${p.id}">${p.name}</option>`).join("");
  $("palette").value = state.palette;
  $("feed").min = F_MIN;
  $("feed").max = F_MAX;
  $("kill").min = K_MIN;
  $("kill").max = K_MAX;
  $("map").checked = state.map;
  $("relief").checked = state.relief;
  $("speed").value = state.speed;
  $("brush").value = state.brush;

  $("preset").addEventListener("change", (e) => setPreset(e.target.value));
  $("feed").addEventListener("input", (e) => setRates(parseFloat(e.target.value), state.k));
  $("kill").addEventListener("input", (e) => setRates(state.f, parseFloat(e.target.value)));
  $("palette").addEventListener("change", (e) => {
    state.palette = e.target.value;
    saveHash();
  });
  $("speed").addEventListener("input", (e) => (state.speed = parseInt(e.target.value, 10)));
  $("brush").addEventListener("input", (e) => (state.brush = parseInt(e.target.value, 10)));
  $("relief").addEventListener("change", (e) => (state.relief = e.target.checked));
  $("map").addEventListener("change", (e) => {
    wake();
    state.map = e.target.checked;
    seed("noise");
    showRates();
  });
  $("btn-pause").addEventListener("click", () => {
    state.paused = !state.paused;
    $("btn-pause").textContent = state.paused ? "Play" : "Pause";
  });
  $("btn-reset").addEventListener("click", () => seed(state.map ? "noise" : "drops"));
  $("btn-clear").addEventListener("click", () => seed("none"));
  $("btn-png").addEventListener("click", download);
  document.addEventListener("keydown", (e) => {
    if (e.target.closest("input, select, textarea, button")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("btn-pause").click();
    }
  });
};

const initRD = () => {
  readHash();
  if (!initGL()) {
    $("rd-wrap").classList.add("no-gl");
    $("rd-msg").textContent =
      "This browser cannot run the simulation: it needs WebGL 2 with floating-point textures.";
    return;
  }
  setupControls();
  setupPainting();
  setupMap();
  seed(state.map ? "noise" : "drops");
  showRates();
  requestAnimationFrame(loop);
};
