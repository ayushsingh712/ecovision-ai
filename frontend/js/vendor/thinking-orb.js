/*
 * thinking-orbs — "working" state (particles on tilted orbits).
 * Vanilla ES-module canvas port of the `orbits` mode from
 * https://github.com/Jakubantalik/thinking-orbs
 *
 * MIT License — Copyright (c) 2026 Jakub Antalik.
 * Full license text: ./thinking-orb.LICENSE
 *
 * Behaviour follows the upstream component: tuned 64px preset, shared
 * ink conventions, capped device-pixel-ratio, a static frame for
 * prefers-reduced-motion, and pausing while offscreen or when the tab
 * is hidden. The ink follows this project's data-theme attribute.
 */

const OPTS = {
  orbitN: 12,
  ghostN: 40,
  ghostR: 0.9,
  ghostA: 0.5,
  particles: 3,
  partR: 1.2,
  partRDepth: 1.6,
  rsPow: 0.6,
  rMin: 0.3,
};

// Baked speed for the 64px `working` preset.
const BASE_SPEED = 1.885;
const STILL_FRAME = 0.6;

function hashD(a, b) {
  const h = Math.sin(a * 12.9898 + b * 78.233) * 43758.5453;
  return h - Math.floor(h);
}

function makeProj(yaw, tilt, cx, cy, scale) {
  const st = Math.sin(tilt);
  const ct = Math.cos(tilt);
  const sy = Math.sin(yaw);
  const cyw = Math.cos(yaw);
  return (x, y, z) => {
    const x1 = x * cyw + z * sy;
    const z1 = -x * sy + z * cyw;
    const y1 = y * ct - z1 * st;
    const z2 = y * st + z1 * ct;
    return [cx + x1 * scale, cy - y1 * scale, z2];
  };
}

function radiusScale(size, pow) {
  return (size / 300) ** pow;
}

function finalizeDots(dots, rMin) {
  const visible = [];
  for (const dot of dots) {
    if ((dot.a ?? 1) < 0.02) continue;
    dot.r = Math.max(rMin, dot.r);
    visible.push(dot);
  }
  visible.sort((a, b) => a.z - b.z);
  return visible;
}

function frameOrbits(size, t, o) {
  const cx = size / 2;
  const cy = size / 2;
  const R = (size / 2) * 0.82;
  const project = makeProj(t * 0.12, 0.3, cx, cy, 1);
  const rs = radiusScale(size, o.rsPow ?? 0.6);

  const dots = [];
  const orbitN = o.orbitN ?? 12;
  const ghostN = o.ghostN ?? 40;
  const particles = o.particles ?? 3;

  for (let orb = 0; orb < orbitN; orb += 1) {
    const h1 = hashD(orb, 1.7);
    const h2 = hashD(orb, 5.2);
    const h3 = hashD(orb, 8.9);
    const ro = R * (0.45 + 0.52 * h1);
    const th = h1 * 2 * Math.PI;
    const phi = Math.acos(2 * h2 - 1);
    const nx = Math.sin(phi) * Math.cos(th);
    const ny = Math.cos(phi);
    const nz = Math.sin(phi) * Math.sin(th);
    let ux = -ny;
    let uy = nx;
    const uz = 0;
    const ul = Math.max(1e-6, Math.sqrt(ux * ux + uy * uy));
    ux /= ul;
    uy /= ul;
    const vx = ny * uz - nz * uy;
    const vy = nz * ux - nx * uz;
    const vz = nx * uy - ny * ux;
    const speed = (0.25 + 0.55 * h3) * (h3 > 0.5 ? 1 : -1);

    for (let k = 0; k < ghostN; k += 1) {
      const a = (k / ghostN) * 2 * Math.PI;
      const [px, py, z] = project(
        (ux * Math.cos(a) + vx * Math.sin(a)) * ro,
        (uy * Math.cos(a) + vy * Math.sin(a)) * ro,
        (uz * Math.cos(a) + vz * Math.sin(a)) * ro,
      );
      const depth = (z / ro + 1) / 2;
      dots.push({
        x: px,
        y: py,
        z,
        r: (o.ghostR ?? 0.9) * rs,
        white: 0.72,
        a: (o.ghostA ?? 0.5) * (0.4 + 0.6 * depth),
      });
    }

    for (let m = 0; m < particles; m += 1) {
      const a = t * speed + (m / particles) * 2 * Math.PI + h2 * 6;
      const [px, py, z] = project(
        (ux * Math.cos(a) + vx * Math.sin(a)) * ro,
        (uy * Math.cos(a) + vy * Math.sin(a)) * ro,
        (uz * Math.cos(a) + vz * Math.sin(a)) * ro,
      );
      const depth = (z / ro + 1) / 2;
      dots.push({
        x: px,
        y: py,
        z,
        r: ((o.partR ?? 1.2) + (o.partRDepth ?? 1.6) * depth) * rs,
        white: 0.3 - 0.22 * depth,
      });
    }
  }

  return finalizeDots(dots, o.rMin);
}

function paint(ctx, dots, dark) {
  for (const dot of dots) {
    const alpha = dot.a ?? 1;
    const ink = Math.min(1, Math.max(0, dot.white));
    const channel = Math.round((dark ? 1 - ink : ink) * 255);
    ctx.fillStyle = `rgba(${channel},${channel},${channel},${alpha})`;
    ctx.beginPath();
    ctx.arc(dot.x, dot.y, dot.r, 0, Math.PI * 2);
    ctx.fill();
  }
}

function prefersReducedMotion() {
  return typeof window.matchMedia === "function"
    && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function isDark() {
  return document.documentElement.getAttribute("data-theme") !== "light";
}

export function mountThinkingOrb(canvas) {
  const size = Number(canvas.dataset.orbSize) || 64;
  const speed = Number(canvas.dataset.orbSpeed) || 1;
  const context = canvas.getContext("2d");
  if (!context) return () => {};

  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.round(size * dpr);
  canvas.height = Math.round(size * dpr);
  canvas.style.width = `${size}px`;
  canvas.style.height = `${size}px`;

  const motionQuery = typeof window.matchMedia === "function"
    ? window.matchMedia("(prefers-reduced-motion: reduce)")
    : null;
  let reduced = prefersReducedMotion();
  const effectiveSpeed = BASE_SPEED * speed;
  let raf = 0;
  let running = false;
  let onScreen = false;

  const render = (time) => {
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, size, size);
    paint(context, frameOrbits(size, time, OPTS), isDark());
  };

  const paintStill = () => render(STILL_FRAME);

  const loop = () => {
    render((performance.now() / 1000) * effectiveSpeed);
    if (running) raf = requestAnimationFrame(loop);
  };

  const start = () => {
    if (running || reduced || !onScreen || document.visibilityState === "hidden") return;
    running = true;
    raf = requestAnimationFrame(loop);
  };

  const stop = () => {
    running = false;
    cancelAnimationFrame(raf);
  };

  paintStill();

  let observer = null;
  if (!reduced) {
    if ("IntersectionObserver" in window) {
      observer = new IntersectionObserver(([entry]) => {
        onScreen = entry.isIntersecting;
        if (onScreen) start();
        else stop();
      });
      observer.observe(canvas);
    } else {
      onScreen = true;
      start();
    }
  }

  const onVisibility = () => {
    if (document.visibilityState === "hidden") stop();
    else start();
  };

  const onThemeChange = () => {
    if (!running) paintStill();
  };

  const onMotionChange = (event) => {
    reduced = event.matches;
    if (reduced) {
      stop();
      paintStill();
    } else {
      start();
    }
  };

  document.addEventListener("visibilitychange", onVisibility);
  document.addEventListener("ecovision:themechange", onThemeChange);
  if (motionQuery) {
    if (motionQuery.addEventListener) motionQuery.addEventListener("change", onMotionChange);
    else if (motionQuery.addListener) motionQuery.addListener(onMotionChange);
  }

  return () => {
    stop();
    if (observer) observer.disconnect();
    document.removeEventListener("visibilitychange", onVisibility);
    document.removeEventListener("ecovision:themechange", onThemeChange);
    if (motionQuery) {
      if (motionQuery.removeEventListener) motionQuery.removeEventListener("change", onMotionChange);
      else if (motionQuery.removeListener) motionQuery.removeListener(onMotionChange);
    }
  };
}

document.querySelectorAll("[data-thinking-orb]").forEach((canvas) => {
  mountThinkingOrb(canvas);
});
