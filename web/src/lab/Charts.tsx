import { useState } from "react";

// Round an axis out to a 1/2/5 x 10^k step so ticks read 0.1, 0.2 rather than 0.0916, 0.183.
export function niceAxis(y0: number, y1: number): [number, number, number] {
  const raw = (y1 - y0) / 3, mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  const step = (n <= 1 ? 1 : n <= 2 ? 2 : n <= 5 ? 5 : 10) * mag;
  return [Math.floor(y0 / step + 1e-9) * step, Math.ceil(y1 / step - 1e-9) * step, step];
}

// X ticks: round 1/2/5 steps (about five) plus the last x, which carries the unit (DS16-02, DS29-01).
// The first x is shown only when it is round: "43" (the first logged step) read as noise.
export function xTicks(x0: number, x1: number): number[] {
  if (!(x1 > x0)) return [x0];
  const raw = (x1 - x0) / 5, mag = 10 ** Math.floor(Math.log10(raw)), n = raw / mag;
  const step = Math.max(1, (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag); // x is a step or generation count: integers only
  const ticks = x0 % step === 0 ? [x0] : [];
  for (let k = Math.floor(x0 / step) + 1; k * step < x1; k++) if (x1 - k * step >= step) ticks.push(k * step); // a full step clear of the end label
  return [...ticks, x1];
}

/** Bottom-axis labels for xTicks: centred on their x, except the last, which ends at the axis and carries the unit. */
export function XAxis({ ticks: all, sx, y, label, fmt = String }: { ticks: number[]; sx: (x: number) => number; y: number; label: string; fmt?: (v: number) => string }) {
  // "1k" ran into "1.2k step" (DS30-09): drop a tick whose text would sit within 8 units of the end label.
  // shortcut: 5.5 units per character at the 10px tick size; measure with getBBox if labels get longer.
  const end = all[all.length - 1], endW = `${fmt(end)} ${label}`.length * 5.5;
  const ticks = all.filter((t, i) => i === all.length - 1 || sx(end) - sx(t) - fmt(t).length * 2.75 >= endW + 8);
  return <>{ticks.map((t, i) => <text key={i} x={sx(t)} y={y} className="tick"
    textAnchor={i === ticks.length - 1 ? "end" : "middle"}>{fmt(t)}{i === ticks.length - 1 ? ` ${label}` : ""}</text>)}</>;
}

// Hand-rolled single-series line chart with a crosshair tooltip. One chart
// per metric (small multiples) so there is never a second y-axis.
export function LineChart({ title, xs, ys, xLabel = "step", band, yRange, pct, chance }: {
  title: string; xs: number[]; ys: number[]; xLabel?: string;
  chance?: number; // dashed "chance" line for win rates, so 46% reads against 25% (DS33-04)
  pct?: boolean; // y is a 0..1 rate: label it as a percentage like the tables do
  band?: ([number, number] | null)[]; // optional per-point [lo, hi], e.g. a 95% CI
  yRange?: [number, number];
}) {
  const [hi, setHi] = useState<number | null>(null);
  const W = 360, H = 170, L = 44, R = 10, T = 12, B = 26;
  const pts = xs.map((x, i) => [x, ys[i], i] as [number, number, number]).filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
  if (pts.length === 0) return null;
  const x0 = Math.min(...pts.map((p) => p[0])), x1 = Math.max(...pts.map((p) => p[0]));
  // an auto axis also spans the band, so a CI never spills out of the plot (DS37-05 SE bands)
  const spanY = [...pts.map((p) => p[1]), ...(band ? pts.flatMap((p) => band[p[2]] ?? []).filter(Number.isFinite) : [])];
  let [y0, y1] = yRange ?? [Math.min(...spanY), Math.max(...spanY)];
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  let step: number;
  [y0, y1, step] = niceAxis(y0, y1);
  const sx = (x: number) => L + ((x - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (y: number) => T + (1 - (y - y0) / (y1 - y0)) * (H - T - B);
  const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(y).toFixed(1)}`).join("");
  const ticks = Array.from({ length: Math.round((y1 - y0) / step) + 1 }, (_, i) => y0 + i * step);
  const bandPts = band ? xs.map((x, i) => [x, band[i]] as const).filter(([x, b]) => Number.isFinite(x) && b) as [number, [number, number]][] : [];
  const bandD = bandPts.length > 1
    ? bandPts.map(([x, b], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)} ${sy(b[1]).toFixed(1)}`).join("")
      + [...bandPts].reverse().map(([x, b]) => `L${sx(x).toFixed(1)} ${sy(b[0]).toFixed(1)}`).join("") + "Z"
    : null;
  // 3 significant digits, no scientific notation: 0.0022, 147, 13.1M
  const fmt = (v: number) => (Math.abs(v) >= 1e6 ? +(v / 1e6).toFixed(1) + "M" : Math.abs(v) >= 1000 ? +(v / 1000).toFixed(1) + "k" : v === 0 ? "0" : String(+v.toPrecision(Math.abs(v) < 0.01 ? 2 : 3)));
  const fy = (v: number) => (pct ? +(v * 100).toFixed(Math.abs(v) < 0.1 && v !== 0 ? 1 : 0) + "%" : fmt(v));
  // DS36-06: a narrow axis (0.1538-0.1541) printed "0.154" on every tick; give ticks the step's decimals
  const tick = (v: number) => (pct || Math.abs(v) >= 1000 || step >= 1 ? fy(v) : String(+v.toFixed(Math.min(10, -Math.floor(Math.log10(step) + 1e-9)))));
  const onMove = (e: React.MouseEvent<SVGRectElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = x0 + ((e.clientX - r.left) / r.width) * (x1 - x0);
    let best = 0;
    pts.forEach((p, i) => { if (Math.abs(p[0] - x) < Math.abs(pts[best][0] - x)) best = i; });
    setHi(best);
  };
  const p = hi !== null ? pts[hi] : null;
  return (
    <figure className="chart">
      <figcaption>{title} <span className="muted">· last {fy(pts[pts.length - 1][1])}</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} over ${xLabel}`}>
        {ticks.map((t, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={L - 6} y={sy(t) + 4} className="tick" textAnchor="end">{tick(t)}</text>
          </g>
        ))}
        <XAxis ticks={xTicks(x0, x1)} sx={sx} y={H - 6} label={xLabel} fmt={fmt} />
        {bandD && <path d={bandD} className="band" />}
        {chance != null && <g data-testid="chance-line"><line x1={L} x2={W - R} y1={sy(chance)} y2={sy(chance)} className="chance" />
          <text x={W - R} y={sy(chance) - 3} className="tick" textAnchor="end">chance {fy(chance)}</text></g>}
        <path d={d} className="line" />
        {/* DS38-04: a one-point series (first panel eval) drew nothing; show the point and its CI whisker */}
        {pts.length === 1 && <g data-testid="single-point">{band?.[pts[0][2]] && <line x1={sx(pts[0][0])} x2={sx(pts[0][0])}
          y1={sy(band[pts[0][2]]![0])} y2={sy(band[pts[0][2]]![1])} className="line" strokeOpacity={0.45} />}
          <circle cx={sx(pts[0][0])} cy={sy(pts[0][1])} r={3.5} fill="var(--rope-hi)" /></g>}
        {p && (
          <g>
            <line x1={sx(p[0])} x2={sx(p[0])} y1={T} y2={H - B} className="crosshair" />
            <circle cx={sx(p[0])} cy={sy(p[1])} r={4.5} className="dot" />
            <text x={Math.min(sx(p[0]) + 8, W - (band ? 160 : 110))} y={T + 12} className="tip">{xLabel} {fmt(p[0])}: {fy(p[1])}{band?.[p[2]] ? ` [${fy(band[p[2]]![0])}, ${fy(band[p[2]]![1])}]` : ""}</text>
          </g>
        )}
        <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="transparent" onMouseMove={onMove} onMouseLeave={() => setHi(null)} />
      </svg>
    </figure>
  );
}

/** Win rate with a 95% CI whisker on a shared 0..1 scale, plus an optional
 *  "expected if equal strength" tick. */
export function RateBar({ rate, ci, expected, width = 160 }: {
  rate: number; ci?: [number, number] | null; expected?: number | null; width?: number;
}) {
  const w = width;
  const exp = typeof expected === "number" && Number.isFinite(expected) ? expected : null;
  return (
    <svg width={w} height={18} viewBox={`0 0 ${w} 18`} className="ratebar"
      aria-label={`win rate ${(rate * 100).toFixed(1)}%${exp !== null ? `, expected if equal ${(exp * 100).toFixed(0)}%` : ""}`}>
      <rect x={0} y={7} width={w} height={4} rx={2} className="track" />
      <rect x={0} y={5} width={Math.max(2, rate * w)} height={8} rx={4} className="fill" />
      {ci && <line x1={ci[0] * w} x2={ci[1] * w} y1={9} y2={9} className="ci" />}
      {ci && [ci[0], ci[1]].map((c, i) => <line key={i} x1={c * w} x2={c * w} y1={4} y2={14} className="ci" />)}
      {exp !== null && <line x1={exp * w} x2={exp * w} y1={1} y2={17} className="expected"><title>expected if equal: {(exp * 100).toFixed(0)}%</title></line>}
    </svg>
  );
}

/** Count histogram with fixed-width bins. Hover a bar for its range and count. */
export function Histogram({ title, values, xLabel = "", bin }: {
  title: string; values: number[]; xLabel?: string; bin?: number;
}) {
  const [hi, setHi] = useState<number | null>(null);
  const vs = values.filter(Number.isFinite);
  if (!vs.length) return null;
  const lo = Math.min(...vs), top = Math.max(...vs);
  // ~20 bins on a "nice" width (1, 2, 5, 10, 20, 25, 50...).
  const raw = Math.max(1, (top - lo) / 20);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const bw = bin ?? [1, 2, 2.5, 5, 10].map((m) => m * mag).find((m) => m >= raw) ?? 10 * mag;
  const start = Math.floor(lo / bw) * bw;
  const nb = Math.max(1, Math.floor((top - start) / bw) + 1);
  const counts = Array(nb).fill(0);
  vs.forEach((v) => counts[Math.min(nb - 1, Math.floor((v - start) / bw))]++);
  const W = 360, H = 170, L = 36, R = 10, T = 12, B = 26;
  const cmax = Math.max(...counts);
  const bwPx = (W - L - R) / nb;
  const sy = (c: number) => T + (1 - c / cmax) * (H - T - B);
  const range = (i: number) => `${+(start + i * bw).toFixed(2)}–${+(start + (i + 1) * bw).toFixed(2)}`;
  return (
    <figure className="chart histogram">
      <figcaption title={`${title} · n ${vs.length}`}>{title} <span className="muted">· n {vs.length}</span></figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} histogram`} onMouseLeave={() => setHi(null)}>
        {[0, cmax].map((t, i) => (
          <g key={i}>
            <line x1={L} x2={W - R} y1={sy(t)} y2={sy(t)} className="grid" />
            <text x={L - 6} y={sy(t) + 4} className="tick" textAnchor="end">{t}</text>
          </g>
        ))}
        {counts.map((c, i) => (
          <rect key={i} x={L + i * bwPx + 0.5} y={sy(c)} width={Math.max(1, bwPx - 1)} height={H - B - sy(c)}
            className={"bar" + (hi === i ? " on" : "")} onMouseEnter={() => setHi(i)}>
            <title>{range(i)} {xLabel}: {c}</title>
          </rect>
        ))}
        <text x={L} y={H - 6} className="tick">{+start.toFixed(2)}</text>
        <text x={W - R} y={H - 6} className="tick" textAnchor="end">{+(start + nb * bw).toFixed(2)} {xLabel}</text>
        {hi !== null && <text x={Math.min(L + hi * bwPx + 4, W - 120)} y={T + 12} className="tip">{range(hi)} {xLabel}: {counts[hi]}</text>}
      </svg>
    </figure>
  );
}
