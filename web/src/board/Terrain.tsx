// Original illustrated terrain tiles, drawn in unit-hex space (pointy top,
// corner radius 1, centre 0,0). The middle (r < 0.38) stays clear for the
// number token, so decoration sits on a ring around it.
import React from "react";

export const HEX_PTS = Array.from({ length: 6 }, (_, i) => {
  const a = ((60 * i - 30) * Math.PI) / 180;
  return [Math.cos(a), Math.sin(a)];
});
export const hexPoints = (r = 1) => HEX_PTS.map(([x, y]) => `${(x * r).toFixed(3)},${(y * r).toFixed(3)}`).join(" ");

const ring = (n: number, rad: number, phase = 0) =>
  Array.from({ length: n }, (_, i) => {
    const a = phase + (i / n) * Math.PI * 2;
    return [Math.cos(a) * rad, Math.sin(a) * rad] as [number, number];
  });

const Pine = ({ x, y, s = 1 }: { x: number; y: number; s?: number }) => (
  <g transform={`translate(${x} ${y}) scale(${s})`}>
    <rect x={-0.025} y={0.05} width={0.05} height={0.08} fill="#4a3220" />
    <path d="M0 -0.22 L0.12 -0.04 L0.06 -0.04 L0.15 0.08 L-0.15 0.08 L-0.06 -0.04 L-0.12 -0.04 Z" fill="#1d4a2b" />
    <path d="M0 -0.22 L0.12 -0.04 L0.06 -0.04 L0.15 0.08 L0 0.08 Z" fill="#2f6a3e" />
  </g>
);

const Sheep = ({ x, y, flip = false }: { x: number; y: number; flip?: boolean }) => (
  <g transform={`translate(${x} ${y}) scale(${flip ? -1 : 1} 1)`}>
    <rect x={-0.06} y={0.02} width={0.025} height={0.06} fill="#3a3a33" />
    <rect x={0.035} y={0.02} width={0.025} height={0.06} fill="#3a3a33" />
    <ellipse cx={0} cy={0} rx={0.1} ry={0.065} fill="#f7f4ea" stroke="#c9c2ad" strokeWidth={0.012} />
    <circle cx={-0.04} cy={-0.04} r={0.035} fill="#f7f4ea" />
    <circle cx={0.04} cy={-0.045} r={0.03} fill="#f7f4ea" />
    <ellipse cx={0.11} cy={-0.025} rx={0.038} ry={0.03} fill="#33312b" />
  </g>
);

const Peak = ({ x, y, s = 1 }: { x: number; y: number; s?: number }) => (
  <g transform={`translate(${x} ${y}) scale(${s})`}>
    <path d="M-0.22 0.1 L-0.02 -0.2 L0.2 0.1 Z" fill="#5c6878" />
    <path d="M-0.02 -0.2 L0.2 0.1 L0.03 0.1 Z" fill="#46505e" />
    <path d="M-0.02 -0.2 L-0.09 -0.1 L-0.04 -0.08 L0.0 -0.12 L0.05 -0.09 L0.05 -0.13 Z" fill="#eef2f5" />
  </g>
);

const BrickStack = ({ x, y }: { x: number; y: number }) => (
  <g transform={`translate(${x} ${y})`}>
    {[[-0.1, 0], [0.01, 0], [-0.045, -0.055]].map(([bx, by], i) => (
      <rect key={i} x={bx} y={by} width={0.1} height={0.05} rx={0.008} fill="#8f3a1d" stroke="#6b2a14" strokeWidth={0.01} />
    ))}
  </g>
);

const Sheaf = ({ x, y }: { x: number; y: number }) => (
  <g transform={`translate(${x} ${y})`} stroke="#8a611b" strokeWidth={0.014} strokeLinecap="round" fill="none">
    <path d="M0 0.1 L0 -0.1 M0 0.1 L-0.05 -0.08 M0 0.1 L0.05 -0.08" />
    <ellipse cx={0} cy={-0.12} rx={0.018} ry={0.04} fill="#f3d27a" stroke="#b78526" />
    <ellipse cx={-0.055} cy={-0.1} rx={0.016} ry={0.035} fill="#f3d27a" stroke="#b78526" transform="rotate(-20 -0.055 -0.1)" />
    <ellipse cx={0.055} cy={-0.1} rx={0.016} ry={0.035} fill="#f3d27a" stroke="#b78526" transform="rotate(20 0.055 -0.1)" />
  </g>
);

export function TerrainDefs() {
  const clip = <clipPath id="hexclip" clipPathUnits="userSpaceOnUse"><polygon points={hexPoints(0.97)} /></clipPath>;
  return (
    <defs>
      {clip}
      <radialGradient id="g-wood" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#4c8a55" /><stop offset="1" stopColor="#285a35" /></radialGradient>
      <radialGradient id="g-brick" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#d27a4e" /><stop offset="1" stopColor="#a3482a" /></radialGradient>
      <radialGradient id="g-sheep" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#b5dc79" /><stop offset="1" stopColor="#7fae4a" /></radialGradient>
      <radialGradient id="g-wheat" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#f5d26a" /><stop offset="1" stopColor="#d59f2c" /></radialGradient>
      <radialGradient id="g-ore" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#a4aeba" /><stop offset="1" stopColor="#6c7887" /></radialGradient>
      <radialGradient id="g-desert" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#efdcab" /><stop offset="1" stopColor="#d3b878" /></radialGradient>
      <radialGradient id="g-sea" cx="50%" cy="50%" r="70%"><stop offset="0" stopColor="#2a7f99" /><stop offset="0.55" stopColor="#1b5d78" /><stop offset="1" stopColor="#0f3d55" /></radialGradient>
      <pattern id="p-waves" width="0.9" height="0.5" patternUnits="userSpaceOnUse">
        <path d="M0 0.25 q0.225 -0.12 0.45 0 t0.45 0" fill="none" stroke="#7fc4d6" strokeOpacity="0.16" strokeWidth="0.03" />
      </pattern>
      <pattern id="p-furrows" width="0.16" height="0.16" patternUnits="userSpaceOnUse" patternTransform="rotate(28)">
        <rect width="0.16" height="0.16" fill="none" />
        <line x1="0" y1="0.08" x2="0.16" y2="0.08" stroke="#b98523" strokeOpacity="0.45" strokeWidth="0.025" />
      </pattern>
      <pattern id="p-courses" width="0.2" height="0.12" patternUnits="userSpaceOnUse">
        <path d="M0 0.06 H0.2 M0.1 0 V0.06 M0 0.06 V0.12" stroke="#8a3a1e" strokeOpacity="0.28" strokeWidth="0.012" fill="none" />
      </pattern>
      <pattern id="p-grass" width="0.22" height="0.2" patternUnits="userSpaceOnUse">
        <path d="M0.05 0.15 l0.02 -0.06 l0.02 0.06 M0.15 0.06 l0.015 -0.05 l0.015 0.05" stroke="#5f8f34" strokeOpacity="0.5" strokeWidth="0.012" fill="none" />
      </pattern>
      <pattern id="p-scree" width="0.24" height="0.2" patternUnits="userSpaceOnUse">
        <circle cx="0.05" cy="0.05" r="0.018" fill="#56606c" fillOpacity="0.35" />
        <circle cx="0.17" cy="0.14" r="0.012" fill="#56606c" fillOpacity="0.35" />
      </pattern>
      <pattern id="p-dunes" width="0.5" height="0.24" patternUnits="userSpaceOnUse">
        <path d="M0 0.16 q0.125 -0.1 0.25 0 t0.25 0" fill="none" stroke="#b8975a" strokeOpacity="0.45" strokeWidth="0.018" />
      </pattern>
      <pattern id="p-forest" width="0.2" height="0.2" patternUnits="userSpaceOnUse">
        <circle cx="0.1" cy="0.1" r="0.018" fill="#1d4a2b" fillOpacity="0.35" />
      </pattern>
      {/* per-terrain tile symbols */}
      <g id="t-wood">
        <polygon points={hexPoints(0.97)} fill="url(#g-wood)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-forest)" />
        <g clipPath="url(#hexclip)">
          {ring(9, 0.66, 0.2).map(([x, y], i) => <Pine key={i} x={x} y={y} s={i % 2 ? 1.05 : 0.85} />)}
          {ring(5, 0.42, 1.1).map(([x, y], i) => <Pine key={"i" + i} x={x * 1.05} y={y * 1.15 - 0.05} s={0.7} />)}
        </g>
      </g>
      <g id="t-brick">
        <polygon points={hexPoints(0.97)} fill="url(#g-brick)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-courses)" />
        <g clipPath="url(#hexclip)">
          <path d="M-1 0.55 q0.5 -0.3 1 0 t1 0 V1 H-1 Z" fill="#b65c35" opacity="0.7" />
          <path d="M-1 -0.45 q0.5 -0.25 1 0 t1 0 V-1 H-1 Z" fill="#c76a40" opacity="0.6" />
          {ring(6, 0.62, 0.5).map(([x, y], i) => <BrickStack key={i} x={x} y={y} />)}
        </g>
      </g>
      <g id="t-sheep">
        <polygon points={hexPoints(0.97)} fill="url(#g-sheep)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-grass)" />
        <g clipPath="url(#hexclip)">
          <path d="M-0.75 -0.15 Q-0.6 -0.25 -0.45 -0.15" stroke="#6b4d2e" strokeWidth="0.02" fill="none" />
          {ring(5, 0.6, 0.3).map(([x, y], i) => <Sheep key={i} x={x} y={y} flip={i % 2 === 0} />)}
        </g>
      </g>
      <g id="t-wheat">
        <polygon points={hexPoints(0.97)} fill="url(#g-wheat)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-furrows)" />
        <g clipPath="url(#hexclip)">
          {ring(7, 0.63, 0).map(([x, y], i) => <Sheaf key={i} x={x} y={y} />)}
        </g>
      </g>
      <g id="t-ore">
        <polygon points={hexPoints(0.97)} fill="url(#g-ore)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-scree)" />
        <g clipPath="url(#hexclip)">
          {ring(6, 0.6, 0.0).map(([x, y], i) => <Peak key={i} x={x} y={y} s={i % 2 ? 1.2 : 0.95} />)}
        </g>
      </g>
      <g id="t-desert">
        <polygon points={hexPoints(0.97)} fill="url(#g-desert)" />
        <polygon points={hexPoints(0.97)} fill="url(#p-dunes)" />
        <g clipPath="url(#hexclip)">
          <g transform="translate(0.45 0.35)" fill="#6f8f4a">
            <rect x="-0.03" y="-0.18" width="0.06" height="0.24" rx="0.03" />
            <rect x="-0.11" y="-0.1" width="0.05" height="0.1" rx="0.025" />
            <rect x="0.06" y="-0.14" width="0.05" height="0.09" rx="0.025" />
          </g>
          <ellipse cx="-0.45" cy="0.4" rx="0.09" ry="0.05" fill="#a98d5c" />
          <ellipse cx="-0.35" cy="-0.5" rx="0.06" ry="0.035" fill="#a98d5c" />
        </g>
      </g>
    </defs>
  );
}

// Tiny resource glyphs for harbor tokens and the hand (same drawing language).
export function ResourceGlyph({ r, size = 1 }: { r: number; size?: number }) {
  const g = [
    <Pine x={0} y={0.03} s={1.3} />,
    <g transform="scale(1.5)"><BrickStack x={0.005} y={0.02} /></g>,
    <Sheep x={-0.01} y={0.01} />,
    <Sheaf x={0} y={0.04} />,
    <Peak x={0} y={0.02} s={0.9} />,
  ][r];
  return <g transform={`scale(${size})`}>{g}</g>;
}

export const TERRAIN_SYMBOL: Record<string, string> = {
  wood: "#t-wood", brick: "#t-brick", sheep: "#t-sheep", wheat: "#t-wheat", ore: "#t-ore", desert: "#t-desert",
};

export const RES_COLOR = ["#2f6a3e", "#b8572f", "#8dbe57", "#e2b33c", "#7b8896"];

export function ResIcon({ r, size = 22 }: { r: number; size?: number }) {
  return (
    <svg className="res-icon" width={size} height={size} viewBox="-0.3 -0.3 0.6 0.6" aria-hidden>
      <circle r={0.29} fill={RES_COLOR[r]} opacity={0.25} />
      <ResourceGlyph r={r} size={1.3} />
    </svg>
  );
}

export default React.memo(TerrainDefs);
