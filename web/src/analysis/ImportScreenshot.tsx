// "Import screenshot": upload a board image, preview what was detected over it,
// then load the position into the editor with low-confidence fields outlined.
// The server reads only the uploaded file (docs/VISION_IMPORT.md).
import { useEffect, useMemo, useState } from "react";
import { ApiError, PositionSpec, Topology } from "../api";
import { Modal } from "../game/Dialogs";
import { PLAYER_COLORS } from "../rules";
import { resText } from "../lab/DecisionInspector";
import type { Mark } from "./BoardMarks";

const MAX_BYTES = 8_000_000; // catan/server/analysis.py MAX_IMAGE_BYTES
const TYPES = ["image/png", "image/jpeg", "image/webp"];
const TERR_SHORT: Record<string, string> = { wood: "Wd", brick: "Br", sheep: "Sh", wheat: "Wh", ore: "Or", desert: "De" };

export interface ImportResult {
  position: PositionSpec;
  errors: (string | { message: string })[];
  confidence: { hex_terrain: number[]; hex_number: number[]; ports: number[]; robber: number; vertices: number[]; edges: number[]; lattice: number };
  low_confidence: { hexes: number[]; vertices: number[]; edges: number[]; fields: string[] };
  lattice: { cx: number; cy: number; r: number; width: number; height: number } | null;
  profile: string;
  backend?: "llm" | "color";
  backend_note?: string;
  colors?: Record<string, number>;
  repaired?: boolean;
  usage?: { calls: number; total_tokens: number; usd: number | null };
  suggestion: { policy: string; candidates: { label: string; policy_prob: number | null; rank: number }[] } | null;
  suggestion_note?: string;
}

type Backend = "auto" | "llm" | "color";

async function upload(file: File, backend: Backend): Promise<ImportResult> {
  const fd = new FormData();
  fd.append("image", file);
  fd.append("backend", backend);
  const r = await fetch("/api/analysis/import-image", { method: "POST", body: fd });
  const j = await r.json().catch(() => ({ detail: r.statusText }));
  if (!r.ok) throw new ApiError(r.status, typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail));
  return j;
}

/** Low-confidence fields as board marks for the editor. */
export function lowMarks(r: ImportResult): Mark[] {
  const l = r.low_confidence;
  return [...l.hexes.map((id) => ({ kind: "h" as const, id, tone: "low" as const })),
    ...l.vertices.map((id) => ({ kind: "v" as const, id, tone: "low" as const })),
    ...l.edges.map((id) => ({ kind: "e" as const, id, tone: "low" as const }))];
}

export default function ImportScreenshot({ topo, onLoad, onClose }: { topo: Topology; onLoad: (r: ImportResult) => void; onClose: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [res, setRes] = useState<ImportResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [backend, setBackend] = useState<Backend>("auto");
  const url = useMemo(() => file && URL.createObjectURL(file), [file]);
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);

  const pick = async (f: File | undefined) => {
    setRes(null); setErr(null); setFile(null);
    if (!f) return;
    if (!TYPES.includes(f.type)) { setErr("Choose a PNG, JPEG or WebP screenshot."); return; }
    if (f.size > MAX_BYTES) { setErr(`The file is ${(f.size / 1e6).toFixed(1)} MB; the limit is 8 MB.`); return; }
    setFile(f); setBusy(true);
    try { setRes(await upload(f, backend)); } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const low = res && new Set([...res.low_confidence.hexes.map((h) => "h" + h), ...res.low_confidence.vertices.map((v) => "v" + v), ...res.low_confidence.edges.map((e) => "e" + e)]);
  const L = res?.lattice;
  const P = (x: number, y: number) => L ? [L.cx + L.r * x, L.cy + L.r * y] : [0, 0];
  const pos = res?.position;

  return (
    <Modal title="Import a screenshot" wide onClose={onClose}>
      <p className="muted small">
        For post-game review and private games where everyone agreed. It reads the board, the robber, harbors and pieces. With Gemini it
        also reads the hand, card counts and the turn when they are visible; check them in the editor.
      </p>
      <p className="small">
        <label>Reader{" "}
          <select value={backend} onChange={(e) => setBackend(e.target.value as Backend)} aria-label="Screenshot reader" data-testid="shot-backend">
            <option value="auto">Auto (Gemini if a key is set)</option>
            <option value="llm">Gemini</option>
            <option value="color">Colour tables (offline)</option>
          </select>
        </label>{" "}
        <input type="file" accept={TYPES.join(",")} aria-label="Screenshot file" data-testid="shot-file" onChange={(e) => pick(e.target.files?.[0])} />{" "}
        <span className="muted" data-testid="shot-privacy">Screenshots are sent to Google Gemini when a key is configured.</span>
      </p>
      {busy && <p role="status">Reading the board…</p>}
      {err && <p className="error small" role="alert">Can't import: {err}</p>}
      {url && (
        <div className="an-shot" data-testid="shot-preview">
          <img src={url} alt="Uploaded screenshot" />
          {res && L && pos && (
            <svg viewBox={`0 0 ${L.width} ${L.height}`} aria-label="Detected board overlay">
              {topo.hexes.map((h) => {
                const [x, y] = P(h.x, h.y);
                const pts = Array.from({ length: 6 }, (_, i) => { const a = (Math.PI / 180) * (60 * i - 30); return P(h.x + Math.cos(a) * 0.95, h.y + Math.sin(a) * 0.95).join(","); }).join(" ");
                const n = pos.layout.hex_number[h.id];
                return (
                  <g key={h.id} className={low!.has("h" + h.id) ? "low" : ""}>
                    <polygon points={pts} />
                    <text x={x} y={y - L.r * 0.45} fontSize={L.r * 0.26}>{TERR_SHORT[pos.layout.hex_terrain[h.id]]}{n ? ` ${n}` : ""}</text>
                    {pos.robber === h.id && <text x={x} y={y + L.r * 0.62} fontSize={L.r * 0.22}>robber</text>}
                  </g>
                );
              })}
              {pos.edge_owner.map((o, e) => {
                if (o < 0) return null;
                const [a, b] = topo.edges[e].v.map((i) => P(topo.vertices[i].x, topo.vertices[i].y));
                return <line key={e} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={PLAYER_COLORS[o]} strokeWidth={L.r * 0.07} className={low!.has("e" + e) ? "low" : ""} />;
              })}
              {pos.vertex_owner.map((o, v) => {
                const [x, y] = P(topo.vertices[v].x, topo.vertices[v].y);
                if (o < 0) return low!.has("v" + v) ? <circle key={v} cx={x} cy={y} r={L.r * 0.16} className="low" fill="none" /> : null;
                return <circle key={v} cx={x} cy={y} r={L.r * (pos.vertex_level[v] === 2 ? 0.2 : 0.13)} fill={PLAYER_COLORS[o]} className={"pc" + (low!.has("v" + v) ? " low" : "")} />;
              })}
              {pos.layout.port_edges.map((e, i) => {
                const [a, b] = topo.edges[e].v.map((j) => topo.vertices[j]);
                const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2, k = 1 + 0.62 / Math.hypot(mx, my);
                const [x, y] = P(mx * k, my * k);
                return <text key={"p" + i} x={x} y={y} fontSize={L.r * 0.22} className="port">{pos.layout.port_types[i]}</text>;
              })}
            </svg>
          )}
        </div>
      )}
      {res && (
        <div className="an-shot-sum small" data-testid="shot-summary">
          <p>
            {res.backend === "llm" ? <>Read by <strong>{res.profile}</strong>{res.repaired ? " (after one repair)" : ""}</> : <>Profile <strong>{res.profile}</strong> · board fit {(100 * res.confidence.lattice).toFixed(0)}%</>} ·{" "}
            {res.low_confidence.hexes.length + res.low_confidence.vertices.length + res.low_confidence.edges.length} low-confidence board fields (outlined) ·{" "}
            {res.errors.length ? `${res.errors.length} problem${res.errors.length === 1 ? "" : "s"} to fix` : "position valid"}
          </p>
          {res.backend_note && <p className="muted">{res.backend_note}</p>}
          {res.usage && <p className="muted" data-testid="shot-usage">{res.usage.calls} Gemini call{res.usage.calls === 1 ? "" : "s"}, {res.usage.total_tokens} tokens
            {res.usage.usd != null ? `, about $${res.usage.usd.toFixed(4)}` : ""}</p>}
          {res.colors && <p className="muted">Seats by colour: {Object.entries(res.colors).map(([c, s]) => `${s}: ${c}`).join(" · ")}. Set "Analyze as" to your seat.</p>}
          {res.suggestion ? (
            <p data-testid="shot-suggestion">Suggested by {res.suggestion.policy} (quick):{" "}
              {res.suggestion.candidates.map((c) => `${c.rank}. ${resText(c.label)}${c.policy_prob != null ? ` (${(100 * c.policy_prob).toFixed(0)}%)` : ""}`).join(" · ")}
            </p>
          ) : res.suggestion_note && <p className="muted">{res.suggestion_note}</p>}
          {res.suggestion && res.suggestion_note && <p className="muted">{res.suggestion_note}</p>}
        </div>
      )}
      <div className="modal-actions">
        <button className="btn" onClick={onClose}>Cancel</button>
        <button className="btn primary" disabled={!res} onClick={() => res && onLoad(res)} data-testid="shot-load">Load into editor</button>
      </div>
    </Modal>
  );
}
