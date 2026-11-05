// Analysis Mode: live board editor + engine analysis + post-game review.
// Routes: #/analysis, #/analysis/game/:id[/:index], #/analysis/file/:name[/:index]
import { useCallback, useEffect, useMemo, useState, useSyncExternalStore } from "react";
import { api, Action, Candidate, GameAnalysis, GameEvent, PositionSpec, Topology, ValidateResult, View } from "../api";
import type { Targets } from "../board/Board";
import TouchBoard from "./TouchBoard";
import QuickUpdate, { Armed } from "./QuickUpdate";
import * as Q from "./quick";
import { go } from "../App";
import { Modal } from "../game/Dialogs";
import { useAgents } from "../Setup";
import { describeEvent, PHASE_LABEL, PLAYER_COLORS, PLAYER_NAMES } from "../rules";
import AnalysisPanel, { DEFAULT_SETTINGS, Results, Settings } from "./AnalysisPanel";
import BoardMarks, { Mark } from "./BoardMarks";
import { resText } from "../lab/DecisionInspector";
import { analysis, useBackendMode } from "./client";
import EditorPanel, { Brush, NUMBER_BRUSH, PHASE_NAME, Tool, TOOLS } from "./EditorPanel";
import { readAutosave, readSaved, Saved, useAutosave, useHistory, writeSaved } from "./history";
import { setMockTopology } from "./mock";
import Review, { EvalGraph } from "./Review";
import ImportScreenshot, { ImportResult, lowMarks } from "./ImportScreenshot";
import { actionTarget, actorOf, emptyPosition, errorTargets, normalize, policySpec, PORT_KINDS, randomPosition, specFromView, TERRAINS, toWire } from "./spec";
import "./analysis.css";
import "./compact.css";

const SETTINGS = "catan.analysis.settings.v1";
type Tab = "board" | "live" | "edit" | "analysis";
function useMedia(q: string) {
  return useSyncExternalStore((f) => { const m = matchMedia(q); m.addEventListener("change", f); return () => m.removeEventListener("change", f); }, () => matchMedia(q).matches);
}
type Branch = { nodes: { position: PositionSpec; label: string; events: GameEvent[]; done: boolean }[]; at: number };
type Src = { kind: "game" | "replay"; ref: string; index: number; spec: PositionSpec; actor: number | null };

export default function AnalysisPage({ topo, route }: { topo: Topology; route: string[] }) {
  setMockTopology(topo);
  const [kindRaw, refId, idxRaw] = route;
  const srcKind = kindRaw === "game" ? "game" : kindRaw === "file" ? "replay" : null;
  // #/analysis/import/<base64url {position, low_confidence}>: sent by the Chrome advisor (tools/chrome-advisor/)
  const [shot] = useState<ImportResult | null>(() => {
    if (kindRaw !== "import" || !refId) return null;
    try { return JSON.parse(atob(refId.replace(/-/g, "+").replace(/_/g, "/"))); } catch { return null; }
  });
  useEffect(() => { if (kindRaw === "import") history.replaceState(null, "", "#/analysis"); }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const hist = useHistory(() => {
    if (shot) { try { return normalize(shot.position, topo); } catch { /* bad link: fall through */ } }
    const saved = readAutosave();
    try { if (saved && !srcKind) return normalize(saved, topo); } catch { /* corrupt autosave: start fresh */ }
    return randomPosition(topo);
  });
  useAutosave(hist.spec);
  const { agents, checkpoints } = useAgents();
  const mode = useBackendMode();

  const [tool, setTool] = useState<Tool | null>("settlement");
  const [seat, setSeat] = useState(0);
  const [brush, setBrush] = useState<Brush>({ terrain: "wood", number: 6, port: "3:1" });
  const [val, setVal] = useState<ValidateResult | null>(null);
  const [hoverErr, setHoverErr] = useState<number | null>(null);
  const [settings, setSettingsRaw] = useState<Settings>(() => { try { return { ...DEFAULT_SETTINGS, ...JSON.parse(localStorage.getItem(SETTINGS) ?? "{}") }; } catch { return DEFAULT_SETTINGS; } });
  const setSettings = (s: Settings) => { setSettingsRaw(s); localStorage.setItem(SETTINGS, JSON.stringify(s)); };
  const [results, setResults] = useState<Results>({});
  const [hoverCand, setHoverCand] = useState<number | null>(null);
  const [attrRefs, setAttrRefs] = useState<{ vertices?: number[]; edges?: number[]; hexes?: number[] } | null>(null);
  const [branch, setBranch] = useState<Branch | null>(null);
  const [exploring, setExploring] = useState(false);
  const [exploreErr, setExploreErr] = useState<string | null>(null);
  const [modal, setModal] = useState<"io" | "saved" | "shot" | null>(null);
  const [imported, setImported] = useState<Mark[] | null>(() => (shot?.low_confidence ? lowMarks(shot) : null)); // low-confidence fields of a screenshot import
  const [armed, setArmed] = useState<Armed>(null);
  const compact = useMedia("(max-width: 1024px)");
  const [tab, setTab] = useState<Tab>("live");
  const [leftTab, setLeftTab] = useState<"edit" | "live">("edit");
  // DS18-08: shrinking into the compact layout with results on screen opened Live, hiding them
  useEffect(() => { if (compact && results.a) setTab("analysis"); }, [compact]);

  // ---- source (replay / finished game)
  const [src, setSrc] = useState<Src | null>(null);
  const [actions, setActions] = useState<Action[] | null>(null);
  const [srcErr, setSrcErr] = useState<string | null>(null);
  const [reviewViewer, setReviewViewer] = useState(0);
  const [game, setGame] = useState<GameAnalysis | null>(null);
  const [numPlayers, setNumPlayers] = useState(4);

  const jump = useCallback(async (i: number) => {
    if (!srcKind || !refId) return;
    try {
      const v: View = await api.replayState(kindRaw as "game" | "file", refId, i);
      const viewer = v.actor ?? reviewViewer;
      const s = specFromView(v, viewer);
      hist.load(s);
      setSrc({ kind: srcKind, ref: refId, index: v.index ?? i, spec: s, actor: v.actor });
      setBranch(null);
      setNumPlayers(v.num_players);
      return v;
    } catch (e: any) { setSrcErr(e.message); }
  }, [srcKind, kindRaw, refId, reviewViewer, hist.load]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!srcKind || !refId) { setSrc(null); setActions(null); return; }
    (kindRaw === "game" ? api.replay(refId) : api.replayFile(refId)).then((r) => {
      setActions(r.actions ?? []);
      const agentsList: string[] = r.agents ?? [];
      const human = agentsList.indexOf("human");
      setReviewViewer(human >= 0 ? human : 0);
      // with no human seat, review the seat being analysed rather than always Ember (DS31-07)
      jump(Math.min(Number(idxRaw ?? 0) || 0, (r.actions ?? []).length)).then((v) => { if (human < 0 && v?.actor != null) setReviewViewer(v.actor); });
    }).catch((e) => setSrcErr(e.message));
  }, [kindRaw, refId]); // eslint-disable-line react-hooks/exhaustive-deps

  const clean = !!src && src.spec === hist.spec;
  const shown = branch && branch.at >= 0 ? branch.nodes[branch.at].position : hist.spec;
  const readOnly = !!branch && branch.at >= 0;
  const wire = useMemo(() => toWire(shown), [shown]);

  // ---- live validation
  useEffect(() => {
    const ctl = new AbortController();
    const t = setTimeout(() => analysis.validate(wire, ctl.signal).then(setVal).catch((e) => e?.name !== "AbortError" && setVal({ ok: false, errors: [navigator.onLine ? `Validation unavailable: ${e.message}` : "Offline: validation needs the server. Edits are saved locally."] })), 200);
    return () => { clearTimeout(t); ctl.abort(); };
  }, [wire]);
  // a finished game has no move to analyze; that is said once below, not as a red "unsupported phase" problem
  const over = shown.phase === "GAME_OVER";
  const errs = useMemo(() => (val?.errors ?? []).filter((e) => !(over && (typeof e === "string" ? e : e.message).startsWith("unsupported phase"))).map(errorTargets), [val, over]);

  // ---- editing
  const pick = (a: Action) => {
    if (readOnly || !tool) return;
    const id = a.value as number;
    if (armed) { // one-shot placement from the Live tab, paid by the player to move
      hist.set((s) => {
        const p = s.current;
        if (armed.kind === "robber") Q.moveRobber(s, id, topo);
        else if (s.phase === "SETUP_SETTLEMENT" && armed.kind === "settlement") Q.setupSettlement(s, id, topo);
        else if (s.phase === "SETUP_ROAD" && armed.kind === "road") Q.setupRoad(s, id);
        else {
          if (armed.kind === "road") s.edge_owner[id] = p; else { s.vertex_owner[id] = p; s.vertex_level[id] = armed.kind === "city" ? 2 : 1; }
          if (armed.pay) Q.pay(s, p, Q.COST[armed.kind]);
        }
        return s;
      });
      setArmed(null); setTool(null);
      return;
    }
    hist.set((s) => {
      const v = s.vertex_owner, l = s.vertex_level;
      switch (tool) {
        case "settlement": if (v[id] === seat && l[id] === 1) l[id] = 2; else { v[id] = seat; l[id] = 1; } break;
        case "city": v[id] = seat; l[id] = 2; break;
        case "road": s.edge_owner[id] = s.edge_owner[id] === seat ? -1 : seat; break;
        case "erase": if (a.type === "V") { v[id] = -1; l[id] = 0; } else s.edge_owner[id] = -1; break;
        case "robber": s.robber = id; break;
        case "terrain": s.layout.hex_terrain[id] = brush.terrain; if (brush.terrain === "desert") s.layout.hex_number[id] = null; break;
        case "number": s.layout.hex_number[id] = brush.number; break;
        case "port": s.layout.port_types[s.layout.port_edges.indexOf(id)] = brush.port; break;
      }
      return s;
    });
  };

  const arm = (a: Armed) => {
    setArmed(a);
    if (!a) { setTool(null); return; }
    setTool(a.kind); setSeat(shown.current);
  };
  // Long-press removes the piece under the finger, whatever tool is active.
  const removeAt = (p: { x: number; y: number }) => {
    if (readOnly) return;
    const dd = (q: { x: number; y: number }) => (q.x - p.x) ** 2 + (q.y - p.y) ** 2;
    const v = topo.vertices.filter((x) => shown.vertex_owner[x.id] >= 0 && dd(x) < 0.45 ** 2).sort((a, b) => dd(a) - dd(b))[0];
    const mid = (e: number) => { const [a, b] = topo.edges[e].v.map((i) => topo.vertices[i]); return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; };
    const e = topo.edges.filter((x) => shown.edge_owner[x.id] >= 0 && dd(mid(x.id)) < 0.4 ** 2).sort((a, b) => dd(mid(a.id)) - dd(mid(b.id)))[0];
    if (!v && !e) return;
    hist.set((s) => { if (v) { s.vertex_owner[v.id] = -1; s.vertex_level[v.id] = 0; } else s.edge_owner[e!.id] = -1; return s; });
  };

  const targets: Targets | undefined = useMemo(() => {
    if (readOnly || !tool) return undefined;
    const m = (n: number, type: string, f: (i: number) => boolean = () => true) =>
      new Map(Array.from({ length: n }, (_, i) => i).filter(f).map((i) => [i, { player: seat, type, value: i } as Action]));
    const s = shown, V = topo.vertices.length, E = topo.edges.length, H = topo.hexes.length;
    switch (tool) {
      case "settlement": return { vertices: m(V, "V"), ghost: "settlement", seat };
      case "city": return { vertices: m(V, "V"), ghost: "city", seat };
      case "road": return { edges: m(E, "E"), ghost: "road", seat };
      case "erase": return { vertices: m(V, "V", (i) => s.vertex_owner[i] >= 0), edges: m(E, "E", (i) => s.edge_owner[i] >= 0) };
      case "robber": return { hexes: m(H, "H", (i) => i !== s.robber), ghost: "robber" };
      case "terrain": case "number": return { hexes: m(H, "H") };
      case "port": return { edges: m(E, "E", (i) => s.layout.port_edges.includes(i)) };
    }
  }, [tool, seat, shown, readOnly, topo]);

  // ---- highlights
  const reviewCur = clean && !branch ? game?.timeline.find((t) => t.index === src!.index) : undefined;
  const marks: Mark[] = useMemo(() => {
    const out: Mark[] = [...(imported ?? [])];
    errs.forEach((e, i) => {
      const tone = hoverErr === null || hoverErr === i ? "error" : null;
      if (!tone) return;
      e.vertices.forEach((id) => out.push({ kind: "v", id, tone }));
      e.edges.forEach((id) => out.push({ kind: "e", id, tone }));
      e.hexes.forEach((id) => out.push({ kind: "h", id, tone }));
    });
    const cands = results.a?.candidates ?? [];
    if (attrRefs) {  // hovering a model attribution: show the entities it names instead of the candidates
      attrRefs.vertices?.forEach((id) => out.push({ kind: "v", id, tone: "hot" }));
      attrRefs.edges?.forEach((id) => out.push({ kind: "e", id, tone: "hot" }));
      attrRefs.hexes?.forEach((id) => out.push({ kind: "h", id, tone: "hot" }));
    } else if (cands.length) {
      cands.forEach((c) => {
        const t = actionTarget(c.action);
        if (t && (hoverCand === null || hoverCand === c.rank)) out.push({ ...t, tone: hoverCand === c.rank ? "hot" : "cand", rank: c.rank });
      });
    } else if (reviewCur) {
      const p = actionTarget(reviewCur.played), b = reviewCur.best && actionTarget(reviewCur.best);
      if (p) out.push({ ...p, tone: "played" });
      if (b) out.push({ ...b, tone: "best" });
    }
    return out;
  }, [errs, hoverErr, results, hoverCand, reviewCur, attrRefs, imported]);

  const boardView = useMemo(() => ({ ...shown, rolled: false, last_roll: null } as unknown as View), [shown]);

  // ---- explore a branch
  const explore = async (c: Candidate, policy: string) => {
    setExploring(true); setExploreErr(null);
    try {
      const r = await analysis.continue({ position: wire, action: c.action, policy, until: "turn_end" });
      const node = { position: normalize(r.position, topo), label: c.label, events: r.events ?? [], done: r.done };
      setBranch((b) => {
        const nodes = [...(b ? b.nodes.slice(0, b.at + 1) : []), node];
        return { nodes, at: nodes.length - 1 };
      });
    } catch (e: any) { setExploreErr(e.message); } finally { setExploring(false); }
  };

  // ---- keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || modal) return;
      const k = e.key.toLowerCase();
      if ((e.metaKey || e.ctrlKey) && k === "z") { e.preventDefault(); if (!readOnly) (e.shiftKey ? hist.redo : hist.undo)(); return; }
      if ((e.metaKey || e.ctrlKey) && k === "y") { e.preventDefault(); if (!readOnly) hist.redo(); return; }
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
        const d = e.key === "ArrowLeft" ? -1 : 1;
        if (branch) { e.preventDefault(); setBranch({ ...branch, at: Math.max(-1, Math.min(branch.nodes.length - 1, branch.at + d)) }); }
        else if (src && actions) { e.preventDefault(); jump(Math.max(0, Math.min(actions.length, src.index + d))); }
        return;
      }
      const t = TOOLS.find((x) => x.key === k);
      if (t) { setTool(t.id); setArmed(null); return; }
      if (k === "escape") { setTool(null); setArmed(null); return; }
      if (/^[1-4]$/.test(k) && Number(k) <= shown.num_players) { setSeat(Number(k) - 1); return; }
      if (k === "[" || k === "]") {
        const d = k === "[" ? -1 : 1, cyc = <T,>(arr: readonly T[], x: T) => arr[(arr.indexOf(x) + d + arr.length) % arr.length];
        if (tool === "terrain") setBrush({ ...brush, terrain: cyc(TERRAINS, brush.terrain as any) });
        if (tool === "number") setBrush({ ...brush, number: cyc(NUMBER_BRUSH, brush.number) });
        if (tool === "port") setBrush({ ...brush, port: cyc(PORT_KINDS, brush.port) });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  const blocked = shown.phase === "GAME_OVER" ? "The game is over: there is no move to analyze."
    : val && !val.ok && !(clean && !branch) ? `Fix ${val.errors.length} problem${val.errors.length === 1 ? "" : "s"} in the position first.`
    : actorOf(shown) !== shown.viewer && !(clean && !branch) ? `${PLAYER_NAMES[actorOf(shown)]} acts next. The engine analyzes only the acting seat's move: switch "Analyze as" to ${PLAYER_NAMES[actorOf(shown)]}.` : null;
  // Trade sub-phases have an actor that the spec cannot express; trust the game for unedited positions.
  const actor = clean && !branch && src!.actor !== null ? src!.actor : actorOf(shown);
  const pol = policySpec(settings.agent, settings.checkpoint);

  const setToolManual = (t: Tool | null) => { setTool(t); setArmed(null); };
  const actionsEl = (
    <>
      <button className="btn small" onClick={hist.redo} disabled={!hist.canRedo || readOnly} title="Redo (Ctrl+Shift+Z)">Redo</button>
      <button className="btn small" onClick={() => { setBranch(null); hist.load(randomPosition(topo)); }} data-testid="new-random">Random board</button>
      <button className="btn small" onClick={() => { setBranch(null); hist.load(emptyPosition(topo)); setToolManual("terrain"); if (compact) setTab("edit"); }} data-testid="new-empty">Empty island</button>
      <button className="btn small" onClick={() => { setBranch(null); hist.set((s) => ({ ...s, vertex_owner: s.vertex_owner.map(() => -1), vertex_level: s.vertex_level.map(() => 0), edge_owner: s.edge_owner.map(() => -1) })); }}>Clear pieces</button>
      <button className="btn small" onClick={() => setModal("saved")} data-testid="open-saved">Saved positions</button>
      <button className="btn small" onClick={() => setModal("io")} data-testid="open-io">Import / export</button>
      <button className="btn small" onClick={() => setModal("shot")} data-testid="open-shot" title="Screenshots are sent to Google Gemini when a key is configured.">Import screenshot</button>
    </>
  );
  const editorEl = <EditorPanel spec={shown} set={hist.set} tool={tool} setTool={setToolManual} seat={seat} setSeat={setSeat} brush={brush} setBrush={setBrush} readOnly={readOnly} />;
  const liveEl = readOnly
    ? <p className="an-card note small">Viewing a branch. Return to the position to record turns.</p>
    : <QuickUpdate spec={shown} set={hist.set} topo={topo} arm={arm} armed={armed} />;
  const timelineEl = src && actions && game && (
    <section className="an-timeline">
      <EvalGraph timeline={game.timeline} total={actions.length} index={src.index} onJump={jump} />
    </section>
  );
  const analysisEl = (
    <>
      <AnalysisPanel spec={wire} source={clean && !branch ? { kind: src!.kind, ref: src!.ref, index: src!.index } : null} blocked={blocked}
        settings={settings} setSettings={setSettings} agents={agents} checkpoints={checkpoints}
        actor={actor} results={results} setResults={setResults} hover={hoverCand} setHover={setHoverCand} onExplore={explore} exploring={exploring} onAttr={setAttrRefs} topo={topo} />
      {compact && timelineEl}
      {src && actions && (
        <Review kind={src.kind} refId={src.ref} actions={actions} index={src.index} onJump={jump} viewer={reviewViewer} setViewer={setReviewViewer}
          numPlayers={numPlayers} policy={pol} game={game} setGame={setGame} where={{ topo, layout: shown.layout }} />
      )}
    </>
  );
  const boardEl = (
    <main className="board-wrap an-board">
      <h2 className="sr-only">Board</h2>
      <TouchBoard topo={topo} view={boardView} targets={targets} onPick={pick} onLongPress={removeAt}>
        <BoardMarks topo={topo} marks={marks} />
      </TouchBoard>
      {armed && (
        <div className="an-armed" role="status" data-testid="armed">
          {armed.kind === "robber" ? "Tap the robber's new tile" : `Tap where ${PLAYER_NAMES[shown.current]} built the ${armed.kind}`}
          <button className="btn small" onClick={() => arm(null)}>Cancel</button>
        </div>
      )}
      {branch && (
        <div className="an-branch" data-testid="branch">
          <span className="field-label">Branch</span>
          <button className={"chip" + (branch.at === -1 ? " on" : "")} onClick={() => setBranch({ ...branch, at: -1 })}>Position</button>
          {branch.nodes.map((n, i) => <button key={i} className={"chip" + (branch.at === i ? " on" : "")} onClick={() => setBranch({ ...branch, at: i })} title={n.events.map((e) => describeEvent(e)).filter(Boolean).join("\n")}>{i + 1}. {resText(n.label)}</button>)}
          <span className="an-branch-actions">
            {readOnly && <button className="btn small" onClick={() => { hist.load(shown); setBranch(null); }}>Edit from here</button>}
            <button className="btn small ghost" onClick={() => setBranch(null)} data-testid="leave-branch">Return to game</button>
          </span>
          {readOnly && (
            <ol className="an-branch-log small">
              {branch.nodes[branch.at].events.map((e, i) => { const d = describeEvent(e); return d ? <li key={i} style={{ ["--pc" as any]: PLAYER_COLORS[e.player ?? 0] }}>{d}</li> : null; })}
              {branch.nodes[branch.at].done && <li>Game over in this line.</li>}
            </ol>
          )}
        </div>
      )}
      {(exploring || exploreErr) && <div className="an-toast" role="status">{exploring ? "Simulating the line to the end of the turn…" : `Explore failed: ${exploreErr}`}</div>}
      {errs.length > 0 && (
        <ul className="an-errors" data-testid="errors" aria-label="Problems in this position">
          {errs.slice(0, 8).map((e, i) => (
            <li key={i} onMouseEnter={() => setHoverErr(i)} onMouseLeave={() => setHoverErr(null)} onClick={() => setHoverErr(hoverErr === i ? null : i)} className={e.vertices.length + e.edges.length + e.hexes.length ? "has-target" : ""}>{e.msg}</li>
          ))}
          {errs.length > 8 && <li className="muted">…and {errs.length - 8} more</li>}
        </ul>
      )}
      {imported && (
        <div className="an-armed" role="status" data-testid="imported">
          Imported: dotted outlines are low confidence. Set hands and the turn in the editor.
          <button className="btn small" onClick={() => setImported(null)}>Dismiss</button>
        </div>
      )}
      {srcErr && <div className="an-toast error">Could not load the game: {srcErr}</div>}
    </main>
  );
  const TABS: [Tab, string][] = [["board", "Board"], ["live", "Live"], ["edit", "Edit"], ["analysis", "Analysis"]];

  return (
    <div className={"an" + (src ? " with-review" : "") + (tool ? ` tool-${tool}` : "") + (compact ? ` compact tab-${tab}` : "")}>
      <h1 className="sr-only">Analysis board</h1>
      <header className="topbar an-top" style={{ ["--actor" as any]: PLAYER_COLORS[actor] }}>
        <button className="btn ghost small" onClick={() => go("/")}>Menu</button>
        <div className="banner-text">
          <span className="banner-phase">Analysis board · {PHASE_LABEL[shown.phase] ?? shown.phase} · Turn {shown.turn + 1}
            {src && <> · {src.kind === "game" ? "game" : "replay"} {src.ref}, before move {src.index + 1}{clean ? "" : " (edited)"}</>}
          </span>
          <strong className="banner-instr">{over ? "Game over" : `${PLAYER_NAMES[actor]} to act`} · analyzing as {PLAYER_NAMES[shown.viewer]}</strong>
        </div>
        {mode === "mock" && <span className="an-mock" data-testid="mock-badge" title="/api/analysis is not available on this server. Validation is a partial local check; every number is fake.">MOCK backend</span>}
        <span className={"an-valid" + (val ? (val.ok || !errs.length ? (over ? "" : " ok") : " bad") : "")} data-testid="validity">{!val ? "Checking…" : val.ok || !errs.length ? (over ? "Game over" : "Position valid") : `${errs.length} problem${errs.length === 1 ? "" : "s"}`}</span>
        <div className="an-actions">
          <button className="btn small" onClick={hist.undo} disabled={!hist.canUndo || readOnly} title="Undo (Ctrl+Z)" data-testid="undo">Undo</button>
          {compact ? (
            <details className="an-more">
              <summary className="btn small" data-testid="more">More</summary>
              <div className="an-more-menu" onClick={(e) => (e.currentTarget.parentElement as HTMLDetailsElement).removeAttribute("open")}>{actionsEl}</div>
            </details>
          ) : actionsEl}
        </div>
      </header>

      {compact ? (
        <>
          {boardEl}
          <nav className="an-tabs" role="tablist" aria-label="Panels">
            {TABS.map(([t, l]) => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? "on" : ""} onClick={() => setTab(t)} data-tab={t}>{l}</button>)}
          </nav>
          {tab !== "board" && <div className="an-pane" role="tabpanel"><h2 className="sr-only">{TABS.find(([t]) => t === tab)![1]}</h2>{tab === "live" ? liveEl : tab === "edit" ? editorEl : analysisEl}</div>}
        </>
      ) : (
        <>
          <div className="an-left">
            <h2 className="sr-only">{leftTab === "edit" ? "Position editor" : "Live turn"}</h2>
            <span className="segmented small an-lefttabs" role="tablist">
              <button role="tab" aria-selected={leftTab === "edit"} className={leftTab === "edit" ? "on" : ""} onClick={() => setLeftTab("edit")} data-tab="edit">Edit</button>
              <button role="tab" aria-selected={leftTab === "live"} className={leftTab === "live" ? "on" : ""} onClick={() => setLeftTab("live")} data-tab="live">Live turn</button>
            </span>
            {leftTab === "edit" ? editorEl : liveEl}
          </div>
          {boardEl}
          {timelineEl}
          <aside className="an-right"><h2 className="sr-only">Analysis{src ? " and game review" : ""}</h2>{analysisEl}</aside>
        </>
      )}

      {modal === "io" && <ImportExport spec={hist.spec} topo={topo} onLoad={(s) => { setBranch(null); hist.load(s); setModal(null); }} onClose={() => setModal(null)} />}
      {modal === "shot" && <ImportScreenshot topo={topo} onClose={() => setModal(null)}
        onLoad={(r) => { setBranch(null); hist.load(normalize(r.position, topo)); setImported(lowMarks(r)); setModal(null); setResults({}); }} />}
      {modal === "saved" && <SavedPositions spec={hist.spec} topo={topo} onLoad={(s) => { setBranch(null); hist.load(s); setModal(null); }} onClose={() => setModal(null)} />}
    </div>
  );
}

function ImportExport({ spec, topo, onLoad, onClose }: { spec: PositionSpec; topo: Topology; onLoad: (s: PositionSpec) => void; onClose: () => void }) {
  const out = JSON.stringify(toWire(spec), null, 1);
  const [text, setText] = useState("");
  const [err, setErr] = useState<string | null>(null);
  // one blob URL per export text, released when it changes (was a new leaked URL every render)
  const href = useMemo(() => URL.createObjectURL(new Blob([out], { type: "application/json" })), [out]);
  useEffect(() => () => URL.revokeObjectURL(href), [href]);
  const load = (t: string) => { try { onLoad(normalize(JSON.parse(t), topo)); } catch (e: any) { setErr(e.message); } };
  return (
    <Modal title="Import / export position" wide onClose={onClose}>
      <div className="an-io">
        <label className="field"><span className="field-label">This position (PositionSpec v1)</span>
          <textarea readOnly value={out} rows={10} data-testid="export-json" onFocus={(e) => e.target.select()} /></label>
        <div className="modal-actions">
          <button className="btn" onClick={() => navigator.clipboard?.writeText(out)}>Copy</button>
          <a className="btn" href={href} download="position.json">Download</a>
        </div>
        <label className="field"><span className="field-label">Paste a PositionSpec (or an evaluate response)</span>
          <textarea value={text} onChange={(e) => { setText(e.target.value); setErr(null); }} rows={6} data-testid="import-json" placeholder='{"version": 1, ...}' /></label>
        {err && <p className="error small" role="alert">Can't import: {err}</p>}
        <div className="modal-actions">
          <input type="file" accept="application/json,.json" aria-label="Import file" onChange={(e) => e.target.files?.[0]?.text().then(load)} />
          <button className="btn primary" onClick={() => load(text)} disabled={!text.trim()} data-testid="do-import">Load position</button>
        </div>
      </div>
    </Modal>
  );
}

function SavedPositions({ spec, topo, onLoad, onClose }: { spec: PositionSpec; topo: Topology; onLoad: (s: PositionSpec) => void; onClose: () => void }) {
  const [saved, setSaved] = useState<Saved>(readSaved);
  const [name, setName] = useState(`Turn ${spec.turn + 1}, ${new Date().toLocaleString()}`);
  const put = (s: Saved) => { setSaved(s); writeSaved(s); };
  const list = Object.entries(saved).sort((a, b) => b[1].at - a[1].at);
  return (
    <Modal title="Saved positions" onClose={onClose}>
      <form className="an-row" onSubmit={(e) => { e.preventDefault(); if (name.trim()) put({ ...saved, [name.trim()]: { spec, at: Date.now() } }); }}>
        <input value={name} onChange={(e) => setName(e.target.value)} aria-label="Position name" style={{ flex: 1 }} data-testid="save-name" />
        <button className="btn primary" type="submit" data-testid="save-position">Save current</button>
      </form>
      {list.length === 0 ? <p className="muted">Nothing saved yet. Positions are stored in this browser.</p> : (
        <ul className="an-saved">
          {list.map(([k, v]) => (
            <li key={k}>
              <span><strong>{k}</strong><br /><span className="muted small">{v.spec.num_players} players · {PHASE_NAME[v.spec.phase] ?? v.spec.phase} · turn {v.spec.turn + 1}</span></span>
              <button className="btn small" onClick={() => { try { onLoad(normalize(v.spec, topo)); } catch { /* skip */ } }} aria-label={`Load ${k}`}>Load</button>
              <button className="btn small ghost" onClick={() => { const s = { ...saved }; delete s[k]; put(s); }} aria-label={`Delete ${k}`}>Delete</button>
            </li>
          ))}
        </ul>
      )}
      <p className="muted small">The current position also autosaves on every change and is restored when you come back.</p>
    </Modal>
  );
}
