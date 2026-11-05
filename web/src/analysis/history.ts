// Undo/redo for the editor, plus localStorage autosave and named positions.
import { useCallback, useEffect, useRef, useState } from "react";
import type { PositionSpec } from "../api";

const AUTOSAVE = "catan.analysis.autosave.v1";
const SAVED = "catan.analysis.saved.v1";
const LIMIT = 200;

export function useHistory(init: () => PositionSpec) {
  const [h, setH] = useState(() => ({ past: [] as PositionSpec[], now: init(), future: [] as PositionSpec[], rev: 0 }));
  const lastKey = useRef<string | null>(null);
  /** `key` coalesces repeated edits (e.g. holding + on a stepper) into one undo step. */
  const set = useCallback((f: (s: PositionSpec) => PositionSpec, key?: string) => {
    setH((x) => {
      const next = f(structuredClone(x.now));
      delete (next as any).actor; // an edit invalidates a server-supplied actor; the phase rules derive it
      const merge = key !== undefined && key === lastKey.current;
      lastKey.current = key ?? null;
      return { past: merge ? x.past : [...x.past.slice(-LIMIT), x.now], now: next, future: [], rev: x.rev + 1 };
    });
  }, []);
  const undo = useCallback(() => { lastKey.current = null; setH((x) => x.past.length ? { past: x.past.slice(0, -1), now: x.past[x.past.length - 1], future: [x.now, ...x.future], rev: x.rev + 1 } : x); }, []);
  const redo = useCallback(() => { lastKey.current = null; setH((x) => x.future.length ? { past: [...x.past, x.now], now: x.future[0], future: x.future.slice(1), rev: x.rev + 1 } : x); }, []);
  /** Replace the position (load/new); still undoable. */
  const load = useCallback((s: PositionSpec) => { lastKey.current = null; setH((x) => ({ past: [...x.past.slice(-LIMIT), x.now], now: s, future: [], rev: x.rev + 1 })); }, []);
  return { spec: h.now, rev: h.rev, set, undo, redo, load, canUndo: h.past.length > 0, canRedo: h.future.length > 0 };
}

export function readAutosave(): PositionSpec | null {
  try { return JSON.parse(localStorage.getItem(AUTOSAVE) ?? "null"); } catch { return null; }
}

export function useAutosave(spec: PositionSpec) {
  useEffect(() => {
    const save = () => { try { localStorage.setItem(AUTOSAVE, JSON.stringify(spec)); } catch { /* quota: skip */ } };
    const t = setTimeout(save, 300);
    // Tablets kill background tabs: flush when the page is hidden or unloaded.
    const onHide = () => { if (document.visibilityState === "hidden") save(); };
    addEventListener("pagehide", save);
    document.addEventListener("visibilitychange", onHide);
    return () => { clearTimeout(t); removeEventListener("pagehide", save); document.removeEventListener("visibilitychange", onHide); };
  }, [spec]);
}

export type Saved = Record<string, { spec: PositionSpec; at: number }>;
export function readSaved(): Saved {
  try { return JSON.parse(localStorage.getItem(SAVED) ?? "{}") ?? {}; } catch { return {}; }
}
export function writeSaved(s: Saved) { localStorage.setItem(SAVED, JSON.stringify(s)); }
