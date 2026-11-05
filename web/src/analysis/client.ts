// Analysis calls with a dev-only fallback to ./mock.ts when the backend route
// does not exist yet. "Route missing" = 404 "Not Found" or 405 (static mount).
import { useSyncExternalStore } from "react";
import { api, ApiError, Action, AnalysisSource, Depth, EvaluateRequest, PositionSpec } from "../api";
import { mockContinue, mockEvaluate, mockGame, mockLegal, mockValidate } from "./mock";

type Mode = "unknown" | "real" | "mock";
let mode: Mode = "unknown";
const subs = new Set<() => void>();
const setMode = (m: Mode) => { if (mode !== m) { mode = m; subs.forEach((f) => f()); } };
const subscribe = (f: () => void) => { subs.add(f); return () => { subs.delete(f); }; };
export const useBackendMode = () => useSyncExternalStore(subscribe, () => mode);

const missing = (e: unknown) => e instanceof ApiError && ((e.status === 404 && e.message === "Not Found") || e.status === 405);

async function call<T>(real: () => Promise<T>, fake: () => Promise<T>): Promise<T> {
  if (mode === "mock") return fake();
  try {
    const r = await real();
    setMode("real");
    return r;
  } catch (e) {
    if (!missing(e)) throw e;
    setMode("mock");
    return fake();
  }
}

export const analysis = {
  validate: (p: PositionSpec, signal?: AbortSignal) => call(() => api.analysis.validate(p, signal), async () => mockValidate(p)),
  legal: (p: PositionSpec, signal?: AbortSignal) => call(() => api.analysis.legal(p, signal), async () => ({ legal: mockLegal(p) })),
  // With a source the server builds the spec itself; `position` is only for the mock.
  evaluate: (b: EvaluateRequest & { position: PositionSpec }, signal?: AbortSignal) =>
    call(() => api.analysis.evaluate(b.source ? { ...b, position: undefined } : b, signal),
      () => mockEvaluate(b.position, b.policy, b.depth, b.top_k, signal, b.rollout_policy)),
  game: (b: { source: AnalysisSource; viewer: number; policy: string; depth: Depth }, signal?: AbortSignal) =>
    call(() => api.analysis.game(b, signal), () => mockGame(b.source.kind, b.source.ref, b.viewer, signal)),
  continue: (b: { position: PositionSpec; action: Action; policy: string; until?: "turn_end" | "game_end" }, signal?: AbortSignal) =>
    call(() => api.analysis.continue(b, signal), () => mockContinue(b.position, b.action, signal)),
};
