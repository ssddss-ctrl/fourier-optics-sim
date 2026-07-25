/**
 * frontend/src/lib/hooks.ts
 * ----------------------------
 * Shared hooks used by every Simulator section component (extracted from
 * the original single-page Simulator.tsx, where they were defined inline
 * at module scope -- now needed independently by multiple section files).
 */

import { useEffect, useRef, useState } from "react";

export const SLIDER_DEBOUNCE_MS = 250;

export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(id);
  }, [value, delayMs]);
  return debounced;
}

export interface AsyncPanel<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

/**
 * Fires `fetcher(req)` whenever `req`'s identity changes (callers memoize
 * req so that only happens once controls settle, not on every keystroke/
 * drag tick) and keeps only the latest in-flight response, in case a slow
 * request resolves after a newer one already landed.
 */
export function useApiPanel<TReq, TRes>(req: TReq, fetcher: (req: TReq) => Promise<TRes>): AsyncPanel<TRes> {
  const [state, setState] = useState<AsyncPanel<TRes>>({ data: null, loading: true, error: null });
  const requestIdRef = useRef(0);

  useEffect(() => {
    const requestId = ++requestIdRef.current;
    setState((prev) => ({ ...prev, loading: true, error: null }));
    fetcher(req)
      .then((data) => {
        if (requestIdRef.current === requestId) {
          setState({ data, loading: false, error: null });
        }
      })
      .catch((err: unknown) => {
        if (requestIdRef.current === requestId) {
          setState((prev) => ({
            ...prev,
            loading: false,
            error: err instanceof Error ? err.message : String(err),
          }));
        }
      });
  }, [req, fetcher]);

  return state;
}
