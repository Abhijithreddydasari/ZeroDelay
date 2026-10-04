// Thin client for the ZeroDelay FastAPI backend (backend/api/server.py).
// The backend has open CORS, so a browser on :3000 can call it on :8000 directly.

import type { Decision, Sensor, SessionState } from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_ZD_API || "http://127.0.0.1:8000"
).replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  started: boolean;
  constructor(status: number, message: string, started = false) {
    super(message);
    this.status = status;
    this.started = started;
    this.name = "ApiError";
  }
}

export type RetrievedDiagram = {
  diagram_id: string;
  title: string;
  image_path: string;
  image_exists: boolean;
  score: number;
};

export type ConverseResponse = {
  query: string;
  decision: Decision;
  retrieval?: { procedures: unknown[]; diagrams: RetrievedDiagram[] };
  sensor_snapshot: Sensor[];
  tool_calls: unknown[];
  tts_wav_base64: string | null;
  transcribed?: boolean;
  session?: SessionState;
  audio_chunks_base64?: string[];
  timing_ms?: Record<string, number>;
};

export async function createSession(): Promise<SessionState> {
  return asJson(await fetch(`${API_BASE}/sessions`, { method: "POST", signal: AbortSignal.timeout(15000) }));
}

export async function getSession(id: string): Promise<SessionState> {
  return asJson(await fetch(`${API_BASE}/sessions/${encodeURIComponent(id)}`, { signal: AbortSignal.timeout(15000) }));
}

/** URL for a diagram PNG served by the backend's /diagrams static mount. */
export function diagramUrl(diagramId: string): string {
  return `${API_BASE}/diagrams/${encodeURIComponent(diagramId)}.png`;
}

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail: string = res.statusText;
    try {
      const body = await res.json();
      if (body?.detail) detail = String(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

/** POST recorded WAV -> ASR + session decision + TTS. */
export async function converse(wav: Blob, sessionId: string, turnId: string): Promise<ConverseResponse> {
  const form = new FormData();
  form.append("audio", wav, "input.wav");
  form.append("session_id", sessionId);
  form.append("turn_id", turnId);
  const res = await fetch(`${API_BASE}/converse`, { method: "POST", body: form, signal: AbortSignal.timeout(180000) });
  return asJson<ConverseResponse>(res);
}

export type ConverseStreamHandlers = {
  /** Fired once with the transcribed technician utterance. */
  onQuery?: (query: string) => void;
  /** Legacy model-only text event; session turns do not currently emit deltas. */
  onDelta?: (text: string) => void;
  onAudioChunk?: (wavBase64: string) => void;
};

/**
 * Session stream: query, validated audio chunks, then authoritative final state.
 * The returned result omits audio already delivered through onAudioChunk.
 */
export async function converseStream(
  wav: Blob,
  sessionId: string,
  turnId: string,
  handlers: ConverseStreamHandlers = {}
): Promise<ConverseResponse> {
  const form = new FormData();
  form.append("audio", wav, "input.wav");
  form.append("session_id", sessionId);
  form.append("turn_id", turnId);
  const res = await fetch(`${API_BASE}/converse/stream`, {
    method: "POST",
    body: form,
    signal: AbortSignal.timeout(180000),
  });
  // Non-2xx (incl. 422 "no speech") has a JSON error body — reuse the error path.
  if (!res.ok || !res.body) {
    return asJson<ConverseResponse>(res);
  }

  type StreamEvent =
    | { type: "query"; text: string }
    | { type: "delta"; text: string }
    | { type: "final"; result: ConverseResponse }
    | { type: "audio_chunk"; wav_base64: string }
    | { type: "error"; detail: string };

  let final: ConverseResponse | null = null;
  let started = false;
  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    const msg = JSON.parse(trimmed) as StreamEvent;
    if (msg.type !== "error") started = true;
    if (msg.type === "query") handlers.onQuery?.(msg.text);
    else if (msg.type === "delta") handlers.onDelta?.(msg.text);
    else if (msg.type === "audio_chunk") handlers.onAudioChunk?.(msg.wav_base64);
    else if (msg.type === "final") final = msg.result;
    else if (msg.type === "error") throw new ApiError(500, msg.detail, started);
  };

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let nl: number;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl);
        buffer = buffer.slice(nl + 1);
        handleLine(line);
      }
    }
  } catch (e) {
    if (e instanceof ApiError) throw e;
    throw new ApiError(0, "Voice stream interrupted.", started);
  }
  if (buffer.trim()) handleLine(buffer); // trailing line without newline

  if (!final) throw new ApiError(500, "Stream ended without a final result.", started);
  return final;
}

export async function getSensors(): Promise<{ sensors: Sensor[] }> {
  return asJson(await fetch(`${API_BASE}/sensors`));
}

export async function injectSensor(
  name: string,
  value: number | string
): Promise<{ reading: Sensor }> {
  const res = await fetch(`${API_BASE}/sensors/inject`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, value }),
  });
  return asJson(res);
}

export async function resetSensors(): Promise<{ status: string }> {
  return asJson(await fetch(`${API_BASE}/sensors/reset`, { method: "POST" }));
}

export async function health(): Promise<{ status: string; gemma_model: string }> {
  return asJson(await fetch(`${API_BASE}/health`));
}
