"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Sidebar from "@/components/app-shell/Sidebar";
import VoiceVisual from "@/components/app-shell/VoiceVisual";
import StepOverlay from "@/components/app-shell/StepOverlay";
import SensorsPanel from "@/components/app-shell/SensorsPanel";
import { useVoiceLoop } from "@/lib/useVoiceLoop";
import { loadSessions, saveSessions, type StoredSession } from "@/lib/sessions";
import { createSession, diagramUrl, getSession, type RetrievedDiagram } from "@/lib/api";
import type { Decision, DecisionAction, SessionState } from "@/lib/types";

const actionMeta: Record<DecisionAction, { label: string; className: string }> = {
  advance: { label: "Advance", className: "bg-accent-100 text-accent-700" },
  answer: { label: "Answer", className: "bg-subtle text-secondary" },
  repeat: { label: "Repeat", className: "bg-subtle text-secondary" },
  clarify: { label: "Clarify", className: "bg-info-bg text-info-text" },
  branch: { label: "Re-route", className: "bg-info-bg text-info-text" },
  replan: { label: "Re-plan", className: "bg-info-bg text-info-text" },
  block: { label: "Blocked", className: "bg-warning-bg text-warning-text" },
  escalate: { label: "Escalate", className: "bg-warning-bg text-warning-text" },
  emergency: { label: "Emergency", className: "bg-danger-bg text-danger-text" },
  tool_request: { label: "Checking", className: "bg-subtle text-secondary" },
};

function makeSession(state: SessionState): StoredSession {
  return {
    id: state.id,
    title: "New discussion",
    procedureId: state.procedure_id || "",
    backendSessionId: state.id,
    backendState: state,
    updatedAt: "Just now",
    stepIndex: 0,
    completed: false,
    messages: [],
  };
}

export default function AppPage() {
  const [sessions, setSessions] = useState<StoredSession[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [sessionError, setSessionError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  // Hydrate saved sessions once on the client, then persist on every change.
  useEffect(() => {
    setSessions(loadSessions());
    setHydrated(true);
  }, []);
  useEffect(() => {
    if (hydrated) saveSessions(sessions);
  }, [sessions, hydrated]);

  const current = useMemo(
    () => sessions.find((s) => s.id === selectedId) ?? null,
    [sessions, selectedId]
  );

  // Render the backend's authoritative state alongside the local transcript.
  const handleDecision = useCallback(
    (decision: Decision, query: string, diagram?: RetrievedDiagram | null, state?: SessionState) => {
      if (!state) return;
      setSessions((prev) =>
        prev.map((s) => {
          if (s.backendSessionId !== state.id) return s;
          const stepIndex = state.procedure?.steps.findIndex((step) => step.id === state.step_id) ?? 0;
          const hasUserTurn = s.messages.some((m) => m.role === "user");
          const title = hasUserTurn
            ? s.title
            : query.trim().slice(0, 48) || s.title;
          return {
            ...s,
            title,
            stepIndex,
            completed: state.completed,
            procedureId: state.procedure_id || "",
            backendState: state,
            updatedAt: "Just now",
            messages: [
              ...s.messages,
              { role: "user", text: query },
              {
                role: "ai",
                text: decision.spoken_text,
                action: decision.action,
                risk: decision.risk,
                citations: decision.citations,
                diagramId: diagram?.diagram_id,
                diagramTitle: diagram?.title,
              },
            ],
          };
        })
      );
    },
    []
  );

  const sessionActive = current !== null && !!current.backendSessionId &&
    !current.completed && !current.backendState?.halted;
  const voice = useVoiceLoop({ active: sessionActive, sessionId: current?.backendSessionId ?? null, onDecision: handleDecision });

  // Keep the thread scrolled to the latest turn / thinking indicator (including as
  // the streamed answer grows).
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [current?.messages.length, voice.phase, voice.partial]);

  async function startNewSession() {
    try {
      const state = await createSession();
      setSessions((prev) => [makeSession(state), ...prev]);
      setSelectedId(state.id);
      setSessionError(null);
    } catch {
      setSessionError("Backend unavailable. Start the backend before creating a session.");
    }
  }

  async function selectSession(id: string) {
    const saved = sessions.find((s) => s.id === id);
    if (!saved) return;
    try {
      const state = saved.backendSessionId
        ? await getSession(saved.backendSessionId)
        : await createSession(); // Legacy transcript stays; its step pointer is untrusted.
      setSessions((prev) => prev.map((s) => s.id === id ? {
        ...s, backendSessionId: state.id, backendState: state,
        procedureId: state.procedure_id || "", stepIndex: 0, completed: state.completed,
      } : s));
      setSelectedId(id);
      setSessionError(null);
    } catch {
      setSessionError("Could not resume this session from the backend.");
    }
  }

  const lastAi = current?.messages.filter((m) => m.role === "ai").at(-1);
  const emergency = lastAi?.action === "emergency";
  const procedure = current?.backendState?.procedure;
  const stepIndex = procedure?.steps.findIndex((s) => s.id === current?.backendState?.step_id) ?? -1;
  const step = stepIndex >= 0 ? procedure?.steps[stepIndex] : undefined;

  const statusText =
    voice.permission === "pending"
      ? "Requesting microphone access…"
      : voice.phase === "thinking"
        ? "Thinking"
        : voice.phase === "speaking"
          ? "Speaking"
          : "Listening";

  return (
    <div className="flex h-screen bg-page">
      <Sidebar
        conversations={sessions.map((s) => ({
          id: s.id,
          title: s.title,
          procedureId: s.procedureId,
          updatedAt: s.updatedAt,
        }))}
        activeId={selectedId}
        onSelect={selectSession}
        onNewSession={startNewSession}
      />

      <main className="relative flex-1 overflow-hidden bg-page">
        {current === null ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <p className="max-w-xs text-sm text-secondary">
              Click &quot;+ New&quot; to start a hands-free walkthrough, then describe
              the fault out loud. Say &quot;next step&quot; when a step is done.
            </p>
            {sessionError && <p className="text-sm text-danger-text">{sessionError}</p>}
          </div>
        ) : voice.permission === "denied" ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
            <p className="max-w-xs text-sm text-secondary">
              Microphone access is required for hands-free voice control. Please
              allow microphone access and reopen this session.
            </p>
          </div>
        ) : (
          <div className="h-full overflow-y-auto px-6 pb-56 pt-10">
            <div className="mx-auto flex max-w-xl flex-col gap-4">
              {emergency && (
                <div className="rounded-2xl border border-danger/40 bg-danger-bg px-5 py-4">
                  <p className="text-sm font-semibold text-danger-text">Emergency mode</p>
                  <p className="mt-1 text-sm text-danger-text">
                    {lastAi?.risk || lastAi?.text || "Unsafe condition — stop and stabilize."}
                  </p>
                </div>
              )}

              {!current.completed && step && (
                <StepOverlay
                  step={step}
                  stepNumber={stepIndex + 1}
                  totalSteps={procedure?.steps.length ?? 0}
                />
              )}

              {!step && !current.completed && (
                <p className="rounded-2xl border border-dashed border-border px-4 py-4 text-center text-sm text-muted">
                  Name the airlock, coolant, battery, or CDRA procedure. Confirm the suggested procedure before starting.
                </p>
              )}

              {current.messages.length === 0 && (
                <p className="rounded-2xl border border-dashed border-border px-4 py-6 text-center text-sm text-muted">
                  Describe the fault out loud to begin.
                </p>
              )}

              {current.messages.map((m, i) =>
                m.role === "user" ? (
                  <div key={i} className="flex justify-end">
                    <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-accent-100 px-4 py-2 text-sm text-accent-900">
                      {m.text}
                    </div>
                  </div>
                ) : (
                  <div key={i} className="flex justify-start">
                    <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3 text-sm shadow-sm">
                      <div className="mb-1 flex items-center gap-2">
                        <span className="text-xs font-medium text-primary">ZeroDelay</span>
                        {m.action && (
                          <span
                            className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${actionMeta[m.action].className}`}
                          >
                            {actionMeta[m.action].label}
                          </span>
                        )}
                      </div>
                      <p className="leading-relaxed text-secondary">{m.text}</p>
                      {m.risk && (
                        <p className="mt-2 rounded-lg border border-warning/30 bg-warning-bg px-3 py-2 text-warning-text">
                          {m.risk}
                        </p>
                      )}
                      {m.diagramId && (
                        <figure className="mt-3 overflow-hidden rounded-xl border border-border bg-sunken">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img
                            src={diagramUrl(m.diagramId)}
                            alt={m.diagramTitle || "Retrieved diagram"}
                            className="h-auto w-full object-contain"
                            loading="lazy"
                          />
                          {m.diagramTitle && (
                            <figcaption className="px-3 py-1.5 text-xs text-muted">
                              {m.diagramTitle}
                            </figcaption>
                          )}
                        </figure>
                      )}
                      {m.citations && m.citations.length > 0 && (
                        <p className="mt-2 text-xs text-muted">
                          Sources: {m.citations.join(", ")}
                        </p>
                      )}
                    </div>
                  </div>
                )
              )}

              {voice.phase === "thinking" && (
                <div className="flex justify-start">
                  {voice.partial ? (
                    // The answer streaming in live, token by token.
                    <div className="max-w-[92%] rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3 text-sm shadow-sm">
                      <div className="mb-1 flex items-center gap-2">
                        <span className="text-xs font-medium text-primary">ZeroDelay</span>
                      </div>
                      <p className="leading-relaxed text-secondary">
                        {voice.partial}
                        <span className="ml-0.5 inline-block h-3.5 w-[2px] translate-y-0.5 animate-pulse rounded-sm bg-muted align-middle" />
                      </p>
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 rounded-2xl rounded-bl-sm border border-border bg-card px-4 py-3 shadow-sm">
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
                      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted" />
                    </div>
                  )}
                </div>
              )}

              {current.completed && (
                <p className="rounded-2xl border border-accent-300/50 bg-accent-50 px-4 py-3 text-center text-sm text-accent-700">
                  Session complete — all steps confirmed.
                </p>
              )}

              {voice.error && (
                <p className="rounded-lg border border-danger/30 bg-danger-bg px-3 py-2 text-sm text-danger-text">
                  {voice.error}
                </p>
              )}
              {sessionError && <p className="text-sm text-danger-text">{sessionError}</p>}

              <div ref={endRef} />
            </div>
          </div>
        )}

        {sessionActive && voice.permission !== "denied" && (
          <div className="pointer-events-none absolute inset-x-0 bottom-0 flex flex-col items-center gap-1 bg-gradient-to-t from-page via-page/90 to-transparent pb-8 pt-10">
            <VoiceVisual amplitude={voice.amplitude} phase={voice.phase} size="compact" />
            <p className="text-xs font-medium text-secondary">
              {statusText}
              {voice.phase !== "listening" && voice.permission === "granted" && (
                <span className="animate-pulse">…</span>
              )}
            </p>
          </div>
        )}
      </main>

      <SensorsPanel />
    </div>
  );
}
