// Local persistence for voice sessions so past discussions (and their full
// conversation threads) survive reloads and can be reopened from the sidebar.
// Transcripts stay in localStorage; procedure state is authoritative in backend SQLite.

import type { Message, SessionState } from "./types";

export type StoredSession = {
  id: string;
  title: string;
  procedureId: string;
  backendSessionId?: string;
  backendState?: SessionState;
  updatedAt: string;
  stepIndex: number;
  completed: boolean;
  messages: Message[];
};

const KEY = "zerodelay:sessions";

export function loadSessions(): StoredSession[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as StoredSession[]) : [];
  } catch {
    return [];
  }
}

export function saveSessions(sessions: StoredSession[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(KEY, JSON.stringify(sessions));
  } catch {
    /* quota / privacy mode — non-fatal */
  }
}
