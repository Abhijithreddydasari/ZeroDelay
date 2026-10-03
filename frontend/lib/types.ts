export type SafetyTier = "routine" | "caution" | "critical";

export type Step = {
  id: number;
  title: string;
  safetyTier: SafetyTier;
  instruction: string;
  warnings?: string[];
  diagram?: string;
};

export type Procedure = {
  id: string;
  title: string;
  system: string;
  summary: string;
  steps: Step[];
};

export type SessionState = {
  id: string;
  procedure_id: string | null;
  step_id: number | null;
  completed_step_ids: number[];
  pending_procedure_id: string | null;
  pending_confirmation: boolean;
  completed: boolean;
  halted: boolean;
  procedure: Procedure | null;
};

export type Conversation = {
  id: string;
  title: string;
  procedureId: string;
  updatedAt: string;
};

export type MicPermission = "idle" | "pending" | "granted" | "denied";

export type VoicePhase = "idle" | "listening" | "thinking" | "speaking";

// ---------------------------------------------------------------------------
// Backend contract (see backend/agent/schema.py and backend/tools/sensor_sim.py).
// ---------------------------------------------------------------------------
export type DecisionAction =
  | "answer"
  | "advance"
  | "block"
  | "branch"
  | "replan"
  | "repeat"
  | "escalate"
  | "emergency"
  | "clarify"
  | "tool_request";

export type Decision = {
  action: DecisionAction;
  spoken_text: string;
  procedure_id: string | null;
  step_id: number | null;
  citations: string[];
  tool_request: { tool: string; args: Record<string, unknown> } | null;
  risk: string | null;
  needs_clarification: string | null;
  confidence: number;
};

export type SensorStatus = "nominal" | "caution" | "critical" | "unknown";

export type Sensor = {
  sensor: string;
  known: boolean;
  value: number | string | null;
  unit?: string | null;
  status?: SensorStatus;
  label?: string | null;
};

// One turn in a session's conversation thread.
export type Message = {
  role: "user" | "ai";
  text: string;
  action?: DecisionAction;
  risk?: string | null;
  citations?: string[];
  // A retrieved diagram to display with an AI turn (id maps to a PNG served by
  // the backend at /diagrams/<id>.png). Stored by id so the URL isn't baked in.
  diagramId?: string;
  diagramTitle?: string;
};
