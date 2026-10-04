import { API_BASE } from "@/lib/api";
import type { Step } from "@/lib/types";

const tierStyles: Record<Step["safetyTier"], string> = {
  routine: "bg-subtle text-secondary",
  caution: "bg-warning-bg text-warning-text",
  critical: "bg-danger-bg text-danger-text",
};

export default function StepOverlay({
  step,
  stepNumber,
  totalSteps,
}: {
  step: Step;
  stepNumber: number;
  totalSteps: number;
}) {
  return (
    <div className="w-full max-w-xl rounded-2xl border border-border bg-card p-6 shadow-sm">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wide text-muted">
          Step {stepNumber} of {totalSteps}
        </span>
        <span
          className={`rounded-full px-2.5 py-1 text-xs font-medium capitalize ${tierStyles[step.safetyTier]}`}
        >
          {step.safetyTier}
        </span>
      </div>

      <h2 className="mt-2 text-lg font-semibold tracking-tight">{step.title}</h2>
      <p className="mt-2 text-sm leading-relaxed text-secondary">{step.instruction}</p>

      {step.warnings && step.warnings.length > 0 && (
        <div className="mt-4 space-y-2">
          {step.warnings.map((w, i) => (
            <div
              key={i}
              className="rounded-lg border border-warning/30 bg-warning-bg px-3 py-2 text-sm text-warning-text"
            >
              {w}
            </div>
          ))}
        </div>
      )}

      {step.diagram && (
        <div className="mt-4 overflow-hidden rounded-xl border border-border">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={`${API_BASE}${step.diagram}`}
            alt={`${step.title} diagram`}
            className="h-auto w-full object-cover"
          />
        </div>
      )}
    </div>
  );
}
