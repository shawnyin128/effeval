import type { ReactNode } from "react";

export interface StepItem {
  id: string;
  label: ReactNode;
  state: "done" | "current" | "todo";
  reachable: boolean;
  /** Severity of the worst validation issue pointing at this step, if any. */
  issue?: "error" | "warning";
}

/** Horizontal numbered step bar; reachable steps jump on click, a dot marks steps with validation issues. */
export function StepBar({ steps, onSelect, ariaLabel }: { steps: StepItem[]; onSelect: (id: string) => void; ariaLabel: string }) {
  return (
    <nav className="stepper" aria-label={ariaLabel}>
      {steps.map((step, index) => (
        <button key={step.id} type="button" className={`stepper-item ${step.state}`} disabled={!step.reachable}
          aria-current={step.state === "current" ? "step" : undefined} onClick={() => onSelect(step.id)}>
          <span className="stepper-num">{index + 1}</span>
          <span>{step.label}</span>
          {step.issue && <span className={`dot ${step.issue === "error" ? "red" : "orange"}`} />}
        </button>
      ))}
    </nav>
  );
}
