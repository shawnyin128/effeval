import { Check, ChevronDown, Minus } from "lucide-react";
import { Checkbox as RCheckbox, Select as RSelect, Switch as RSwitch, ToggleGroup } from "radix-ui";
import type { ButtonHTMLAttributes, InputHTMLAttributes, ReactNode, TextareaHTMLAttributes } from "react";

type Variant = "primary" | "secondary" | "ghost" | "link" | "danger";

export function Button({ variant = "secondary", size, icon, block, className = "", children, ...props }: {
  variant?: Variant;
  size?: "sm" | "lg";
  icon?: ReactNode;
  block?: boolean;
} & ButtonHTMLAttributes<HTMLButtonElement>) {
  const classes = ["btn", `btn-${variant}`, size ? `btn-${size}` : "", !children && icon ? "btn-icon" : "", block ? "btn-block" : "", className];
  return (
    <button type="button" className={classes.join(" ")} {...props}>
      {icon}
      {children}
    </button>
  );
}

/** A labelled form field; `error` (or `invalid`) outlines it in red, and `error` is shown under it. */
export function Field({ label, help, invalid, error, extra, children, className = "" }: {
  label?: ReactNode;
  help?: ReactNode;
  invalid?: boolean;
  error?: ReactNode;
  extra?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <div className={`field ${invalid || error ? "invalid" : ""} ${className}`}>
      {label !== undefined && <div className="field-label">{label}<span className="spacer" />{extra}</div>}
      {children}
      {error && <div className="field-error">{error}</div>}
      {help && <div className="field-help">{help}</div>}
    </div>
  );
}

export function Input({ className = "", ...props }: InputHTMLAttributes<HTMLInputElement>) {
  return <input className={`input ${className}`} {...props} />;
}

export function Textarea({ className = "", ...props }: TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={`textarea ${className}`} {...props} />;
}

export interface Option {
  value: string;
  label: ReactNode;
  hint?: ReactNode;
}

const EMPTY = "__empty__";

/** Popup select. An empty-string value maps to the placeholder option, so "unset" stays representable. */
export function Select({ value, onChange, options, placeholder, disabled, ariaLabel }: {
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  placeholder?: string;
  disabled?: boolean;
  ariaLabel?: string;
}) {
  const items = placeholder !== undefined ? [{ value: EMPTY, label: placeholder }, ...options] : options;
  return (
    <RSelect.Root value={value === "" ? EMPTY : value} onValueChange={(v) => onChange(v === EMPTY ? "" : v)} disabled={disabled}>
      <RSelect.Trigger className="select-trigger" aria-label={ariaLabel}>
        <RSelect.Value />
        <RSelect.Icon><ChevronDown size={15} /></RSelect.Icon>
      </RSelect.Trigger>
      <RSelect.Portal>
        <RSelect.Content className="select-content" position="popper" sideOffset={4}>
          <RSelect.Viewport>
            {items.map((o) => (
              <RSelect.Item key={o.value} value={o.value} className="select-item">
                <RSelect.ItemIndicator className="select-item-check"><Check size={14} /></RSelect.ItemIndicator>
                <RSelect.ItemText>{o.label}</RSelect.ItemText>
                {"hint" in o && o.hint && <span className="faint tiny ml-auto">{o.hint}</span>}
              </RSelect.Item>
            ))}
          </RSelect.Viewport>
        </RSelect.Content>
      </RSelect.Portal>
    </RSelect.Root>
  );
}

export function Checkbox({ checked, onChange, label }: { checked: boolean | "indeterminate"; onChange: (v: boolean) => void; label?: string }) {
  return (
    <RCheckbox.Root className="checkbox" checked={checked} onCheckedChange={(v) => onChange(v === true)} aria-label={label}>
      <RCheckbox.Indicator>{checked === "indeterminate" ? <Minus size={12} strokeWidth={3} /> : <Check size={12} strokeWidth={3} />}</RCheckbox.Indicator>
    </RCheckbox.Root>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <RSwitch.Root className="switch" checked={checked} onCheckedChange={onChange} aria-label={label}>
      <RSwitch.Thumb className="switch-thumb" />
    </RSwitch.Root>
  );
}

export function Segmented({ value, onChange, options }: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: ReactNode; disabled?: boolean; title?: string }[];
}) {
  return (
    <ToggleGroup.Root type="single" className="segmented" value={value} onValueChange={(v) => v && onChange(v)}>
      {options.map((o) => (
        <ToggleGroup.Item key={o.value} value={o.value} disabled={o.disabled} title={o.title}>{o.label}</ToggleGroup.Item>
      ))}
    </ToggleGroup.Root>
  );
}
