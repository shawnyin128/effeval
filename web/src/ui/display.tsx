import { Check, CircleAlert, Copy, Info, TriangleAlert } from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import type { Issue } from "../api";
import { useT } from "../i18n";
import { Button } from "./controls";
import { highlight, type Language } from "./highlight";
import { Tip } from "./overlay";

export function PageHeader({ title, actions }: { title: ReactNode; actions?: ReactNode }) {
  return (
    <header className="page-header">
      <h1>{title}</h1>
      {actions && <div className="actions">{actions}</div>}
    </header>
  );
}

export function Card({ title, hint, actions, children, bodyStyle }: {
  title?: ReactNode;
  hint?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  bodyStyle?: React.CSSProperties;
}) {
  return (
    <section className="card">
      {title && (
        <div className="card-head">
          <h2>{title}</h2>
          {hint && <span className="hint">{hint}</span>}
          {actions && <><span className="spacer" />{actions}</>}
        </div>
      )}
      <div className="card-body" style={bodyStyle}>{children}</div>
    </section>
  );
}

export function Badge({ tone, mono, title, children }: { tone?: "accent" | "red" | "green" | "orange"; mono?: boolean; title?: string; children: ReactNode }) {
  return <span className={`badge ${tone ?? ""} ${mono ? "mono" : ""}`} title={title}>{children}</span>;
}

export function Empty({ icon, title, children, action }: { icon: ReactNode; title: string; children?: ReactNode; action?: ReactNode }) {
  return (
    <div className="empty">
      <div className="icon-tile">{icon}</div>
      <h3>{title}</h3>
      {children && <p className="small" style={{ maxWidth: 420 }}>{children}</p>}
      {action}
    </div>
  );
}

/** The three shell steps of a run: write the plan file, run it, push the results package when offline. */
export function CommandSteps({ command }: { command: { write: string; run: string; push: string } }) {
  const t = useT();
  return (
    <div className="command-steps">
      <div className="step-label"><span className="step-num">1</span>{t("Write the plan")}</div>
      <CodeBlock text={command.write} language="bash" />
      <div className="step-label"><span className="step-num">2</span>{t("Run on the compute node")}</div>
      <CodeBlock text={command.run} language="bash" />
      <div className="step-label"><span className="step-num">3</span>{t("Write back results")}<span className="faint tiny">{t("only if the run could not reach the server")}</span></div>
      <CodeBlock text={command.push} language="bash" />
    </div>
  );
}

/** A small "?" that shows `text` on hover; renders nothing without text. */
export function Help({ text }: { text?: string }) {
  if (!text) return null;
  return <Tip content={text}><span className="help-mark" aria-label={text}>?</span></Tip>;
}

export function CodeBlock({ text, language }: { text: string; language: Language }) {
  const t = useT();
  const [copied, setCopied] = useState(false);
  const html = useMemo(() => highlight(text, language), [text, language]);
  return (
    <div className="code">
      <pre className="hljs" dangerouslySetInnerHTML={{ __html: html }} />
      <span className="copy">
        <Button size="sm" icon={copied ? <Check size={13} /> : <Copy size={13} />}
          onClick={async () => {
            await navigator.clipboard.writeText(text);
            setCopied(true);
            setTimeout(() => setCopied(false), 1400);
          }}>
          {copied ? t("Copied") : t("Copy")}
        </Button>
      </span>
    </div>
  );
}

export function IssueList({ issues, show = ["error", "warning"] }: { issues: Issue[]; show?: string[] }) {
  const visible = issues.filter((i) => show.includes(i.severity));
  if (visible.length === 0) return null;
  return (
    <div>
      {visible.map((issue, index) => (
        <div key={index} className={`issue ${issue.severity}`}>
          <span className="issue-icon">
            {issue.severity === "error" ? <CircleAlert size={16} /> : issue.severity === "warning" ? <TriangleAlert size={16} /> : <Info size={16} />}
          </span>
          <div style={{ minWidth: 0 }}>
            <div className="issue-code">{issue.code}{issue.paths.length > 0 && ` · ${issue.paths.join(", ")}`}</div>
            <div className="issue-msg">{issue.message}</div>
            {issue.hint && <div className="issue-hint">{issue.hint}</div>}
          </div>
        </div>
      ))}
    </div>
  );
}

export function SourceBadge({ source }: { source: string }) {
  const t = useT();
  if (source.startsWith("paper")) return <Badge tone="accent" title={source}>{t("paper")}</Badge>;
  if (source.startsWith("code")) return <Badge title={source}>{t("official code")}</Badge>;
  return <Badge tone="orange">{t("assumed")}</Badge>;
}

export function formatSeconds(seconds: number | null): string {
  if (seconds === null) return "—";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  if (seconds < 3600) return `${(seconds / 60).toFixed(1)} min`;
  return `${(seconds / 3600).toFixed(1)} h`;
}

export function formatParams(params: number): string {
  return params >= 1e9 ? `${(params / 1e9).toFixed(2)}B` : `${(params / 1e6).toFixed(0)}M`;
}
