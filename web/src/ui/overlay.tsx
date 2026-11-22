import { CircleAlert, CircleCheck, Info, TriangleAlert, X } from "lucide-react";
import { Dialog as RDialog, Toast as RToast, Tooltip as RTooltip } from "radix-ui";
import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import type { Issue } from "../api";
import { useT } from "../i18n";
import { Button } from "./controls";

export function Dialog({ open, onOpenChange, title, description, icon, wide, footer, children }: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: ReactNode;
  description?: ReactNode;
  icon?: ReactNode;
  wide?: boolean;
  footer?: ReactNode;
  children?: ReactNode;
}) {
  const t = useT();
  return (
    <RDialog.Root open={open} onOpenChange={onOpenChange}>
      <RDialog.Portal>
        <RDialog.Overlay className="overlay" />
        <RDialog.Content className={`dialog ${wide ? "wide" : ""}`} aria-describedby={undefined}>
          <div className="dialog-head">
            {icon}
            <div className="grow">
              <RDialog.Title asChild><h2>{title}</h2></RDialog.Title>
              {description && <p>{description}</p>}
            </div>
            <RDialog.Close asChild><Button variant="ghost" size="sm" icon={<X size={15} />} aria-label={t("Close")} /></RDialog.Close>
          </div>
          {children && <div className="dialog-body">{children}</div>}
          {footer && <div className="dialog-foot">{footer}</div>}
        </RDialog.Content>
      </RDialog.Portal>
    </RDialog.Root>
  );
}

export function Tip({ content, children }: { content: ReactNode; children: ReactNode }) {
  return (
    <RTooltip.Root delayDuration={250}>
      <RTooltip.Trigger asChild>{children}</RTooltip.Trigger>
      <RTooltip.Portal>
        <RTooltip.Content className="tooltip" sideOffset={6}>{content}</RTooltip.Content>
      </RTooltip.Portal>
    </RTooltip.Root>
  );
}

type Tone = "success" | "error" | "warning" | "info";
export interface NoticeOptions {
  tone?: Tone;
  description?: string;
  issues?: Issue[];
  /** Notices sharing a key (or, without a key, the same title and tone) replace each other instead of stacking. */
  key?: string;
}
interface Notice extends NoticeOptions {
  id: number;
  /** Changes on every post, so a merged duplicate restarts its countdown. */
  posted: number;
  tone: Tone;
  title: string;
}

const ToastContext = createContext<{ show: (title: string, opts?: NoticeOptions) => void; dismiss: (key: string) => void }>({
  show: () => {},
  dismiss: () => {},
});

/** Returns `show(title, opts)`, which posts a banner at the top of the window that fades by itself. */
export function useToast() {
  return useContext(ToastContext).show;
}

/** Returns `dismiss(key)`, which closes the banner posted with that key, if any. */
export function useDismissToast() {
  return useContext(ToastContext).dismiss;
}

const ISSUE_PREVIEW = 3;
const DURATION: Record<Tone, number> = { success: 3000, info: 3000, warning: 8000, error: 10000 };

function NoticeIssues({ issues }: { issues: Issue[] }) {
  const t = useT();
  const [all, setAll] = useState(false);
  const shown = all ? issues : issues.slice(0, ISSUE_PREVIEW);
  return (
    <ul className="notice-issues">
      {shown.map((i, n) => <li key={n}><span className="mono">{i.paths[0] ?? i.code}</span> {i.message}</li>)}
      {issues.length > ISSUE_PREVIEW && (
        <li><button className="link-button" onClick={() => setAll(!all)}>{all ? t("Show less") : t("{n} more", { n: issues.length - ISSUE_PREVIEW })}</button></li>
      )}
    </ul>
  );
}

/** Closes `notice` after its tone's duration, counted from its latest post; renders nothing. */
function AutoClose({ notice, onClose }: { notice: Notice; onClose: (id: number) => void }) {
  useEffect(() => {
    const timer = window.setTimeout(() => onClose(notice.id), DURATION[notice.tone]);
    return () => window.clearTimeout(timer);
  }, [notice.id, notice.posted, notice.tone, onClose]);
  return null;
}

export function UiProvider({ children }: { children: ReactNode }) {
  const t = useT();
  const [items, setItems] = useState<Notice[]>([]);
  const show = useCallback((title: string, opts: NoticeOptions = {}) => {
    setItems((list) => {
      const posted = Date.now() + Math.random();
      const notice = { ...opts, id: posted, posted, tone: opts.tone ?? "success", title };
      const at = list.findIndex((x) => (opts.key ? x.key === opts.key : !x.key && x.title === title && x.tone === notice.tone));
      if (at < 0) return [...list, notice];
      return list.map((x, i) => (i === at ? { ...notice, id: x.id } : x));
    });
  }, []);
  const dismiss = useCallback((key: string) => setItems((list) => list.filter((x) => x.key !== key)), []);
  const remove = useCallback((id: number) => setItems((list) => list.filter((x) => x.id !== id)), []);
  const value = useMemo(() => ({ show, dismiss }), [show, dismiss]);
  const icons = {
    success: <CircleCheck size={17} color="var(--green)" />,
    error: <CircleAlert size={17} color="var(--red)" />,
    warning: <TriangleAlert size={17} color="var(--orange)" />,
    info: <Info size={17} color="var(--accent)" />,
  };
  return (
    <ToastContext.Provider value={value}>
      <RTooltip.Provider>
        <RToast.Provider swipeDirection="up">
          {children}
          {items.map((n) => (
            <RToast.Root key={n.id} className={`notice ${n.tone}`} duration={Infinity}
              onOpenChange={(open) => !open && remove(n.id)}>
              <AutoClose notice={n} onClose={remove} />
              <span className="notice-icon">{icons[n.tone]}</span>
              <div className="grow">
                <RToast.Title className="notice-title">{n.title}</RToast.Title>
                {n.description && <RToast.Description className="notice-desc">{n.description}</RToast.Description>}
                {n.issues && n.issues.length > 0 && <NoticeIssues issues={n.issues} />}
              </div>
              <RToast.Close className="notice-close" aria-label={t("Close")}><X size={14} /></RToast.Close>
            </RToast.Root>
          ))}
          <RToast.Viewport className="notice-viewport" />
        </RToast.Provider>
      </RTooltip.Provider>
    </ToastContext.Provider>
  );
}
