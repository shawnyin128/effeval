import { Check, ChevronDown, Search } from "lucide-react";
import { Popover } from "radix-ui";
import { useMemo, useState } from "react";
import type { ModelEntry } from "../api";
import { useT } from "../i18n";
import { Badge, formatParams } from "../ui/display";

/** Searchable picker over the global model registry. */
export function ModelPicker({ models, value, onChange, invalid }: {
  models: Record<string, ModelEntry>;
  value: string;
  onChange: (code: string) => void;
  invalid?: boolean;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const selected = models[value];
  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    const hits = Object.values(models).filter((m) =>
      !q || [m.code, m.checkpoint, m.architecture, m.description].some((v) => v.toLowerCase().includes(q)));
    const groups = new Map<string, ModelEntry[]>();
    hits.sort((a, b) => a.params - b.params || a.code.localeCompare(b.code))
      .forEach((m) => groups.set(m.architecture, [...(groups.get(m.architecture) ?? []), m]));
    return [...groups].sort(([a], [b]) => a.localeCompare(b));
  }, [models, query]);
  return (
    <Popover.Root open={open} onOpenChange={setOpen}>
      <Popover.Trigger asChild>
        <button className="select-trigger" style={invalid ? { borderColor: "var(--red)" } : undefined}>
          {selected ? (
            <span className="row gap-2 nowrap-row" style={{ minWidth: 0 }}>
              <span className="mono text-sm">{selected.code}</span>
              <span className="faint small truncate">{selected.checkpoint}</span>
            </span>
          ) : <span className="faint">{value || t("Choose a model")}</span>}
          <ChevronDown size={15} />
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content className="popover-content" sideOffset={4} align="start" style={{ width: 520 }}>
          <div className="search" style={{ padding: 8, borderBottom: "1px solid var(--line)" }}>
            <Search size={14} style={{ left: 18 }} />
            <input className="input" autoFocus placeholder={t("Search")} value={query}
              onChange={(e) => setQuery(e.target.value)} style={{ width: "100%" }} />
          </div>
          <div style={{ maxHeight: 340, overflow: "auto", padding: 5 }}>
            {list.length === 0 && <div className="faint small center" style={{ padding: 16 }}>{t("No match")}</div>}
            {list.map(([architecture, entries]) => (
              <div key={architecture}>
                <div className="menu-label row gap-2">{architecture}<span className="faint">{entries.length}</span></div>
                {entries.map((m) => (
              <button key={m.code} className="menu-item" style={{ width: "100%", border: "none", background: "none", font: "inherit", padding: "7px 10px", gap: 10 }}
                onClick={() => { onChange(m.code); setOpen(false); setQuery(""); }}>
                <span style={{ width: 16, display: "inline-flex" }}>{m.code === value && <Check size={14} />}</span>
                <span className="grow" style={{ display: "flex", flexDirection: "column", alignItems: "flex-start" }}>
                  <span className="mono text-sm">{m.code}</span>
                  <span className="faint tiny">{m.checkpoint}</span>
                </span>
                <Badge mono>{formatParams(m.params)}</Badge>
                {m.tie_word_embeddings && <Badge>{t("tied")}</Badge>}
              </button>
                ))}
              </div>
            ))}
          </div>
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
