import { Puzzle, Search, Share2 } from "lucide-react";
import { useEffect, useState } from "react";
import { api, fieldTitle, methodTitle, type Manifest, type PreprocessPreset, type PreprocessStep } from "../api";
import { useApp } from "../components/common";
import { MethodConfig, configSections } from "../compose/MethodConfig";
import { useT } from "../i18n";
import { useLive } from "../live";
import { Button, Select } from "../ui/controls";
import { Badge, Card, CodeBlock, PageHeader } from "../ui/display";
import { Dialog, useToast } from "../ui/overlay";

const METHOD_TEMPLATE = `# my_pkg/quant.py
from pydantic import BaseModel, Field
from effeval.core.fragments import QuantRule
from effeval.core.registry import method

class Config(BaseModel):
    rules: list[QuantRule]                      # reuses the quantization rule editor
    iters: int = Field(200, ge=1, description="optimization steps")   # description = hover help

@method("ours_quant", version="0.1", field="quantization", config=Config,
        uses=("quant_rules",), presets={"paper": {"iters": 200}}, default_preset="paper")
def apply(model, config: Config, ctx):
    ...                                         # change the model in place
    return model`;

const ENTRY_POINT = `# pyproject.toml of my_pkg
[project.entry-points."effeval.methods"]
ours_quant = "my_pkg.quant:apply"`;

const SYNC = `pip install -e .            # on any machine that can reach the server
effeval method sync`;

const SOURCE_LABEL: Record<string, string> = {
  hadamard: "Hadamard",
  random_hadamard: "Random Hadamard",
  random_orthogonal: "Random orthogonal",
  duquant: "DuQuant",
  smoothquant: "SmoothQuant",
  awq: "AWQ",
  trained: "Trained",
};

function sourceSummary(source: PreprocessStep["entries"][number]["source"]): string {
  const rest = Object.entries(source).filter(([k, v]) => k !== "type" && v !== null && v !== undefined && typeof v !== "object");
  const label = SOURCE_LABEL[source.type] ?? source.type;
  return rest.length ? `${label} (${rest.map(([k, v]) => `${k} ${v}`).join(", ")})` : label;
}

const unique = (xs: string[]) => [...new Set(xs)];

type Item = { manifest: Manifest; owner: string; visibility: string };

export function Methods() {
  const { me, describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [items, setItems] = useState<Item[]>([]);
  const [query, setQuery] = useState("");
  const [fieldFilter, setFieldFilter] = useState("-");
  const [open, setOpen] = useState<Item | null>(null);
  const load = () => api.methods().then(setItems);
  useLive(["catalog"], load);
  useEffect(() => {
    load();
  }, []);

  const builtins = Object.values(describe.methods).filter((m) => m.builtin).map((m) => ({ manifest: m, owner: "builtin", visibility: "public" }));
  const available = [...builtins, ...items].filter(({ manifest: m, owner }) => {
    const q = query.trim().toLowerCase();
    return (!q || [m.name, m.title ?? "", owner, m.description ?? ""].some((v) => v.toLowerCase().includes(q))) && (fieldFilter === "-" || m.field === fieldFilter);
  });

  const q = query.trim().toLowerCase();
  const preprocessRows = Object.values(describe.preprocess_presets ?? {}).filter((p) =>
    (!q || [p.name, p.title, p.description].some((v) => v.toLowerCase().includes(q))) && (fieldFilter === "-" || fieldFilter === "quantization"));

  return (
    <div className="page">
      <PageHeader title={t("Methods")} />
      <Card title={t("Available")} actions={<div className="row nowrap-row">
        <div className="search"><Search size={14} /><input className="input" placeholder={t("Search")} value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: 200 }} /></div>
        <div className="w-md"><Select value={fieldFilter} onChange={setFieldFilter} options={[{ value: "-", label: t("All fields") }, ...Object.keys(describe.fields).map((f) => ({ value: f, label: f }))]} /></div>
      </div>}>
        <table className="list">
          <thead><tr><th className="col-md">{t("Name")}</th><th className="col-xs">{t("Version")}</th><th className="col-md">{t("Field")}</th><th>{t("Configurable")}</th><th className="col-lg">{t("Preset")}</th><th className="col-md">{t("Owner")}</th><th className="col-sm" /></tr></thead>
          <tbody>
            {available.map((item) => {
              const { manifest: m, owner, visibility } = item;
              const sections = configSections(m);
              const presets = [...Object.keys(describe.presets[m.name] ?? {}), ...Object.keys(m.custom_presets ?? {})];
              return (
                <tr key={m.name} className="clickable" onClick={() => setOpen(item)}>
                  <td className="mono" style={{ fontWeight: 550 }}>{methodTitle(describe, m.name)}</td>
                  <td className="small">{m.version}</td>
                  <td className="small">{fieldTitle(describe, m.field)}</td>
                  <td className="small">{sections.length ? sections.map((name) => t(name)).join(" · ") : <span className="faint">—</span>}</td>
                  <td className="small muted">{presets.length ? presets.join(", ") : "—"}</td>
                  <td className="small">{owner === "builtin" ? t("builtin") : owner} {owner !== "builtin" && <Badge>{t(visibility)}</Badge>}</td>
                  <td className="actions" onClick={(e) => e.stopPropagation()}>
                    {owner === me.username && visibility === "private" &&
                      <Button variant="link" icon={<Share2 size={13} />} onClick={async () => { await api.publishMethod(m.name); toast(t("Shared")); load(); }}>{t("Share")}</Button>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </Card>

      <Card title={t("Model preprocessing")}>
        <table className="list">
          <thead><tr><th className="col-md">{t("Name")}</th><th className="col-md">{t("Kind")}</th><th>{t("Positions")}</th><th>{t("Source")}</th><th className="col-lg">{t("Paper")}</th></tr></thead>
          <tbody>
            {preprocessRows.map((p) => <PreprocessRow key={p.name} preset={p} />)}
          </tbody>
        </table>
      </Card>

      <Card title={t("Add a method")}>
        <div className="command-steps">
          <div className="step-label"><span className="step-num">1</span>{t("Write the function and its config class")}</div>
          <CodeBlock text={METHOD_TEMPLATE} language="python" />
          <div className="step-label"><span className="step-num">2</span>{t("Announce it in the package")}</div>
          <CodeBlock text={ENTRY_POINT} language="toml" />
          <div className="step-label"><span className="step-num">3</span>{t("Sync")}<span className="faint tiny">{t("the form, checks and presets appear here and in New evaluation")}</span></div>
          <CodeBlock text={SYNC} language="bash" />
        </div>
      </Card>

      <Dialog open={!!open} onOpenChange={(o) => !o && setOpen(null)} wide title={<span className="mono">{open?.manifest.title || open?.manifest.name}</span>}
        icon={<div className="icon-tile accent"><Puzzle size={18} /></div>}>
        {open && <MethodDetail item={open} />}
      </Dialog>
    </div>
  );
}

function PreprocessRow({ preset }: { preset: PreprocessPreset }) {
  const t = useT();
  const kinds = unique(preset.steps.map((step) => t(step.kind === "rotation" ? "Rotation" : "Scaling")));
  const positions = (["offline", "online"] as const).map((mode) => ({
    mode,
    ids: unique(preset.steps.flatMap((step) => step.entries.filter((e) => e.mode === mode).map((e) => e.position))),
  })).filter((g) => g.ids.length);
  const sources = unique(preset.steps.map((step) => unique(step.entries.map((e) => sourceSummary(e.source))).join(", ")));
  return (
    <tr>
      <td title={preset.description}>
        <div style={{ fontWeight: 550 }}>{preset.title}</div>
      </td>
      <td className="small">{kinds.join(" + ")}</td>
      <td className="small">
        {positions.map(({ mode, ids }) => <div key={mode}><span className="muted">{t(mode)}</span> <span className="mono">{ids.join(", ")}</span></div>)}
      </td>
      <td className="small">{sources.join(" → ")}</td>
      <td className="small"><a href={`https://arxiv.org/abs/${preset.paper.arxiv}`} target="_blank" rel="noreferrer">{preset.paper.title}</a></td>
    </tr>
  );
}

function MethodDetail({ item }: { item: Item }) {
  const { describe } = useApp();
  const t = useT();
  const m = item.manifest;
  const preset = m.default_preset ? describe.presets[m.name]?.[m.default_preset] ?? null : null;
  return (
    <div className="stack">
      <dl className="kv">
        <dt>{t("Entry")}</dt><dd className="mono">{m.entry ?? "—"}</dd>
        <dt>{t("Version")}</dt><dd>{m.version}</dd>
        <dt>{t("Field")}</dt><dd>{fieldTitle(describe, m.field)}</dd>
        <dt>{t("Architectures")}</dt><dd>{m.architectures?.join(", ") ?? t("any")}</dd>
        {m.description && <><dt>{t("Description")}</dt><dd>{m.description}</dd></>}
      </dl>
      <div>
        <div className="field-label mb-2">{t("Form in New evaluation")}</div>
        <MethodConfig manifest={m} preset={preset} config={{}} onChange={() => {}} invalidPaths={new Set()} sites={[]} />
      </div>
    </div>
  );
}
