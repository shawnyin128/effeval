import { Cpu, Search } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type ModelEntry } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { useLive } from "../live";
import { Button, Checkbox, Field, Input, Segmented, Select } from "../ui/controls";
import { Badge, Card, PageHeader, formatParams } from "../ui/display";
import { useToast } from "../ui/overlay";

export function Models() {
  const { describe, reloadDescribe } = useApp();
  const t = useT();
  const [models, setModels] = useState<{ model: ModelEntry; registered_by: string }[] | null>(null);
  const [query, setQuery] = useState("");
  const [architecture, setArchitecture] = useState("-");
  const [addedBy, setAddedBy] = useState("-");
  const shown = (models ?? []).filter(({ model: m, registered_by }) => {
    const q = query.trim().toLowerCase();
    return (!q || [m.code, m.checkpoint, m.description].some((v) => v.toLowerCase().includes(q)))
      && (architecture === "-" || m.architecture === architecture) && (addedBy === "-" || registered_by === addedBy);
  });
  const choices = (values: string[]) => [...new Set(values)].sort().map((v) => ({ value: v, label: v === "builtin" ? t("builtin") : v }));
  useEffect(() => {
    api.models().then(setModels);
  }, []);
  useLive(["catalog"], () => api.models().then(setModels));
  return (
    <div className="page">
      <PageHeader title={t("Models")} />
      <Card title={t("Registry")} hint={models ? (shown.length === models.length ? String(models.length) : `${shown.length} / ${models.length}`) : undefined}
        actions={<div className="row nowrap-row">
          <div className="search"><Search size={14} /><input className="input" placeholder={t("Search")} value={query} onChange={(e) => setQuery(e.target.value)} style={{ width: 200 }} /></div>
          <div className="w-md"><Select value={architecture} onChange={setArchitecture} options={[{ value: "-", label: t("All architectures") }, ...choices((models ?? []).map((x) => x.model.architecture))]} /></div>
          <div className="w-sm"><Select value={addedBy} onChange={setAddedBy} options={[{ value: "-", label: t("Anyone") }, ...choices((models ?? []).map((x) => x.registered_by))]} /></div>
        </div>}>
        <table className="list">
          <thead><tr><th className="col-lg">{t("Code")}</th><th>{t("Checkpoint")}</th><th className="col-md">{t("Revision")}</th><th className="col-lg">{t("Architecture")}</th><th className="col-sm">{t("dtype")}</th><th className="col-sm right">{t("Params")}</th><th className="col-sm">lm_head</th><th className="col-sm">{t("Added by")}</th></tr></thead>
          <tbody>
            {shown.map(({ model: m, registered_by }) => (
              <tr key={m.code}>
                <td className="mono" style={{ fontWeight: 550 }}>{m.code}</td>
                <td className="mono small muted">{m.checkpoint}</td>
                <td>{m.revision ? <span className="mono small" title={m.revision}>{m.revision.slice(0, 10)}</span> : <Badge tone="orange">{t("unpinned")}</Badge>}</td>
                <td><Badge>{m.architecture}</Badge></td>
                <td className="small">{m.dtype}</td>
                <td className="num small right">{formatParams(m.params)}</td>
                <td>{m.tie_word_embeddings ? <Badge tone="orange">{t("tied")}</Badge> : <span className="faint small">—</span>}</td>
                <td className="small muted">{registered_by === "builtin" ? t("builtin") : registered_by}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </Card>
      <div className="grid-2 mt-4" style={{ alignItems: "start" }}>
        <Card title={t("Register")}><RegisterModel onRegistered={async () => { await reloadDescribe(); setModels(await api.models()); }} /></Card>
        <Card title={t("Architectures")}>
          {Object.values(describe.architectures).map((a) => (
            <div key={a.name}>
              <div className="row mb-3">
                <span className="icon-tile accent" style={{ width: 28, height: 28, borderRadius: 8 }}><Cpu size={15} /></span>
                <strong>{a.name}</strong>
                <span className="spacer" />
                <span className="faint tiny">transformers {a.extracted_with.transformers}</span>
              </div>
              <dl className="kv">
                <dt>{t("Decoder layers")}</dt><dd className="mono">{a.layers_path}</dd>
                <dt>{t("Attention")}</dt><dd><span className="mono">{a.attention_path ?? "—"}</span> {a.supports_attention_backend && <Badge tone="accent">QKᵀ</Badge>}</dd>
                <dt>{t("Linear sites")}</dt>
                <dd className="row gap-1">{Object.values(a.linears).map((c) => <Badge key={c} mono>{c}</Badge>)}</dd>
                {a.uncovered_weights.length > 0 && <><dt>{t("Not covered")}</dt><dd className="row gap-1">{a.uncovered_weights.map((w) => <Badge key={w} tone="orange" mono>{w}</Badge>)}</dd></>}
              </dl>
            </div>
          ))}
        </Card>
      </div>
    </div>
  );
}

const DTYPES = ["bfloat16", "float16", "float32"];

function codeOf(checkpoint: string): string {
  return checkpoint.replace(/\/+$/, "").split("/").pop()?.toLowerCase() ?? "";
}

/** Register a model from a Hugging Face repo (looked up by the server) or a local checkpoint path (by hand). */
function RegisterModel({ onRegistered }: { onRegistered: () => Promise<void> }) {
  const { describe } = useApp();
  const t = useT();
  const toast = useToast();
  const [source, setSource] = useState("hub");
  const [repo, setRepo] = useState("");
  const [revision, setRevision] = useState("");
  const [found, setFound] = useState<ModelEntry | null>(null);
  const [local, setLocal] = useState({ checkpoint: "", architecture: Object.keys(describe.architectures)[0] ?? "", dtype: "bfloat16", params: "", tied: false });
  const fail = (e: unknown) => toast(t("Action failed"), { tone: "error", description: (e as Error).message });
  const register = async (entry: ModelEntry) => {
    try {
      await api.registerModel(entry);
      toast(`${t("Registered")} ${entry.code}`);
      setFound(null);
      setRepo("");
      setLocal({ ...local, checkpoint: "", params: "" });
      await onRegistered();
    } catch (e) {
      fail(e);
    }
  };
  return (
    <div className="stack">
      <Segmented value={source} onChange={setSource} options={[{ value: "hub", label: "Hugging Face" }, { value: "local", label: t("Local path") }]} />
      {source === "hub" ? (
        <>
          <div className="row nowrap-row" style={{ alignItems: "flex-end" }}>
            <div style={{ flex: 2 }}><Field label={t("Repo")}><Input className="mono" placeholder="Qwen/Qwen3-8B-Base" value={repo} onChange={(e) => { setRepo(e.target.value); setFound(null); }} /></Field></div>
            <div className="grow"><Field label={t("Revision")}><Input className="mono" placeholder="main" value={revision} onChange={(e) => { setRevision(e.target.value); setFound(null); }} /></Field></div>
            <Button disabled={!repo.trim()} onClick={async () => { try { setFound(await api.lookupModel(repo.trim(), revision.trim())); } catch (e) { fail(e); } }}>{t("Look up")}</Button>
          </div>
          {found && (
            <div className="subtle-box" style={{ padding: 12 }}>
              <dl className="kv">
                <dt>{t("Code")}</dt><dd className="mono">{found.code}</dd>
                <dt>{t("Architecture")}</dt><dd>{found.architecture}</dd>
                <dt>{t("Params")}</dt><dd>{formatParams(found.params)}</dd>
                <dt>{t("dtype")}</dt><dd>{found.dtype}</dd>
                <dt>{t("Revision")}</dt><dd className="mono">{found.revision?.slice(0, 12)}</dd>
                <dt>lm_head</dt><dd>{found.tie_word_embeddings ? t("tied") : "—"}</dd>
              </dl>
              <div className="row mt-3"><span className="spacer" /><Button variant="primary" onClick={() => register(found)}>{t("Register")}</Button></div>
            </div>
          )}
        </>
      ) : (
        <>
          <Field label={t("Checkpoint")}><Input className="mono" placeholder="/data/ckpt/My-Finetune" value={local.checkpoint} onChange={(e) => setLocal({ ...local, checkpoint: e.target.value })} /></Field>
          <div className="grid-3">
            <Field label={t("Architecture")}><Select value={local.architecture} onChange={(v) => setLocal({ ...local, architecture: v })} options={Object.keys(describe.architectures).map((a) => ({ value: a, label: a }))} /></Field>
            <Field label={t("dtype")}><Select value={local.dtype} onChange={(v) => setLocal({ ...local, dtype: v })} options={DTYPES.map((d) => ({ value: d, label: d }))} /></Field>
            <Field label={t("Params")}><Input type="number" placeholder="8190000000" value={local.params} onChange={(e) => setLocal({ ...local, params: e.target.value })} /></Field>
          </div>
          <label className="row small clickable"><Checkbox checked={local.tied} onChange={(v) => setLocal({ ...local, tied: v })} />{t("lm_head shares the input embedding")}</label>
          <div className="row"><span className="faint small mono">{local.checkpoint ? `→ ${codeOf(local.checkpoint)}` : ""}</span><span className="spacer" />
            <Button variant="primary" disabled={!local.checkpoint.trim() || !Number(local.params)} onClick={() => register({
              code: codeOf(local.checkpoint), checkpoint: local.checkpoint.trim(), revision: null, architecture: local.architecture,
              dtype: local.dtype, params: Number(local.params), tie_word_embeddings: local.tied, description: "",
            })}>{t("Register")}</Button>
          </div>
        </>
      )}
      <div className="faint tiny">{t("Or from a compute node:")} <span className="mono">effeval model add &lt;repo or path&gt;</span></div>
    </div>
  );
}
