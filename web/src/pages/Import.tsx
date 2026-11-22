import { CircleAlert, CircleCheck, FileJson, FolderUp, LoaderCircle, Upload, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { api, type Row } from "../api";
import { ResultsSheet } from "../components/ResultsSheet";
import { useT } from "../i18n";
import { Button, Textarea } from "../ui/controls";
import { Badge, Card, PageHeader } from "../ui/display";
import { DestinationSelect, useDestination } from "./Projects";
import { useToast } from "../ui/overlay";

interface Item {
  id: number;
  name: string;
  text: string;
  file?: File;
  withSamples?: number;
  state: "checking" | "ok" | "error";
  rows: Row[];
  error: string;
}

/** Splits text holding one or more concatenated JSON documents into the text of each top-level object. */
export function splitDocuments(text: string): string[] {
  const out: string[] = [];
  let depth = 0;
  let start = -1;
  let inString = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (ch === "\\") i++;
      else if (ch === '"') inString = false;
    } else if (ch === '"') inString = true;
    else if (ch === "{") {
      if (depth++ === 0) start = i;
    } else if (ch === "}" && depth > 0 && --depth === 0) out.push(text.slice(start, i + 1));
  }
  if (depth !== 0 || (!out.length && text.trim())) out.push(text.slice(start < 0 ? 0 : start));
  return out;
}

let nextId = 1;

export function Import() {
  const t = useT();
  const toast = useToast();
  const [items, setItems] = useState<Item[]>([]);
  const [paste, setPaste] = useState("");
  const [dragging, setDragging] = useState(false);
  const [busy, setBusy] = useState(false);
  const [to, setTo] = useDestination();
  const picker = useRef<HTMLInputElement>(null);

  const add = (docs: { name: string; text: string; file?: File }[]) => {
    const fresh = docs.map((d): Item => ({ id: nextId++, ...d, state: "checking", rows: [], error: "" }));
    setItems((list) => [...list, ...fresh]);
    fresh.forEach(async (item) => {
      try {
        const { rows, with_samples } = item.file ? await api.previewPackage(item.file) : { ...(await api.previewBundle(item.text)), with_samples: undefined };
        setItems((list) => list.map((x) => (x.id === item.id ? { ...x, state: "ok", rows, withSamples: with_samples } : x)));
      } catch (e) {
        setItems((list) => list.map((x) => (x.id === item.id ? { ...x, state: "error", error: (e as Error).message } : x)));
      }
    });
  };
  const addFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    add(await Promise.all([...files].map(async (f) => (/\.(tar\.gz|tgz)$/.test(f.name) ? { name: f.name, text: "", file: f } : { name: f.name, text: await f.text() }))));
  };
  const addPaste = () => {
    const docs = splitDocuments(paste);
    add(docs.map((text, i) => ({ name: docs.length > 1 ? `${t("Pasted")} ${i + 1}` : t("Pasted"), text })));
    setPaste("");
  };

  const ready = items.filter((i) => i.state === "ok");
  const rows = useMemo(() => ready.flatMap((i) => i.rows), [items]);
  const importAll = async () => {
    setBusy(true);
    let imported = 0;
    let duplicates = 0;
    let redone = 0;
    for (const item of ready) {
      try {
        const result = item.file ? await api.importPackage(item.file, to) : await api.importBundle(item.text, to);
        imported += result.imported.length;
        duplicates += result.duplicates.length;
        redone += result.redone.length;
        setItems((list) => list.filter((x) => x.id !== item.id));
      } catch (e) {
        setItems((list) => list.map((x) => (x.id === item.id ? { ...x, state: "error", error: (e as Error).message } : x)));
      }
    }
    setBusy(false);
    const notes = [duplicates && t("{n} already stored", { n: duplicates }), redone && t("{n} redo results replaced", { n: redone })].filter(Boolean);
    toast(t("Imported {n}", { n: imported }), notes.length ? { description: notes.join(" · ") } : undefined);
  };

  return (
    <div className="page">
      <PageHeader title={t("Import")} actions={<>
        <div className="w-md"><DestinationSelect value={to} onChange={setTo} /></div>
        <Button variant="primary" icon={<Upload size={14} />} disabled={!ready.length || busy} onClick={importAll}>
          {ready.length > 1 ? t("Import {n} bundles", { n: ready.length }) : t("Import")}
        </Button>
      </>} />
      <div className="import-grid">
        <div
          className={`dropzone ${dragging ? "over" : ""}`}
          onClick={() => picker.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => { e.preventDefault(); setDragging(false); addFiles(e.dataTransfer.files); }}
        >
          <FolderUp size={22} />
          <div className="dropzone-title">{t("Drop bundle files")}</div>
          <div className="faint small">.json · .results.tar.gz</div>
          <input ref={picker} type="file" accept=".json,application/json,.gz,.tgz" multiple hidden
            onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }} />
        </div>
        <Card>
          <Textarea className="mono" placeholder={'{ "bundle_version": 1, "records": [ … ], "checksum": "sha256:…" }\n{ … }'} value={paste}
            onChange={(e) => setPaste(e.target.value)} style={{ minHeight: 132 }} />
          <div className="row mt-3">
            <span className="spacer" />
            <Button disabled={!paste.trim()} onClick={addPaste}>{t("Add")}</Button>
          </div>
        </Card>
      </div>
      {items.length > 0 && (
        <Card>
          <table className="list">
            <tbody>
              {items.map((item) => (
                <tr key={item.id}>
                  <td className="col-check">
                    {item.state === "checking" ? <LoaderCircle size={15} className="spin" color="var(--text-3)" />
                      : item.state === "ok" ? <CircleCheck size={15} color="var(--green)" /> : <CircleAlert size={15} color="var(--red)" />}
                  </td>
                  <td><span className="row nowrap-row"><FileJson size={14} color="var(--text-3)" /><span className="mono small">{item.name}</span></span></td>
                  <td className="small muted">
                    {item.state === "ok" ? t("{n} rows", { n: item.rows.length }) : item.error}
                    {item.withSamples !== undefined && <> · {t("{n} with per-sample files", { n: item.withSamples })}</>}
                    {item.rows.some((r) => r.replaces?.length) && <> <Badge tone="orange">{t("redo of {n}", { n: item.rows.filter((r) => r.replaces?.length).length })}</Badge></>}
                  </td>
                  <td className="col-check actions">
                    <Button variant="ghost" size="sm" icon={<X size={14} />} aria-label={t("Remove")}
                      onClick={() => setItems((list) => list.filter((x) => x.id !== item.id))} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
      {rows.length > 0 && <div className="mt-4" style={{ display: "flex", minHeight: 360 }}><ResultsSheet rows={rows} scope="preview" /></div>}
    </div>
  );
}
