import type { ReactNode } from "react";
import type { Spec } from "./model";

interface Line {
  indent: number;
  bullet: boolean;
  cont: boolean;
  key: string;
  value?: unknown;
  path: string;
}

const isRecord = (v: unknown): v is Spec => !!v && typeof v === "object" && !Array.isArray(v);
const isRecordList = (v: unknown): v is Spec[] => Array.isArray(v) && v.some((x) => isRecord(x));

/** Lines of one `key: value` pair; `lead` is "bullet" or "cont" for the keys of a list item. */
function pair(key: string, value: unknown, path: string, indent: number, lead: "bullet" | "cont" | null, out: Line[]) {
  const line = { indent, bullet: lead === "bullet", cont: lead === "cont", key, path };
  const below = lead ? indent + 3 : indent + 1;
  if (isRecord(value) && Object.keys(value).length > 0) {
    out.push(line);
    lines(value, path, below, out);
  } else if (isRecordList(value)) {
    out.push(line);
    for (const item of value) listItem(item, path, below, out);
  } else {
    out.push({ ...line, value });
  }
}

function listItem(item: Spec, path: string, indent: number, out: Line[]) {
  Object.entries(item).forEach(([k, v], i) => pair(k, v, path, indent, i === 0 ? "bullet" : "cont", out));
}

function lines(node: Spec, path: string, indent: number, out: Line[]): Line[] {
  for (const [key, value] of Object.entries(node)) pair(key, value, path ? `${path}.${key}` : key, indent, null, out);
  return out;
}

function scalar(value: unknown): ReactNode {
  if (value === null || value === undefined) return <span className="y-b">null</span>;
  if (typeof value === "boolean") return <span className="y-b">{String(value)}</span>;
  if (typeof value === "number") return <span className="y-n">{String(value)}</span>;
  if (Array.isArray(value)) {
    return <>[{value.map((v, i) => <span key={i}>{i > 0 && ", "}{scalar(v)}</span>)}]</>;
  }
  if (value && typeof value === "object") return <span className="y-b">{"{}"}</span>;
  const text = String(value);
  const quote = text === "" || /[:#{}[\],&*?|<>=!%@`]|^\s|\s$/.test(text) || /^(true|false|null|-?\d+(\.\d+)?)$/.test(text);
  return <span className="y-s">{quote ? JSON.stringify(text) : text}</span>;
}

/** A config rendered as highlighted YAML. `sources` annotates paths; `changed` marks paths that differ. */
export function ConfigView({ config, sources = {}, changed = new Set<string>() }: {
  config: Spec;
  sources?: Record<string, string>;
  changed?: Set<string>;
}) {
  const all = lines(config, "", 0, []);
  const isChanged = (path: string) => changed.has(path) || [...changed].some((c) => c.startsWith(`${path}.`));
  const shownSource = new Set<string>();
  return (
    <div className="yaml">
      {all.map((line, i) => {
        const source = sources[line.path] && !shownSource.has(line.path) ? sources[line.path] : undefined;
        if (source) shownSource.add(line.path);
        const mark = line.value !== undefined || !line.key ? isChanged(line.path) : changed.has(line.path);
        return (
          <div key={i} className={`yaml-line ${mark ? "changed" : ""}`}>
            <span className="yaml-gutter" />
            <span style={{ paddingLeft: line.indent * 16 }}>
              {line.bullet && <span className="y-p">- </span>}
              {line.cont && <span>{"  "}</span>}
              <span className="y-k">{line.key}</span><span className="y-p">:</span>
              {line.value !== undefined && <> {scalar(line.value)}</>}
              {source && <span className="y-c" title={source}>  # {source.length > 72 ? `${source.slice(0, 72)}…` : source}</span>}
            </span>
          </div>
        );
      })}
    </div>
  );
}
