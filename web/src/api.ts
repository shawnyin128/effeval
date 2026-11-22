// Thin client over the effeval server API (session cookie auth).
import type { LayerGraph, PositionTable } from "./compose/ArchDiagram";

export type Severity = "error" | "warning" | "info";
export interface Issue {
  code: string;
  severity: Severity;
  paths: string[];
  message: string;
  hint: string;
}

export interface Me {
  username: string;
  role: "admin" | "member";
  can_publish: boolean;
  locale: "en" | "zh";
}

export interface JsonSchema {
  type?: string;
  properties?: Record<string, JsonSchema>;
  required?: string[];
  enum?: unknown[];
  const?: unknown;
  default?: unknown;
  description?: string;
  minimum?: number;
  maximum?: number;
  items?: JsonSchema;
  $ref?: string;
  anyOf?: JsonSchema[];
  examples?: unknown[];
  $defs?: Record<string, JsonSchema>;
}

export interface Manifest {
  name: string;
  /** Display name with its standard capitalisation, e.g. "GPTQ"; absent for methods that declare none. */
  title?: string | null;
  version: string;
  field: string;
  entry: string | null;
  uses: string[];
  config_schema: JsonSchema;
  display: string[];
  default_preset: string | null;
  custom_presets?: Record<string, Record<string, unknown>>;
  architectures?: string[] | null;
  builtin: boolean;
  description: string;
}

export interface PresetValue {
  value: unknown;
  source: string;
}
export interface Preset {
  method: string;
  name: string;
  description: string;
  paper: { title: string; arxiv: string; official_repo: string };
  values: Record<string, PresetValue>;
  reported: { model: string; task: string; metric: string; value: string; table: string; tolerance: string }[];
}

export interface TaskMetric {
  name: string;
  higher_is_better: boolean | null;
  stderr_of: string | null;
}

export interface TaskDef {
  /** `<evaluator>/<target>`, the identity of the task everywhere. */
  name: string;
  evaluator: string;
  /** The evaluator's own name for the task. */
  target: string;
  category: string;
  description: string;
  modes: string[];
  default_mode: string;
  produces: Record<string, string[]>;
  /** Default few-shot count; null: subtasks keep their own. */
  num_fewshot: number | null;
  /** Accepted num_fewshot overrides; null accepts any, [] accepts none. */
  fewshot_allowed: number[] | null;
  /** The first is primary. */
  metrics: TaskMetric[];
  sample_scores: string[];
  version: string;
  reference: boolean;
  subsets: string[] | null;
  subtasks: string[];
  tags: string[];
  /** JSON schema of the benchmark's settings; defaults are the paper protocol. */
  params_schema: JsonSchema;
}

/** A selected task as written in a RunSpec; `params` holds only changed settings. */
export interface TaskRef {
  name: string;
  mode?: string;
  params?: Record<string, unknown>;
}

export interface ModelEntry {
  code: string;
  checkpoint: string;
  revision: string | null;
  architecture: string;
  dtype: string;
  params: number;
  tie_word_embeddings: boolean;
  description: string;
  shape?: ModelShape | null;
  /** Preprocessing availability per position, kind and mode ("ok" or the reason); null when the model has no inspected graph. */
  positions?: PositionTable | null;
}

export interface ModelShape {
  hidden: number;
  intermediate: number;
  num_layers: number;
  heads: number;
  kv_heads: number;
  head_dim: number;
  vocab: number;
  tie_word_embeddings: boolean;
}

export interface Architecture {
  name: string;
  layers_path: string;
  linears: Record<string, string>;
  attention_path: string | null;
  supports_attention_backend: boolean;
  uncovered_weights: string[];
  extracted_with: Record<string, string>;
  sites: string[];
  graph?: LayerGraph | null;
}

/** One position a preprocessing step acts on, with how (offline / online) and where its matrix or scale comes from. */
export interface PreprocessEntry {
  position: string;
  layers: unknown;
  mode: "offline" | "online";
  source: Record<string, any> & { type: string };
  options: Record<string, unknown>;
}

export interface PreprocessStep {
  kind: "rotation" | "scaling";
  calibration?: Record<string, any> | null;
  entries: PreprocessEntry[];
}

/** RunSpec.preprocess: a preset name, explicit steps, or both (explicit steps win over the preset's). */
export interface PreprocessSpec {
  preset?: string;
  steps?: PreprocessStep[];
}

export interface PreprocessPreset {
  name: string;
  title: string;
  description: string;
  paper: { title: string; arxiv: string; official_repo: string };
  steps: PreprocessStep[];
  versions: Record<string, string>;
}

export interface PreprocessSource {
  version: string;
  needs_data: boolean;
  implemented: boolean;
  schema: JsonSchema;
  /** Position globs (`*` wildcard) this source may be used at; absent or null means any position. */
  positions?: string[] | null;
  /** Position globs at which this source has no published method. */
  no_paper?: string[] | null;
}

export interface MetricDef {
  name: string;
  label: string;
  kind: "quality" | "system";
  description: string;
  unit: string;
  format: { style: "fixed" | "percent" | "scientific" | "integer" | "duration"; decimals: number };
  higher_is_better: boolean | null;
  placement: "column" | "detail";
}

export interface EvaluatorDef {
  name: string;
  version: string;
  description: string;
  source: "builtin" | "listing";
  library: string | null;
  library_version: string | null;
}

export interface Describe {
  methods: Record<string, Manifest>;
  presets: Record<string, Record<string, Preset>>;
  tasks: Record<string, TaskDef>;
  fields: Record<string, { name: string; description: string; modes: string[]; compare: string[]; roles?: { name: string; path: string }[] }>;
  evaluators: Record<string, EvaluatorDef>;
  /** Tasks pinned at the top of the task picker. */
  common_tasks: string[];
  task_catalogs: { evaluator: string; library_version: string | null; tasks: number; uploaded_by: string; created_at: string }[];
  gpus: Record<string, { name: string; aliases: string[] }>;
  /** Inference providers by RunSpec `runtime.name`; `available` is whether this machine can run it. */
  runtimes?: Record<string, { label: string; library: string | null; available: boolean }>;
  comparative_metrics: Record<string, string>;
  models: Record<string, ModelEntry>;
  architectures: Record<string, Architecture>;
  preprocess_presets?: Record<string, PreprocessPreset>;
  /** Per kind and source type: schema, version, whether it needs calibration data, whether it is implemented. */
  preprocess_sources?: Record<string, Record<string, PreprocessSource>>;
  fragments: Record<string, string>;
  metrics: Record<string, MetricDef>;
  categories: Record<string, { name: string; label: string; order: number; title?: string | null }>;
  averages: { label: string; metric: string; categories: string[] }[];
  derived_metrics: string[];
}

/** A method's display name: its manifest title, else the identifier. */
export function methodTitle(describe: Describe, name: string): string {
  return describe.methods[name]?.title || name;
}

/** A record's display name: the server's label ("QuaRot + GPTQ"), else the method's title. */
export function rowMethodTitle(describe: Describe, row: { method: string; label?: string }): string {
  return row.label || methodTitle(describe, row.method);
}

export interface Deviation {
  path: string;
  preset: unknown;
  actual: unknown;
}

export interface Row {
  record_id: string;
  field: string;
  created_at: string;
  owner: string | null;
  visibility: "private" | "public" | "retracted";
  model: string;
  dtype: string;
  method: string;
  /** "<preprocess title>[*] + <method title>", e.g. "QuaRot + GPTQ"; the method title without preprocessing. */
  label?: string;
  preprocess?: { preset: string | null; steps: PreprocessStep[]; versions: Record<string, string> } | null;
  preset: string | null;
  /** Speculative decoding config columns: the draft model code, its tokens per step and verification, the provider. */
  draft?: string | null;
  decoding?: { k: number; verification: "greedy" | "sampling" } | null;
  runtime?: string;
  task: string;
  mode: string;
  evaluator: string;
  seed: number;
  gpu: string | null;
  role: "main" | "reference";
  metrics: Record<string, string>;
  derived: Record<string, string>;
  comparisons: { reference: string | null; values: Record<string, string> };
  fidelity: { preset: string | null; confirmed: number; assumed: string[]; deviations: Deviation[]; repro: string };
  display: Record<string, unknown>;
  annotation: { tags: string[]; note: string; status: string; updated_by: string } | null;
  submitted_via: string;
  run_key: string;
  state: "active" | "redo" | "trashed";
  problems: Problem[];
  projects: string[];
  replaces?: string[];
  trashed?: { changed_by: string; changed_at: string; superseded_by: string | null };
  record: Record<string, unknown>;
}

export interface SampleDiff {
  record_id: string;
  against: string;
  coverage: { main: number; reference: number; aligned: number };
  counts: { to_wrong: number; to_right: number; both_right: number; both_wrong: number };
  subtasks: Record<string, { to_wrong: number; to_right: number; both_right: number; both_wrong: number }>;
  total: number;
  samples: { id: string; subtask: string; flip: string; target: unknown; main: unknown; reference: unknown }[];
}

export interface Snapshot {
  snapshot_id: string;
  table_id: string;
  created_by: string;
  created_at: string;
  title: string | null;
  body?: TableView & { metrics: Record<string, MetricDef>; averages: Describe["averages"]; categories: Describe["categories"] };
}

export type Problem = "redo" | "trashed" | "retracted" | "reference_trashed";

export interface Project {
  name: string;
  description: string;
  created_by: string;
  created_at: string;
  members: string[];
  records: number;
}

export interface Dependents {
  comparisons: { record_id: string; owner: string | null; model: string }[];
  projects: string[];
}

export type Scope = "mine" | "pool" | `project:${string}`;

export type CellStatus = "done" | "reusable" | "running" | "failed" | "pending" | "invalid";

export interface Cell {
  run_key: string;
  role: "main" | "reference";
  status: CellStatus;
  record_id?: string;
  metrics?: Record<string, string>;
  owner?: string | null;
  visibility?: "private" | "public" | "retracted";
  at?: string;
  /** Sample limit stored in the cell's record (null: full set); absent while the cell has no result. */
  limit?: number | null;
  activity?: { status: string; owner: string | null; at: string; message: string };
  claim?: { by: string; at: string };
  /** Why an "invalid" cell cannot run; such cells carry no run_key. */
  issues?: Issue[];
  /** Per-repeat cells of a repeated run; `metrics` is then the mean over those with a result. */
  repeats?: Cell[];
  /** Sample std (full-precision decimal, null for one result) and number of results per metric of a repeated run. */
  spread?: Record<string, { std: string | null; n: number }>;
}

/** One planned job and the stored result covering it, if any (POST /api/reuse-preview). */
export interface ReusePreviewJob {
  role: "main" | "reference";
  run_key: string;
  model: string;
  method: string;
  label?: string;
  task: string;
  seed: number;
  repeat: number;
  status: "reused" | "run" | "not_reused";
  reason: string | null;
  record: { record_id: string; owner: string | null; created_at: string; gpu: string | null; metrics: Record<string, string> } | null;
}

export interface TableDefinition {
  name: string;
  field: string;
  models: string[];
  tasks: (string | TaskRef)[];
  rows: { label: string; method: { name: string; preset?: string | null; config?: Record<string, unknown> }; preprocess?: PreprocessSpec | string | null; decoding?: Record<string, unknown> | null }[];
  shared: Record<string, unknown>;
}

export interface TableView {
  table_id: string;
  created_by: string;
  updated_at: string;
  definition: TableDefinition;
  tasks: string[];
  blocks: { model: string; rows: { key: string; label: string; title?: string; method: TableDefinition["rows"][number]["method"]; draft?: string | null; decoding?: { k: number; verification: string } | null; runtime?: string; baseline: boolean; issues: Issue[]; cells: Record<string, Cell> }[] }[];
  counts: Partial<Record<CellStatus, number>>;
  /** Frozen per-column definitions; only snapshots carry them. */
  columns?: { task: string; metric: string; category: string; band: string; order: number }[];
}

export interface Command {
  write: string;
  run: string;
  push: string;
}

export interface RunItem {
  plan_id: string;
  created_at: string;
  title: string | null;
  summary: string;
  plan: { plan_id: string; jobs: { job_id: string; role: string; spec: any; fidelity: any }[]; reused: { run_key: string; role: string; record_id: string; reason: string }[] };
  heartbeats: Record<string, { status: string; message: string; at: string }>;
  records: Record<string, string>;
}

export interface Rendered {
  plan: {
    plan_id: string;
    jobs: { job_id: string; role: string; spec: any; fidelity: any }[];
    reused: { run_key: string; role: string; record_id: string; reason: string }[];
  } | null;
  issues: Issue[];
  estimate: {
    hardware: string | null;
    jobs: { job_id: string; role: string; task: string; prepare_s: number | null; measure_s: number | null; basis: string }[];
    total_s: number | null;
    notes: string[];
  } | null;
  commands: Command[];
}

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function call<T>(method: string, path: string, body?: unknown, text = false): Promise<T> {
  const response = await fetch(path, {
    method,
    credentials: "same-origin",
    headers: body === undefined ? {} : { "Content-Type": text ? "text/plain" : "application/json" },
    body: body === undefined ? undefined : text ? (body as string) : JSON.stringify(body),
  });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const payload = await response.json();
      detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail);
    } catch {
      // keep statusText
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}

async function upload<T>(path: string, file: Blob, name: string): Promise<T> {
  const form = new FormData();
  form.append("file", file, name);
  const response = await fetch(path, { method: "POST", credentials: "same-origin", body: form });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      // keep statusText
    }
    throw new ApiError(response.status, typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return (await response.json()) as T;
}

type ImportResult = { imported: string[]; duplicates: string[]; comparisons_updated: string[]; redone: string[]; evidence_attached: string[]; unknown_metrics: string[] };

export const api = {
  me: () => call<Me>("GET", "/api/me"),
  login: (username: string, password: string) => call<Me>("POST", "/api/auth/login", { username, password }),
  logout: () => call("POST", "/api/auth/logout"),
  describe: () => call<Describe>("GET", "/api/describe"),
  validate: (spec: unknown) => call<{ issues: Issue[] }>("POST", "/api/validate", { spec }),
  reusePreview: (spec: unknown) => call<{ issues: Issue[]; jobs: ReusePreviewJob[] }>("POST", "/api/reuse-preview", { spec }),
  render: (spec: unknown, shards: number, to: string) => call<Rendered>("POST", "/api/render", { spec, shards, to }),
  records: (scope: Scope, filters: Record<string, string> = {}) =>
    call<Row[]>("GET", `/api/records?${new URLSearchParams({ scope, ...filters })}`),
  record: (id: string) => call<Row>("GET", `/api/records/${id}`),
  publish: (id: string) => call("POST", `/api/records/${id}/publish`),
  retract: (id: string) => call("POST", `/api/records/${id}/retract`),
  annotate: (id: string, body: { tags: string[]; note: string; status: string }) =>
    call("PATCH", `/api/records/${id}/annotation`, body),
  previewBundle: (text: string) => call<{ rows: Row[] }>("POST", "/api/bundles/preview", text, true),
  importBundle: (text: string, to: string) =>
    call<ImportResult>("POST", `/api/bundles?${new URLSearchParams({ via: "paste", to })}`, text, true),
  previewPackage: (file: Blob) => upload<{ rows: Row[]; with_samples: number }>("/api/packages/preview", file, "package.tar.gz"),
  importPackage: (file: Blob, to: string) => upload<ImportResult>(`/api/packages?${new URLSearchParams({ to })}`, file, "package.tar.gz"),
  attachEvidence: (id: string, file: Blob) => upload<{ comparisons_updated: string[] }>(`/api/records/${id}/evidence`, file, "per_sample.jsonl.gz"),
  sampleDiff: (id: string, q: { flip: string; subtask?: string; offset: number; limit: number }) =>
    call<SampleDiff>("GET", `/api/records/${id}/samples?${new URLSearchParams(Object.entries(q).filter(([, v]) => v !== undefined).map(([k, v]) => [k, String(v)]))}`),
  dependents: (id: string) => call<Dependents>("GET", `/api/records/${id}/dependents`),
  trash: (id: string) => call("DELETE", `/api/records/${id}`),
  restore: (id: string) => call("POST", `/api/records/${id}/restore`),
  purge: (id: string) => call("DELETE", `/api/trash/${id}`),
  trashList: () => call<Row[]>("GET", "/api/trash"),
  redo: (id: string, to: string) => call<Rendered>("POST", `/api/records/${id}/redo`, { to }),
  cancelRedo: (id: string) => call("DELETE", `/api/records/${id}/redo`),
  projects: () => call<Project[]>("GET", "/api/projects"),
  createProject: (name: string, description: string) => call<Project>("POST", "/api/projects", { name, description }),
  updateProject: (name: string, changes: { name?: string; description?: string }) =>
    call<Project>("PATCH", `/api/projects/${name}`, changes),
  deleteProject: (name: string) => call<{ ok: boolean }>("DELETE", `/api/projects/${name}`),
  project: (name: string) => call<Project>("GET", `/api/projects/${name}`),
  addMember: (name: string, username: string) => call("POST", `/api/projects/${name}/members`, { username }),
  removeMember: (name: string, username: string) => call("DELETE", `/api/projects/${name}/members/${username}`),
  link: (name: string, recordIds: string[]) => call("POST", `/api/projects/${name}/records`, { record_ids: recordIds }),
  tables: (name: string) => call<TableView[]>("GET", `/api/projects/${name}/tables`),
  saveTable: (name: string, id: string | null, definition: TableDefinition) =>
    call<TableView>(id ? "PUT" : "POST", `/api/projects/${name}/tables${id ? `/${id}` : ""}`, definition),
  deleteTable: (name: string, id: string) => call("DELETE", `/api/projects/${name}/tables/${id}`),
  tableCommand: (name: string, id: string, runKeys: string[], force: boolean) =>
    call<Rendered>("POST", `/api/projects/${name}/tables/${id}/command`, { run_keys: runKeys, force }),
  storage: () => call<Record<"shared" | "personal", { path: string; files: number; bytes: number; editable: boolean }>>("GET", "/api/settings/storage"),
  setStorage: (scope: "shared" | "personal", path: string) =>
    call<{ moved: number }>("PUT", scope === "shared" ? "/api/settings/storage/shared" : "/api/me/storage", { path }),
  resumeRun: (planId: string, to: string) => call<Rendered>("POST", `/api/runs/${planId}/resume`, { to }),
  freezeTable: (name: string, id: string) => call<Snapshot>("POST", `/api/projects/${name}/tables/${id}/snapshots`),
  snapshots: (name: string) => call<Snapshot[]>("GET", `/api/projects/${name}/snapshots`),
  snapshot: (name: string, id: string) => call<Snapshot>("GET", `/api/projects/${name}/snapshots/${id}`),
  renameSnapshot: (name: string, id: string, title: string) => call<{ ok: boolean }>("PATCH", `/api/projects/${name}/snapshots/${id}`, { title }),
  deleteSnapshot: (name: string, id: string) => call<{ ok: boolean }>("DELETE", `/api/projects/${name}/snapshots/${id}`),
  claim: (name: string, runKeys: string[]) => call("POST", `/api/projects/${name}/claims`, { run_keys: runKeys }),
  release: (name: string, runKey: string) => call("DELETE", `/api/projects/${name}/claims/${runKey}`),
  unlink: (name: string, id: string) => call("DELETE", `/api/projects/${name}/records/${id}`),
  methods: () => call<{ manifest: Manifest; owner: string; visibility: string }[]>("GET", "/api/methods"),
  models: () => call<{ model: ModelEntry; registered_by: string }[]>("GET", "/api/models"),
  lookupModel: (repo: string, revision: string) =>
    call<ModelEntry>("GET", `/api/models/lookup?${new URLSearchParams(revision ? { repo, revision } : { repo })}`),
  registerModel: (entry: ModelEntry) => call<ModelEntry>("POST", "/api/models", entry),
  publishMethod: (name: string) => call("POST", `/api/methods/${name}/publish`),
  runs: () => call<RunItem[]>("GET", "/api/runs"),
  renameRun: (planId: string, title: string) => call("PATCH", `/api/runs/${planId}`, { title }),
  deleteRun: (planId: string) => call("DELETE", `/api/runs/${planId}`),
  runCommand: (planId: string, to: string) => call<Command>("POST", `/api/runs/${planId}/command`, { to }),
  users: () => call<{ username: string; role: string; disabled: number; created_at: string }[]>(
    "GET", "/api/admin/users"),
  addUser: (username: string, password: string, role: string) =>
    call("POST", "/api/admin/users", { username, password, role }),
  setUserEnabled: (username: string, enabled: boolean) =>
    call("POST", `/api/admin/users/${username}/${enabled ? "enable" : "disable"}`),
  updateUser: (username: string, changes: { username?: string; role?: string }) =>
    call<{ username: string; role: string }>("PATCH", `/api/admin/users/${username}`, changes),
  deleteUser: (username: string) => call("DELETE", `/api/admin/users/${username}`),
  resetPassword: (username: string, password: string) =>
    call("POST", `/api/admin/users/${username}/password`, { password }),
  changePassword: (password: string) => call("PUT", "/api/me/password", { password }),
  setLocale: (locale: "en" | "zh") => call<{ locale: "en" | "zh" }>("PUT", "/api/me/locale", { locale }),
  adminTokens: () => call<{ token_id: string; username: string; label: string; can_publish: number; created_at: string; expires_at: string | null; last_used: string | null }[]>(
    "GET", "/api/admin/tokens"),
  revokeAdminToken: (id: string) => call("DELETE", `/api/admin/tokens/${id}`),
};
