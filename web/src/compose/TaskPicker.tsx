import { ChevronDown, ChevronRight, Search, X } from "lucide-react";
import { Fragment, useMemo, useState, type ReactElement } from "react";
import type { TaskDef, TaskRef } from "../api";
import { useApp } from "../components/common";
import { useT } from "../i18n";
import { Button, Checkbox, Input, Segmented } from "../ui/controls";
import { Badge } from "../ui/display";
import { MODE_LABEL, PPL_CATEGORY, SOURCES, buildIndex, common, inMode, metricLabel, search, sourceOf, splitName, type Source } from "./taskIndex";

const PAGE = 100;
const SEARCH_LIMIT = 200;
const SOURCE_LABEL: Record<Source, string> = { lm_eval: "lm-eval", evalscope: "evalscope", specdecode: "SpecDecode", custom: "Custom" };
type Filter = "all" | Source;
/** What the evaluation measures: perplexity on a text dataset, KL against the reference model, or downstream task scores. */
type Measure = "all" | "ppl" | "kl" | "downstream";
type ModeFilter = "all" | "loglikelihood" | "generation";

/** One-line description: markdown heading marks and line breaks removed. */
const blurb = (task: TaskDef) => task.description.replace(/^#+\s*/, "").replace(/\s+/g, " ").trim();

/**
 * Task selection for a catalog of thousands, narrowed top-down: what to measure (perplexity, KL, downstream), how
 * downstream tasks are scored (likelihood or generation), then the kind of task, with the common tasks first.
 * Search is a shortcut at every level. Selected tasks show as removable chips. `readOnly` browses without checkboxes.
 */
export function TaskPicker({ allowedModes, selected, presetTasks = [], invalid = new Set<string>(), onChange, readOnly = false, onOpen, onRestore, keep, whyExcluded }: {
  allowedModes: string[];
  selected: TaskRef[];
  presetTasks?: string[];
  invalid?: Set<string>;
  onChange?: (tasks: TaskRef[]) => void;
  readOnly?: boolean;
  /** Called when a row's name is clicked (details). */
  onOpen?: (task: TaskDef) => void;
  /** Puts the preset's tasks back; offered while the selection differs from them. */
  onRestore?: () => void;
  /** Tasks that fail it are left out of the list; undefined lists every task. */
  keep?: (task: TaskDef) => boolean;
  /** Why a selected task cannot run with the current settings, or null; such tasks stay selected and show as invalid. */
  whyExcluded?: (task: TaskDef) => string | null;
}) {
  const { describe } = useApp();
  const t = useT();
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");
  const [measureChoice, setMeasure] = useState<Measure>("all");
  const [modeFilter, setModeFilter] = useState<ModeFilter>("all");
  const [category, setCategory] = useState("all");
  const [open, setOpen] = useState<Set<string>>(new Set());
  const [shown, setShown] = useState(PAGE);
  const index = useMemo(() => buildIndex(describe, allowedModes, keep), [describe, allowedModes, keep]);
  const source = filter === "all" ? undefined : filter;
  const titles = useMemo(() => Object.fromEntries(Object.values(describe.categories).map((c) => [c.name, c.title ?? c.label])), [describe]);
  const categoryRank = useMemo(() => new Map(Object.values(describe.categories).map((c, i) => [c.name, c.order * 100 + i])), [describe]);
  const rank = (name: string) => categoryRank.get(name) ?? 9999;
  const pinned = useMemo(() => {
    const tasks = common(describe, allowedModes, keep);
    const listed = new Set(tasks.map((x) => x.name));
    return [...tasks, ...index.roots.specdecode.filter((x) => !listed.has(x.name))];
  }, [describe, allowedModes, keep, index]);
  const pinnedNames = useMemo(() => new Set(pinned.map((x) => x.name)), [pinned]);
  const found = useMemo(() => (query.trim() ? search(index, query, source, SEARCH_LIMIT, titles, pinnedNames) : null), [index, query, source, titles, pinnedNames]);
  const specOnly = filter === "specdecode" || (index.counts.specdecode > 0 && index.counts.lm_eval + index.counts.evalscope + index.counts.custom === 0);
  const measure: Measure = specOnly ? "all" : measureChoice;
  const names = new Set(selected.map((x) => x.name));
  const wantedMode = measure === "kl" ? "loglikelihood" : measure === "downstream" ? modeFilter : "all";
  const toggle = (task: TaskDef, on: boolean) => {
    const chosen: TaskRef = wantedMode !== "all" && task.modes.includes(wantedMode) && wantedMode !== task.default_mode ? { name: task.name, mode: wantedMode } : { name: task.name };
    onChange?.(on ? [...selected, chosen] : selected.filter((x) => x.name !== task.name));
  };
  const allRoots = useMemo(() => SOURCES.flatMap((s) => (source && source !== s ? [] : index.roots[s])), [index, source]);
  const inMeasure = (task: TaskDef) => (measure === "ppl" ? task.category === PPL_CATEGORY : measure === "all" || task.category !== PPL_CATEGORY);
  const narrowed = useMemo(() => allRoots.filter((x) => inMeasure(x) && inMode(x, wantedMode)), [allRoots, measure, wantedMode]);
  const categories = useMemo(() => {
    const counts = new Map<string, number>();
    for (const x of narrowed) counts.set(x.category, (counts.get(x.category) ?? 0) + 1);
    return [...counts].sort((a, b) => rank(a[0]) - rank(b[0]));
  }, [narrowed, categoryRank]);
  const activeCategory = measure === "ppl" || specOnly ? "all" : category;
  const listed = useMemo(() => narrowed.filter((x) => activeCategory === "all" || x.category === activeCategory), [narrowed, activeCategory]);
  const browsing = specOnly || measure !== "all" || category !== "all";
  const roots = found
    ? found.roots.map((n) => index.byName.get(n)!)
    : browsing ? [...listed].sort((a, b) => Number(pinnedNames.has(b.name)) - Number(pinnedNames.has(a.name))) : [];
  const commonGroups = useMemo(() => {
    const groups = new Map<string, TaskDef[]>();
    for (const x of pinned) if (!source || sourceOf(x) === source) groups.set(x.category, [...(groups.get(x.category) ?? []), x]);
    const speed = (tasks: TaskDef[]) => Number(!tasks.some((x) => sourceOf(x) === "specdecode"));
    return [...groups].sort((a, b) => speed(a[1]) - speed(b[1]) || rank(a[0]) - rank(b[0]));
  }, [pinned, source, categoryRank]);
  const reset = () => setShown(PAGE);
  const modeCounts = useMemo(() => {
    const base = allRoots.filter((x) => inMeasure(x) && (category === "all" || x.category === category));
    return { loglikelihood: base.filter((x) => x.modes.includes("loglikelihood")).length, generation: base.filter((x) => x.modes.includes("generation")).length };
  }, [allRoots, measure, category]);
  const differsFromPreset = presetTasks.length > 0 && (selected.length !== presetTasks.length || selected.some((x) => !presetTasks.includes(x.name)));

  const modeBadge = (task: TaskDef) => (
    <Badge tone={task.default_mode === "generation" ? "orange" : undefined} title={t("How the task is scored")}>{MODE_LABEL[task.default_mode] ?? task.default_mode}</Badge>
  );
  const chips = (task: TaskDef, members: number) => (
    <span className="tp-chips">
      {members > 0 && <Badge tone="accent">{t("group · {n} subtasks", { n: members })}</Badge>}
      {modeBadge(task)}
      {task.metrics[0] && <span className="tp-metric"><Badge mono title={task.metrics[0].name}>{metricLabel(task.metrics[0].name)}</Badge></span>}
      {task.num_fewshot !== null && <Badge mono title={t("Default few-shot")}>{t("{n}-shot", { n: task.num_fewshot })}</Badge>}
      {presetTasks.includes(task.name) && <Badge tone="green">{t("paper")}</Badge>}
    </span>
  );

  const row = (task: TaskDef, depth: number): ReactElement => {
    const members = index.children.get(task.name) ?? [];
    const matchOnly = !!found && found.expanded.has(task.name);
    const visibleMembers = matchOnly ? members.filter((m) => found.visible.has(m.name)) : members;
    const expanded = open.has(task.name) || matchOnly;
    const on = names.has(task.name);
    const activate = () => (readOnly ? onOpen?.(task) : toggle(task, !on));
    const [evaluator, target] = splitName(task);
    return (
      <div key={task.name}>
        <div className={`tp-row ${on ? "on" : ""} ${invalid.has(task.name) ? "invalid" : ""} ${readOnly && !onOpen ? "" : "clickable"}`}
          style={{ paddingLeft: 8 + depth * 20 }} onClick={activate}>
          {members.length > 0 ? (
            <button className="icon-button" aria-label={expanded ? t("Collapse") : t("Expand")}
              onClick={(e) => { e.stopPropagation(); setOpen((s) => { const n = new Set(s); if (n.has(task.name)) n.delete(task.name); else n.add(task.name); return n; }); }}>
              {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            </button>
          ) : <span className="tp-spacer" />}
          {!readOnly && <span className="tp-check" onClick={(e) => e.stopPropagation()}><Checkbox checked={on} label={task.name} onChange={(v) => toggle(task, v)} /></span>}
          <span className="tp-name mono" title={task.name}><span className="faint">{evaluator}/</span>{target}</span>
          <span className="tp-desc small faint" title={blurb(task)}>{blurb(task)}</span>
          {chips(task, members.length)}
          {!readOnly && onOpen && <button className="tp-info small" onClick={(e) => { e.stopPropagation(); onOpen(task); }}>{t("Details")}</button>}
        </div>
        {expanded && visibleMembers.map((m) => row(m, depth + 1))}
      </div>
    );
  };

  const hidden = Math.max(0, roots.length - shown);
  const pick = (value: string) => { setMeasure(value as Measure); setCategory("all"); setModeFilter("all"); reset(); };
  const categoryTitle = (name: string) => t(titles[name] ?? name);

  return (
    <div className="stack">
      <div className="tp-toolbar">
        <div className="search tp-search">
          <Search size={14} />
          <Input placeholder={t("Search name, metric or alias, e.g. ppl, wiki2, code")} value={query} onChange={(e) => { setQuery(e.target.value); reset(); }} />
        </div>
        <Segmented value={filter} onChange={(v) => { setFilter(v as Filter); reset(); }} options={[
          { value: "all", label: t("All") },
          ...SOURCES.map((s) => ({ value: s, label: <>{t(SOURCE_LABEL[s])} <span className="faint">{index.counts[s]}</span></>, disabled: index.counts[s] === 0 })),
        ]} />
      </div>
      {!readOnly && (selected.length > 0 || differsFromPreset) && (
        <div className="tp-selected">
          <div className="tp-chiplist">
            {selected.map((x) => {
              const why = whyExcluded?.(describe.tasks[x.name]) ?? null;
              return (
              <span key={x.name} className={`chip ${invalid.has(x.name) || why ? "invalid" : ""}`} title={why ?? undefined}>
                <span className="mono">{x.name}</span>
                {x.mode && <span className="faint">{MODE_LABEL[x.mode] ?? x.mode}</span>}
                <button className="icon-button" aria-label={t("Remove {site}", { site: x.name })} onClick={() => onChange?.(selected.filter((y) => y.name !== x.name))}><X size={12} /></button>
              </span>
              );
            })}
            {selected.length === 0 && <span className="faint small">{t("No tasks selected")}</span>}
          </div>
          {differsFromPreset && onRestore && <Button variant="ghost" size="sm" onClick={onRestore}>{t("Restore preset tasks")}</Button>}
          {selected.length > 0 && <Button variant="ghost" size="sm" onClick={() => onChange?.([])}>{t("Clear")}</Button>}
        </div>
      )}
      <div className="tp-steps">
        {!specOnly && <div className="tp-step">
          <span className="tp-step-label small faint">{t("Measure")}</span>
          <Segmented value={measure} onChange={pick} options={[
            { value: "all", label: t("Common") },
            { value: "ppl", label: <>{t("Perplexity")} <span className="faint">PPL</span></> },
            { value: "kl", label: t("KL divergence") },
            { value: "downstream", label: t("Downstream tasks") },
          ]} />
        </div>}
        {measure === "kl" && <p className="tp-note small faint">{t("KL is computed against the reference model from the answer-choice distributions of Likelihood tasks. KL over a text dataset is not supported yet.")}</p>}
        {measure === "downstream" && (
          <div className="tp-step">
            <span className="tp-step-label small faint">{t("Scoring")}</span>
            <Segmented value={modeFilter} onChange={(v) => { setModeFilter(v as ModeFilter); reset(); }} options={[
              { value: "all", label: t("All") },
              { value: "loglikelihood", label: <>Likelihood <span className="faint">{modeCounts.loglikelihood}</span></>, title: t("Scores the answer choices by their likelihood; no text is generated"), disabled: modeCounts.loglikelihood === 0 },
              { value: "generation", label: <>Generation <span className="faint">{modeCounts.generation}</span></>, title: t("The model generates an answer that is then scored"), disabled: modeCounts.generation === 0 },
            ]} />
          </div>
        )}
        {measure !== "ppl" && categories.length > (specOnly ? 1 : 0) && (
          <div className="tp-step">
            <span className="tp-step-label small faint">{t("Kind")}</span>
            <div className="tp-cats">
              <button className={`tp-cat ${category === "all" ? "on" : ""}`} onClick={() => { setCategory("all"); reset(); }}>{t("All")}</button>
              {categories.map(([name, n]) => (
                <button key={name} className={`tp-cat ${category === name ? "on" : ""}`} onClick={() => { setCategory(name); reset(); }}>{categoryTitle(name)} <span className="faint">{n}</span></button>
              ))}
            </div>
          </div>
        )}
      </div>
      <div className="tp-list">
        {found ? (
          <div className="tp-section">{found.matches.length >= SEARCH_LIMIT ? t("First {n} matches; refine the search", { n: SEARCH_LIMIT }) : t("{n} matches", { n: found.matches.length })}</div>
        ) : browsing ? (
          <div className="tp-section">{measure === "ppl" ? t("Datasets") : activeCategory === "all" ? t("All tasks") : categoryTitle(activeCategory)}<span className="faint">{t("{n} top-level, subtasks inside groups", { n: roots.length })}</span></div>
        ) : (
          <>
            {commonGroups.map(([name, tasks]) => (
              <Fragment key={name}>
                <div className="tp-section">{t("Common")} · {categoryTitle(name)}</div>
                {tasks.map((task) => row(task, 0))}
              </Fragment>
            ))}
            <p className="faint small tp-empty">{t("Pick what to measure or a kind above to browse all {n} tasks, or search.", { n: source ? index.counts[source] : SOURCES.reduce((n, s) => n + index.counts[s], 0) })}</p>
          </>
        )}
        {roots.slice(0, shown).map((task) => row(task, 0))}
        {hidden > 0 && <Button variant="ghost" size="sm" className="tp-more" onClick={() => setShown(shown + PAGE * 2)}>{t("Show {n} more", { n: Math.min(hidden, PAGE * 2) })} ({hidden})</Button>}
        {(found || browsing) && roots.length === 0 && <p className="faint small tp-empty">{t("No tasks match.")}</p>}
      </div>
    </div>
  );
}
