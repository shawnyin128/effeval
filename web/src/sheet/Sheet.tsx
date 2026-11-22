import { ArrowDown, ArrowUp, ChevronsUpDown } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from "react";

/** A read-only spreadsheet: lettered columns, numbered rows, frozen headers, sorting and resizing. */
export interface SheetColumn<T> {
  key: string;
  header: string;
  group: string;
  width: number;
  align?: "left" | "right" | "center";
  numeric?: boolean;
  mono?: boolean;
  /** Full-precision text used for sorting, tooltips and copying. */
  value: (row: T) => string | number | null | undefined;
  render?: (row: T) => ReactNode;
  tone?: (row: T) => "pos" | "neg" | undefined;
}

/** A labelled, tinted column region; columns join a zone through their `group`. */
export interface SheetZone {
  label: string;
  tone: "model" | "config" | "result" | "provenance";
}

export interface SheetTab {
  key: string;
  label: string;
  count: number;
}

const LETTER_HEIGHT = 22;
const BAND_HEIGHT = 24;
const OVERSCAN = 8;
const ROWNUM_WIDTH = 46;

function letter(index: number): string {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

function compare(a: string | number | null | undefined, b: string | number | null | undefined, numeric: boolean): number {
  if (a === null || a === undefined || a === "") return b === null || b === undefined || b === "" ? 0 : 1;
  if (b === null || b === undefined || b === "") return -1;
  if (numeric) return Number(a) - Number(b);
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

export function Sheet<T>({
  columns,
  rows,
  rowKey,
  frozen = 1,
  rowHeight = 28,
  selectedKey,
  onSelect,
  tabs,
  activeTab,
  onTab,
  status,
  onOrderChange,
  zones,
  rowClass,
  rowMark,
}: {
  columns: SheetColumn<T>[];
  rows: T[];
  rowKey: (row: T) => string;
  /** Extra class names for a data row, e.g. to grey out stale results. */
  rowClass?: (row: T) => string | undefined;
  /** Shown in place of the row number when it returns something. */
  rowMark?: (row: T) => ReactNode;
  frozen?: number;
  rowHeight?: number;
  selectedKey?: string | null;
  onSelect?: (row: T) => void;
  tabs?: SheetTab[];
  activeTab?: string;
  onTab?: (key: string) => void;
  status?: ReactNode;
  /** Receives the rows in displayed (sorted) order whenever it changes, e.g. for copying. */
  onOrderChange?: (rows: T[]) => void;
  zones?: Record<string, SheetZone>;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [viewport, setViewport] = useState({ top: 0, height: 600 });
  const [widths, setWidths] = useState<Record<string, number>>({});
  const [sort, setSort] = useState<{ key: string; dir: "asc" | "desc" } | null>(null);
  const [active, setActive] = useState<{ row: number; col: number } | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);

  const widthOf = useCallback((c: SheetColumn<T>) => widths[c.key] ?? c.width, [widths]);
  const offsets = useMemo(() => {
    const out: number[] = [];
    let x = ROWNUM_WIDTH;
    for (const c of columns) {
      out.push(x);
      x += widthOf(c);
    }
    return { left: out, total: x };
  }, [columns, widthOf]);

  const sorted = useMemo(() => {
    if (!sort) return rows;
    const column = columns.find((c) => c.key === sort.key);
    if (!column) return rows;
    const out = [...rows].sort((a, b) => compare(column.value(a), column.value(b), !!column.numeric));
    return sort.dir === "asc" ? out : out.reverse();
  }, [rows, columns, sort]);

  useEffect(() => {
    onOrderChange?.(sorted);
  }, [sorted, onOrderChange]);

  useEffect(() => {
    const el = scroller.current;
    if (!el) return;
    const measure = () => setViewport({ top: el.scrollTop, height: el.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const bandHeight = zones ? BAND_HEIGHT : 0;
  const headerHeight = LETTER_HEIGHT + bandHeight + rowHeight + 2;
  const zoneClass = (i: number) => {
    const zone = zones?.[columns[i].group];
    if (!zone) return "";
    const last = i === columns.length - 1 || columns[i + 1].group !== columns[i].group;
    return `zone-${zone.tone}${last && i !== columns.length - 1 ? " zone-end" : ""}`;
  };
  const bands = useMemo(() => {
    if (!zones) return [];
    const out: { group: string; start: number; end: number }[] = [];
    columns.forEach((c, i) => {
      const prev = out[out.length - 1];
      if (prev && prev.group === c.group && i !== frozen) prev.end = i;
      else out.push({ group: c.group, start: i, end: i });
    });
    return out;
  }, [columns, zones, frozen]);
  const fillerCount = Math.max(0, Math.ceil((viewport.height - headerHeight) / rowHeight) - sorted.length);
  const totalRows = sorted.length + fillerCount;
  const first = Math.max(0, Math.floor(viewport.top / rowHeight) - OVERSCAN);
  const last = Math.min(totalRows, Math.ceil((viewport.top + viewport.height) / rowHeight) + OVERSCAN);

  const startResize = (key: string, startX: number, startWidth: number) => {
    setDragging(key);
    const move = (e: MouseEvent) => setWidths((w) => ({ ...w, [key]: Math.max(48, startWidth + e.clientX - startX) }));
    const up = () => {
      setDragging(null);
      window.removeEventListener("mousemove", move);
      window.removeEventListener("mouseup", up);
    };
    window.addEventListener("mousemove", move);
    window.addEventListener("mouseup", up);
  };

  const toggleSort = (key: string) =>
    setSort((s) => (s?.key !== key ? { key, dir: "asc" } : s.dir === "asc" ? { key, dir: "desc" } : null));

  const activate = (row: number, col: number) => {
    setActive({ row, col });
    const target = sorted[row];
    if (target !== undefined) onSelect?.(target);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (!active) return;
    const moves: Record<string, [number, number]> = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
    const move = moves[e.key];
    if (!move) return;
    e.preventDefault();
    const row = Math.min(Math.max(active.row + move[0], 0), sorted.length - 1);
    const col = Math.min(Math.max(active.col + move[1], 0), columns.length - 1);
    activate(row, col);
    const el = scroller.current;
    if (el) {
      const top = row * rowHeight;
      if (top < el.scrollTop) el.scrollTop = top;
      if (top + rowHeight > el.scrollTop + el.clientHeight - headerHeight) el.scrollTop = top + rowHeight - el.clientHeight + headerHeight;
    }
  };

  const frozenStyle = (index: number) =>
    index < frozen ? { left: offsets.left[index], position: "sticky" as const } : undefined;

  return (
    <div className="sheet">
      <div
        className="sheet-scroll"
        ref={scroller}
        tabIndex={0}
        onKeyDown={onKeyDown}
        onScroll={(e) => setViewport({ top: e.currentTarget.scrollTop, height: e.currentTarget.clientHeight })}
      >
        <div className="sheet-canvas" style={{ width: offsets.total, height: headerHeight + totalRows * rowHeight }}>
          <div className="sheet-head" style={{ width: offsets.total }}>
            <div className="sheet-row" style={{ height: LETTER_HEIGHT }}>
              <div className="sheet-cell sheet-corner" style={{ width: ROWNUM_WIDTH }} />
              {columns.map((c, i) => (
                <div key={c.key} className={`sheet-cell sheet-letter ${active?.col === i ? "active" : ""} ${i === frozen - 1 ? "frozen-edge" : ""} ${i < frozen ? "frozen" : ""}`}
                  style={{ width: widthOf(c), ...frozenStyle(i) }}>
                  {letter(i)}
                </div>
              ))}
            </div>
            {zones && (
              <div className="sheet-row" style={{ height: bandHeight }}>
                <div className="sheet-cell sheet-corner" style={{ width: ROWNUM_WIDTH }} />
                {bands.map((b) => {
                  const zone = zones[b.group];
                  const width = columns.slice(b.start, b.end + 1).reduce((sum, c) => sum + widthOf(c), 0);
                  const continues = bands.some((o) => o.group === b.group && o.start > b.end);
                  const sticky = b.start < frozen ? { left: offsets.left[b.start], position: "sticky" as const } : undefined;
                  return (
                    <div key={`${b.group}-${b.start}`}
                      className={`sheet-cell sheet-band zone-${zone?.tone ?? "provenance"} ${b.start < frozen ? "frozen" : ""} ${b.end === frozen - 1 ? "frozen-edge" : ""} ${b.end < columns.length - 1 && !continues ? "zone-end" : ""}`}
                      style={{ width, ...sticky }}>
                      {zone?.label ?? b.group}
                      {b.end > b.start && <span className="band-count">{b.end - b.start + 1}</span>}
                    </div>
                  );
                })}
              </div>
            )}
            <div className="sheet-row" style={{ height: rowHeight + 2 }}>
              <div className="sheet-cell sheet-corner" style={{ width: ROWNUM_WIDTH, borderBottom: "2px solid var(--sheet-frozen)" }} />
              {columns.map((c, i) => {
                const sortedHere = sort?.key === c.key;
                return (
                  <div key={c.key}
                    className={`sheet-cell sheet-title ${c.align ?? "left"} ${sortedHere ? "sorted" : ""} ${zoneClass(i)} ${i === frozen - 1 ? "frozen-edge" : ""} ${i < frozen ? "frozen" : ""}`}
                    style={{ width: widthOf(c), ...frozenStyle(i) }} title={`${c.group} · ${c.header}`}
                    onClick={() => toggleSort(c.key)}>
                    <span className="clip">{c.header}</span>
                    <span className="sort">
                      {sortedHere ? (sort.dir === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />) : <ChevronsUpDown size={12} opacity={0.45} />}
                    </span>
                    <span className={`sheet-resize ${dragging === c.key ? "dragging" : ""}`}
                      onClick={(e) => e.stopPropagation()}
                      onMouseDown={(e) => { e.preventDefault(); e.stopPropagation(); startResize(c.key, e.clientX, widthOf(c)); }} />
                  </div>
                );
              })}
            </div>
          </div>
          <div className="sheet-body">
            {Array.from({ length: last - first }, (_, k) => first + k).map((index) => {
              const row = sorted[index];
              const top = headerHeight + index * rowHeight;
              if (row === undefined) {
                return (
                  <div key={`filler-${index}`} className="sheet-row filler" style={{ top, height: rowHeight, width: offsets.total }}>
                    <div className="sheet-cell sheet-rownum" style={{ width: ROWNUM_WIDTH }}>{index + 1}</div>
                    {columns.map((c, i) => (
                      <div key={c.key} className={`sheet-cell ${zoneClass(i)} ${i === frozen - 1 ? "frozen-edge" : ""} ${i < frozen ? "frozen" : ""}`}
                        style={{ width: widthOf(c), ...frozenStyle(i) }} />
                    ))}
                  </div>
                );
              }
              const key = rowKey(row);
              return (
                <div key={key} className={`sheet-row ${selectedKey === key ? "selected" : ""} ${rowClass?.(row) ?? ""}`} style={{ top, height: rowHeight, width: offsets.total }}>
                  <div className={`sheet-cell sheet-rownum ${active?.row === index ? "active" : ""}`} style={{ width: ROWNUM_WIDTH }}>{rowMark?.(row) || index + 1}</div>
                  {columns.map((c, i) => {
                    const raw = c.value(row);
                    const tone = c.tone?.(row);
                    const classes = [
                      "sheet-cell", c.align ?? (c.numeric ? "right" : "left"), c.numeric ? "num" : "", c.mono ? "mono" : "",
                      tone ?? "", zoneClass(i), i < frozen ? "frozen" : "", i === frozen - 1 ? "frozen-edge" : "",
                      active?.row === index && active.col === i ? "sheet-active" : "",
                    ].join(" ");
                    return (
                      <div key={c.key} className={classes} style={{ width: widthOf(c), ...frozenStyle(i) }}
                        title={raw === null || raw === undefined ? undefined : String(raw)}
                        onMouseDown={() => activate(index, i)}>
                        {c.render ? c.render(row) : <span className="clip">{raw ?? ""}</span>}
                      </div>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </div>
      </div>
      {(tabs || status) && (
        <div className="sheet-tabs">
          {tabs?.map((t) => (
            <button key={t.key} className={`sheet-tab ${activeTab === t.key ? "active" : ""}`} onClick={() => onTab?.(t.key)}>
              {t.label}<span className="count">{t.count}</span>
            </button>
          ))}
          {status && <span className="sheet-status">{status}</span>}
        </div>
      )}
    </div>
  );
}
