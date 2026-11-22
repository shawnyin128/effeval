// Table copying. Markdown goes to text/plain; the same table goes to text/html so Slides, Docs and Sheets paste
// it as a real table. Marks: best values bold, second best underlined; highlighted rows get a tinted background.

export type CellMark = "best" | "second" | null;

export interface CopyRow {
  cells: string[];
  marks?: CellMark[];
  highlight?: boolean;
}

function escapeHtml(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/** Markdown table text; best cells **bold**, second best <u>underlined</u>, pipes escaped. */
export function markdownTable(header: string[], rows: CopyRow[]): string {
  const cell = (v: string, mark: CellMark | undefined) => {
    const text = v.replace(/\|/g, "\\|");
    return mark === "best" ? `**${text}**` : mark === "second" ? `<u>${text}</u>` : text;
  };
  const lines = [header.map((h) => cell(h, null)), header.map(() => "---"), ...rows.map((r) => r.cells.map((v, i) => cell(v, r.marks?.[i])))];
  return lines.map((line) => `| ${line.join(" | ")} |`).join("\n");
}

function htmlTable(header: string[], rows: CopyRow[], footnote?: string | null): string {
  const td = (v: string, mark: CellMark | undefined) => {
    const text = escapeHtml(v);
    return `<td>${mark === "best" ? `<b>${text}</b>` : mark === "second" ? `<u>${text}</u>` : text}</td>`;
  };
  const head = `<tr>${header.map((h) => `<th>${escapeHtml(h)}</th>`).join("")}</tr>`;
  const body = rows.map((r) => `<tr${r.highlight ? ' style="background:#F3E8F7"' : ""}>${r.cells.map((v, i) => td(v, r.marks?.[i])).join("")}</tr>`).join("");
  return `<table><thead>${head}</thead><tbody>${body}</tbody></table>${footnote ? `<p>${escapeHtml(footnote)}</p>` : ""}`;
}

/** Put a table on the clipboard as Markdown text plus an HTML table with the same marks. */
export async function copyTable(header: string[], rows: CopyRow[], footnote?: string | null): Promise<void> {
  await navigator.clipboard.write([
    new ClipboardItem({
      "text/plain": new Blob([markdownTable(header, rows) + (footnote ? `\n\n${footnote}` : "")], { type: "text/plain" }),
      "text/html": new Blob([htmlTable(header, rows, footnote)], { type: "text/html" }),
    }),
  ]);
}

/** Put plain text (e.g. LaTeX source) on the clipboard. */
export async function copyText(text: string): Promise<void> {
  await navigator.clipboard.writeText(text);
}
