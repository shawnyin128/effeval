import hljs from "highlight.js/lib/core";
import bash from "highlight.js/lib/languages/bash";
import ini from "highlight.js/lib/languages/ini";
import json from "highlight.js/lib/languages/json";
import python from "highlight.js/lib/languages/python";
import yaml from "highlight.js/lib/languages/yaml";

hljs.registerLanguage("bash", bash);
hljs.registerLanguage("toml", ini);
hljs.registerLanguage("json", json);
hljs.registerLanguage("python", python);
hljs.registerLanguage("yaml", yaml);

export type Language = "bash" | "toml" | "json" | "python" | "yaml";

const HEREDOC = /^(.*<<-?\s*'?(\w+)'?[^\n]*\n)([\s\S]*?\n)(\2\n?)$/;
const BY_EXTENSION: Record<string, Language> = { yaml: "yaml", yml: "yaml", json: "json", toml: "toml", py: "python" };

function escape(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function render(text: string, language: Language | undefined): string {
  return language ? hljs.highlight(text, { language, ignoreIllegals: true }).value : escape(text);
}

/**
 * Highlighted HTML of `text` in `language`. A bash heredoc's body is highlighted in the language of the file
 * it is written to (by extension), and left plain when the extension is unknown.
 */
export function highlight(text: string, language: Language): string {
  const heredoc = language === "bash" ? HEREDOC.exec(text) : null;
  if (!heredoc) return render(text, language);
  const [, head, , body, end] = heredoc;
  const extension = /\.(\w+)\s*<</.exec(head)?.[1] ?? "";
  return render(head, "bash") + render(body, BY_EXTENSION[extension]) + `<span class="hljs-meta">${escape(end)}</span>`;
}
