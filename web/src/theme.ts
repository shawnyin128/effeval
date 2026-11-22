// Appearance: the light or dark palette, or following the system setting. Saved per browser.
export type Theme = "light" | "dark" | "system";

const KEY = "effeval.theme";
const systemDark = window.matchMedia("(prefers-color-scheme: dark)");

export function savedTheme(): Theme {
  const value = localStorage.getItem(KEY);
  return value === "light" || value === "dark" ? value : "system";
}

function paint(theme: Theme): void {
  const dark = theme === "dark" || (theme === "system" && systemDark.matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}

/** Saves `theme` for this browser and repaints the page with it. */
export function applyTheme(theme: Theme): void {
  if (theme === "system") localStorage.removeItem(KEY);
  else localStorage.setItem(KEY, theme);
  paint(theme);
}

/** Repaints whenever the system switches between light and dark, if the saved choice follows it. */
export function followSystemTheme(): void {
  systemDark.addEventListener("change", () => paint(savedTheme()));
}
