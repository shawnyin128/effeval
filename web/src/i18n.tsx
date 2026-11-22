// UI strings in English (the keys) and Chinese. Technical terms stay in English in both languages.
import { createContext, useCallback, useContext, type ReactNode } from "react";
import { ZH } from "./i18n.zh";

export type Locale = "en" | "zh";

type Translate = (text: string, vars?: Record<string, string | number>) => string;

const I18nContext = createContext<{ locale: Locale; t: Translate }>({ locale: "en", t: (s) => s });

function fill(text: string, vars?: Record<string, string | number>): string {
  if (!vars) return text;
  return text.replace(/\{(\w+)\}/g, (_, k) => String(vars[k] ?? `{${k}}`));
}

export function I18nProvider({ locale, children }: { locale: Locale; children: ReactNode }) {
  const t = useCallback<Translate>((text, vars) => fill(locale === "zh" ? ZH[text] ?? text : text, vars), [locale]);
  return <I18nContext.Provider value={{ locale, t }}>{children}</I18nContext.Provider>;
}

export function useT(): Translate {
  return useContext(I18nContext).t;
}

export function useLocale(): Locale {
  return useContext(I18nContext).locale;
}
