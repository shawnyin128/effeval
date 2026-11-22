import { Boxes, Check, ChevronRight, ChevronsUpDown, FileInput, FlaskConical, FolderClosed, Gauge, Languages, ListChecks, LogOut, Plus, Puzzle, Settings as SettingsIcon, Sheet as SheetIcon, Target, Trash2, Users, UsersRound } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import { useCallback, useEffect, useState, type ReactNode } from "react";
import { api, type Describe, type Me, type Project } from "./api";
import { AppContext, initials } from "./components/common";
import { I18nProvider, useT, type Locale } from "./i18n";
import { disconnect, useLive } from "./live";
import { Members, Runs, Settings } from "./pages/Account";
import { Compose } from "./pages/Compose";
import { Import } from "./pages/Import";
import { Methods } from "./pages/Methods";
import { Benchmarks } from "./pages/Benchmarks";
import { Models } from "./pages/Models";
import { NewProjectDialog } from "./pages/Projects";
import { SnapshotView } from "./project/SnapshotView";
import { Results } from "./pages/Results";
import { Trash } from "./pages/Trash";
import { Button, Field, Input } from "./ui/controls";
import { IssueList } from "./ui/display";
import { UiProvider } from "./ui/overlay";

const LANGUAGES: Record<Locale, string> = { en: "English", zh: "中文" };

const ROUTES = ["compose", "runs", "mine", "pool", "trash", "import", "models", "methods", "benchmarks", "settings", "members"] as const;
type Route = (typeof ROUTES)[number] | `project/${string}`;

function currentRoute(): Route {
  const hash = decodeURIComponent(window.location.hash.replace(/^#\/?/, "")) as Route;
  return (ROUTES as readonly string[]).includes(hash) || hash.startsWith("project/") ? hash : "compose";
}

function Mark({ size = 28 }: { size?: number }) {
  return <span className="brand-mark" style={{ width: size, height: size }}><Gauge size={size * 0.56} strokeWidth={2.2} /></span>;
}

function Login({ onLogin }: { onLogin: (me: Me) => void }) {
  const t = useT();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  return (
    <div className="login">
      <form className="card" onSubmit={async (e) => {
        e.preventDefault();
        setBusy(true);
        try {
          await api.login(username, password);
          onLogin(await api.me());
        } catch (err) {
          setError((err as Error).message);
        } finally {
          setBusy(false);
        }
      }}>
        <div className="gap-3" style={{ display: "flex", flexDirection: "column", alignItems: "center", marginBottom: 24 }}>
          <Mark size={44} />
          <h1 className="strong" style={{ fontSize: 22 }}>effeval</h1>
        </div>
        <div className="stack">
          <Field label={t("Username")}><Input autoFocus value={username} onChange={(e) => setUsername(e.target.value)} autoComplete="username" /></Field>
          <Field label={t("Password")}><Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" /></Field>
          {error && <IssueList issues={[{ code: "auth", severity: "error", paths: [], message: error, hint: "" }]} />}
          <Button variant="primary" size="lg" block type="submit" disabled={busy || !username || !password}>{busy ? t("Signing in…") : t("Sign in")}</Button>
        </div>
      </form>
    </div>
  );
}

export function App() {
  const [me, setMe] = useState<Me | null | undefined>(undefined);
  const [describe, setDescribe] = useState<Describe | null>(null);
  const [route, setRoute] = useState<Route>(currentRoute());

  useEffect(() => {
    api.me().then(setMe).catch(() => setMe(null));
    const onHash = () => setRoute(currentRoute());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const reloadDescribe = useCallback(async () => setDescribe(await api.describe()), []);
  useEffect(() => {
    if (me) reloadDescribe();
  }, [me, reloadDescribe]);

  if (me === undefined) return null;
  if (me === null) {
    const guess: Locale = navigator.language.startsWith("zh") ? "zh" : "en";
    return <I18nProvider locale={guess}><UiProvider><Login onLogin={setMe} /></UiProvider></I18nProvider>;
  }
  if (describe === null) return null;
  const setLocale = async (locale: Locale) => {
    await api.setLocale(locale);
    setMe({ ...me, locale });
  };
  return (
    <I18nProvider locale={me.locale}>
      <Shell me={me} describe={describe} route={route} reloadDescribe={reloadDescribe} setLocale={setLocale} signOut={async () => { disconnect(); await api.logout(); setMe(null); }} />
    </I18nProvider>
  );
}

function Shell({ me, describe, route, reloadDescribe, setLocale, signOut }: {
  me: Me;
  describe: Describe;
  route: Route;
  reloadDescribe: () => Promise<void>;
  setLocale: (locale: Locale) => Promise<void>;
  signOut: () => Promise<void>;
}) {
  const t = useT();
  const [projects, setProjects] = useState<Project[]>([]);
  const [creating, setCreating] = useState(false);
  const reloadProjects = useCallback(async () => setProjects(await api.projects()), []);
  useEffect(() => {
    reloadProjects();
  }, [reloadProjects]);
  useLive(["projects"], reloadProjects);
  useLive(["catalog"], reloadDescribe);
  const link = (to: Route, label: string, icon: ReactNode) => (
    <a className={`nav-item ${route === to || route.startsWith(`${to}/`) ? "active" : ""}`} href={`#/${to}`}>{icon}<span className="nav-text">{label}</span></a>
  );
  const projectPath = route.startsWith("project/") ? route.slice("project/".length).split("/") : null;
  const project = projectPath?.[0] ?? null;
  const snapshot = projectPath?.[1] === "snapshot" ? projectPath[2] : null;
  useEffect(() => {
    if (route === "members" && me.role !== "admin") window.location.hash = "#/compose";
  }, [route, me.role]);
  return (
    <UiProvider>
      <AppContext.Provider value={{ me, describe, reloadDescribe, projects, reloadProjects }}>
        <div className="shell">
          <nav className="sidebar">
            <div className="brand">
              <Mark />
              <div className="brand-name">effeval</div>
            </div>
            {link("compose", t("New evaluation"), <FlaskConical size={16} />)}
            {link("runs", t("Runs"), <ListChecks size={16} />)}
            <div className="nav-label row">
              <span>{t("Projects")}</span><span className="spacer" />
              <button className="nav-add" aria-label={t("New project")} onClick={() => setCreating(true)}><Plus size={13} /></button>
            </div>
            {projects.map((p) => link(`project/${p.name}`, p.name, <FolderClosed size={16} />))}
            <div className="nav-label">{t("Results")}</div>
            {link("mine", t("My evaluations"), <SheetIcon size={16} />)}
            {link("pool", t("Lab pool"), <UsersRound size={16} />)}
            {link("import", t("Import"), <FileInput size={16} />)}
            <div className="nav-label">{t("Lab")}</div>
            {link("models", t("Models"), <Boxes size={16} />)}
            {link("methods", t("Methods"), <Puzzle size={16} />)}
            {link("benchmarks", t("Benchmarks"), <Target size={16} />)}
            {me.role === "admin" && link("members", t("Members"), <Users size={16} />)}
            <div className="sidebar-foot">
              {link("trash", t("Trash"), <Trash2 size={16} />)}
              <DropdownMenu.Root>
                <DropdownMenu.Trigger asChild>
                  <button className="user-chip">
                    <span className="avatar">{initials(me.username)}</span>
                    <span className="grow">
                      <div className="text-sm" style={{ fontWeight: 550 }}>{me.username}</div>
                      <div className="faint tiny">{t(me.role)}</div>
                    </span>
                    <ChevronsUpDown size={14} color="var(--text-3)" />
                  </button>
                </DropdownMenu.Trigger>
                <DropdownMenu.Portal>
                  <DropdownMenu.Content className="menu-content" side="top" align="start" sideOffset={6} style={{ minWidth: 200 }}>
                    <DropdownMenu.Item className="menu-item" onSelect={() => (window.location.hash = "#/settings")}><SettingsIcon size={14} />{t("Settings")}</DropdownMenu.Item>
                    <DropdownMenu.Separator className="menu-sep" />
                    <DropdownMenu.Sub>
                      <DropdownMenu.SubTrigger className="menu-item">
                        <Languages size={14} />{t("Language")}
                        <span className="faint small ml-auto">{LANGUAGES[me.locale]}</span><ChevronRight size={14} className="faint" />
                      </DropdownMenu.SubTrigger>
                      <DropdownMenu.Portal>
                        <DropdownMenu.SubContent className="menu-content" sideOffset={6} alignOffset={-4} style={{ minWidth: 150 }}>
                          {(Object.keys(LANGUAGES) as Locale[]).map((l) => (
                            <DropdownMenu.Item key={l} className="menu-item" onSelect={() => setLocale(l)}>
                              <span style={{ width: 14, display: "inline-flex" }}>{me.locale === l && <Check size={14} />}</span>{LANGUAGES[l]}
                            </DropdownMenu.Item>
                          ))}
                        </DropdownMenu.SubContent>
                      </DropdownMenu.Portal>
                    </DropdownMenu.Sub>
                    <DropdownMenu.Separator className="menu-sep" />
                    <DropdownMenu.Item className="menu-item" onSelect={signOut}><LogOut size={14} />{t("Sign out")}</DropdownMenu.Item>
                  </DropdownMenu.Content>
                </DropdownMenu.Portal>
              </DropdownMenu.Root>
            </div>
          </nav>
          <main className="main">
            {route === "compose" && <Compose />}
            {route === "runs" && <Runs />}
            {route === "mine" && <Results key="mine" scope="mine" />}
            {route === "pool" && <Results key="pool" scope="pool" />}
            {project !== null && !snapshot && <Results key={route} scope={`project:${project}`} />}
            {project !== null && snapshot && <SnapshotView key={route} project={project} snapshotId={snapshot} />}
            {route === "trash" && <Trash />}
            {route === "import" && <Import />}
            {route === "models" && <Models />}
            {route === "methods" && <Methods />}
            {route === "benchmarks" && <Benchmarks />}
            {route === "settings" && <Settings onLocale={setLocale} />}
            {route === "members" && me.role === "admin" && <Members />}
          </main>
        </div>
        <NewProjectDialog open={creating} onOpenChange={setCreating} />
      </AppContext.Provider>
    </UiProvider>
  );
}
