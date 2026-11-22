import { useEffect, useRef } from "react";

/** Kinds of server data; a change to one reaches every page subscribed to it. */
export type Topic = "records" | "projects" | "runs" | "catalog" | "users";
const ALL: Topic[] = ["records", "projects", "runs", "catalog", "users"];
const DEBOUNCE_MS = 300;

type Listener = (topics: Topic[]) => void;
const listeners = new Set<Listener>();
let source: EventSource | null = null;

function connect() {
  if (source && source.readyState !== EventSource.CLOSED) return;
  let lost = false;
  source = new EventSource("/api/events");
  source.onmessage = (event) => {
    const { topics } = JSON.parse(event.data) as { topics: Topic[] };
    listeners.forEach((listener) => listener(topics));
  };
  source.onerror = () => {
    lost = true;
  };
  source.onopen = () => {
    // Changes made while the connection was down were not pushed: refresh everything once.
    if (lost) listeners.forEach((listener) => listener(ALL));
    lost = false;
  };
}

/** Close the event stream, e.g. on sign-out; the next subscription opens a new one. */
export function disconnect() {
  source?.close();
  source = null;
}

/** Call `refresh` (debounced) whenever another user or process changes one of `topics` on the server. */
export function useLive(topics: Topic[], refresh: () => void) {
  const latest = useRef(refresh);
  latest.current = refresh;
  const key = topics.join(",");
  useEffect(() => {
    connect();
    let timer: number | undefined;
    const listener: Listener = (changed) => {
      if (!changed.some((t) => key.split(",").includes(t))) return;
      window.clearTimeout(timer);
      timer = window.setTimeout(() => latest.current(), DEBOUNCE_MS);
    };
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
      window.clearTimeout(timer);
    };
  }, [key]);
}
