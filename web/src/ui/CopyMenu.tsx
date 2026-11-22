import { ChevronDown, Copy } from "lucide-react";
import { DropdownMenu } from "radix-ui";
import { useT } from "../i18n";
import { Button } from "./controls";

/** "Copy" button with a format menu: Markdown (pastes as a table in Slides / Docs) and, when given, LaTeX. */
export function CopyMenu({ onMarkdown, onLatex, size, label }: {
  onMarkdown: () => void;
  onLatex?: () => void;
  size?: "sm";
  label?: string;
}) {
  const t = useT();
  return (
    <DropdownMenu.Root>
      <DropdownMenu.Trigger asChild>
        <Button size={size} variant={size ? "ghost" : "secondary"} icon={<Copy size={13} />}>{label ?? t("Copy")}<ChevronDown size={12} /></Button>
      </DropdownMenu.Trigger>
      <DropdownMenu.Portal>
        <DropdownMenu.Content className="menu-content" align="end" sideOffset={4} style={{ minWidth: 240 }}>
          <DropdownMenu.Item className="menu-item" onSelect={onMarkdown}>
            <span className="grow">Markdown</span><span className="faint tiny">{t("pastes as a table in Slides / Docs")}</span>
          </DropdownMenu.Item>
          {onLatex && (
            <DropdownMenu.Item className="menu-item" onSelect={onLatex}>
              <span className="grow">LaTeX</span><span className="faint tiny">booktabs</span>
            </DropdownMenu.Item>
          )}
        </DropdownMenu.Content>
      </DropdownMenu.Portal>
    </DropdownMenu.Root>
  );
}
