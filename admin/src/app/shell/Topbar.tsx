import { useExtension } from "../extension/ExtensionProvider";

/**
 * Holds the extension's workspace switcher (start) and account menu (end).
 * The standalone build has a single local workspace and no accounts, so the
 * bar renders nothing there.
 */
export function Topbar() {
  const slots = useExtension()?.topbarSlots ?? [];
  if (slots.length === 0) return null;
  const start = slots.filter((slot) => slot.placement === "start");
  const end = slots.filter((slot) => slot.placement === "end");
  return (
    <header className="flex h-14 shrink-0 items-center gap-3 border-b bg-background px-6">
      {start.map((slot) => (
        <div key={slot.id}>{slot.render()}</div>
      ))}
      <div className="ml-auto flex items-center gap-2">
        {end.map((slot) => (
          <div key={slot.id}>{slot.render()}</div>
        ))}
      </div>
    </header>
  );
}
