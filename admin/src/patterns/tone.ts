/** Semantic status colors shared by badges, cards and notices. */
export type Tone = "ok" | "live" | "warn" | "danger" | "idle";

export const toneText: Record<Tone, string> = {
  ok: "text-tone-ok-foreground",
  live: "text-tone-live-foreground",
  warn: "text-tone-warn-foreground",
  danger: "text-tone-danger-foreground",
  idle: "text-tone-idle-foreground",
};

export const toneSurface: Record<Tone, string> = {
  ok: "bg-tone-ok-soft text-tone-ok-foreground",
  live: "bg-tone-live-soft text-tone-live-foreground",
  warn: "bg-tone-warn-soft text-tone-warn-foreground",
  danger: "bg-tone-danger-soft text-tone-danger-foreground",
  idle: "bg-tone-idle-soft text-tone-idle-foreground",
};

export const toneDot: Record<Tone, string> = {
  ok: "bg-tone-ok",
  live: "bg-tone-live",
  warn: "bg-tone-warn",
  danger: "bg-tone-danger",
  idle: "bg-tone-idle",
};
