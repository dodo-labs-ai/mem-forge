const MINUTE_MS = 60_000;
const HOUR_MINUTES = 60;
const DAY_HOURS = 24;
const WEEK_DAYS = 7;

const numberFormat = new Intl.NumberFormat("en-US");
const dateFormat = new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric" });
const dateTimeFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  hour: "numeric",
  minute: "2-digit",
});

export function formatCount(value: number): string {
  return numberFormat.format(value);
}

/** "1 page", "412 pages". */
export function pluralize(count: number, singular: string, plural = `${singular}s`): string {
  return `${formatCount(count)} ${count === 1 ? singular : plural}`;
}

export function formatDate(iso: string): string {
  return dateFormat.format(new Date(iso));
}

export function formatDateTime(iso: string): string {
  return dateTimeFormat.format(new Date(iso));
}

/** "just now", "5 min ago", "3 h ago", "2 days ago", then a calendar date. */
export function formatRelative(iso: string, now: Date = new Date()): string {
  const minutes = Math.round((now.getTime() - new Date(iso).getTime()) / MINUTE_MS);
  if (minutes < 1) return "just now";
  if (minutes < HOUR_MINUTES) return `${minutes} min ago`;
  const hours = Math.round(minutes / HOUR_MINUTES);
  if (hours < DAY_HOURS) return `${hours} h ago`;
  const days = Math.round(hours / DAY_HOURS);
  if (days < WEEK_DAYS) return `${days} ${days === 1 ? "day" : "days"} ago`;
  return formatDate(iso);
}

/** "in 5 min", "in 3 h", or a date for anything further out. */
export function formatUntil(iso: string, now: Date = new Date()): string {
  const minutes = Math.round((new Date(iso).getTime() - now.getTime()) / MINUTE_MS);
  if (minutes <= 0) return "now";
  if (minutes < HOUR_MINUTES) return `in ${minutes} min`;
  const hours = Math.round(minutes / HOUR_MINUTES);
  if (hours < DAY_HOURS) return `in ${hours} h`;
  return formatDateTime(iso);
}

/** "Every 30 min", "Every 6 h", "Daily", "Every 3 days". */
export function formatInterval(intervalMinutes: number): string {
  if (intervalMinutes < HOUR_MINUTES) return `Every ${intervalMinutes} min`;
  const hours = intervalMinutes / HOUR_MINUTES;
  if (hours < DAY_HOURS) return Number.isInteger(hours) ? `Every ${hours} h` : `Every ${intervalMinutes} min`;
  const days = hours / DAY_HOURS;
  if (days === 1) return "Daily";
  return Number.isInteger(days) ? `Every ${days} days` : `Every ${hours} h`;
}
