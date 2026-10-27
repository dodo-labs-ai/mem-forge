interface BrandMarkProps {
  className?: string;
}

/** The MemForge mark: two peaks forming an M, with an orange spark. */
export function BrandMark({ className }: BrandMarkProps) {
  return (
    <svg viewBox="0 0 96 96" role="img" aria-label="MemForge" className={className}>
      <rect x="4" y="4" width="88" height="88" rx="22" fill="#111827" />
      <path
        d="M22 70V34L38 54L48 39L58 54L74 34V70"
        fill="none"
        stroke="#FFFFFF"
        strokeWidth="8"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx="48" cy="21" r="5.5" className="fill-brand" />
    </svg>
  );
}
