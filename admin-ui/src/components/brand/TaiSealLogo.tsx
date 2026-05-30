import type { SVGProps } from "react";

import { TAI_SEAL_LOGO_TITLE } from "@/brand";

type TaiSealLogoProps = SVGProps<SVGSVGElement> & {
  title?: string;
};

export function TaiSealLogo({ title = TAI_SEAL_LOGO_TITLE, ...props }: TaiSealLogoProps) {
  return (
    <svg viewBox="0 0 96 96" role="img" aria-label={title} fill="none" {...props}>
      <rect width="96" height="96" rx="18" fill="#F8FAFC" />
      <circle cx="48" cy="48" r="34" fill="#111827" />
      <path d="M24 62L39 42L49 54L62 35L73 62H24Z" fill="white" />
      <path d="M31 68H66" stroke="#F97316" strokeWidth="5" strokeLinecap="round" />
      <circle cx="64" cy="30" r="5" fill="#F97316" />
    </svg>
  );
}
