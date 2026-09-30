import type { ReactNode } from "react";
import { Label } from "@/ui/label";

interface FormFieldProps {
  label: string;
  htmlFor: string;
  children: ReactNode;
}

export function FormField({ label, htmlFor, children }: FormFieldProps) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
    </div>
  );
}
