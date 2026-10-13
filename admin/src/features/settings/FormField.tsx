import type { ReactNode } from "react";
import { Label } from "@/ui/label";

interface FormFieldProps {
  label: string;
  htmlFor: string;
  /** Lets a control that is not a native input name itself with `aria-labelledby`. */
  labelId?: string;
  /** A line of help under the control; give the control `aria-describedby={hintId}`. */
  hint?: string | null;
  hintId?: string;
  /** What has to be corrected; give the control `aria-describedby={errorId}`. */
  error?: string;
  errorId?: string;
  children: ReactNode;
}

export function FormField({ label, htmlFor, labelId, hint, hintId, error, errorId, children }: FormFieldProps) {
  return (
    <div className="space-y-1.5">
      <Label id={labelId} htmlFor={htmlFor}>
        {label}
      </Label>
      {children}
      {error ? (
        <p id={errorId} className="text-xs text-tone-danger-foreground">
          {error}
        </p>
      ) : null}
      {hint ? (
        <p id={hintId} className="text-xs text-muted-foreground">
          {hint}
        </p>
      ) : null}
    </div>
  );
}
