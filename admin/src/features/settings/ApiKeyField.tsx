import { Eye, EyeOff } from "lucide-react";
import { useState } from "react";
import { useFormContext } from "react-hook-form";
import { cn } from "@/lib/cn";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { describedBy } from "./describedBy";
import type { ModelFormValues } from "./model/modelForm";
import { apiKeyStatus, type SavedKey } from "./model/savedKey";

interface ApiKeyFieldProps {
  id: string;
  /** The saved key a blank field keeps for the endpoint being edited. */
  savedKey: SavedKey;
  hasTypedKey: boolean;
  /** The id of the message that says what is wrong with the field, when something is. */
  errorId: string | undefined;
  /** Called when the key the server would use changes. */
  onKeyChange: () => void;
}

/**
 * The API key input. A saved key is never sent to the browser, so the field
 * starts empty and a status line says which key is used and what saving does.
 */
export function ApiKeyField({ id, savedKey, hasTypedKey, errorId, onKeyChange }: ApiKeyFieldProps) {
  const { register } = useFormContext<ModelFormValues>();
  const [visible, setVisible] = useState(false);
  const statusId = `${id}-status`;
  const status = apiKeyStatus(savedKey, hasTypedKey);

  return (
    <>
      <div className="relative">
        <Input
          id={id}
          type={visible ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          placeholder={savedKey.status === "kept" ? "Leave blank to keep the current key" : "API key"}
          aria-invalid={errorId ? true : undefined}
          aria-describedby={describedBy(errorId, statusId)}
          className="pr-9 font-mono"
          {...register("apiKey", { onChange: onKeyChange })}
        />
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          aria-label={visible ? "Hide API key" : "Show API key"}
          onClick={() => setVisible((current) => !current)}
          className="absolute top-1/2 right-0.5 -translate-y-1/2"
        >
          {visible ? <EyeOff /> : <Eye />}
        </Button>
      </div>
      <p id={statusId} className={cn("text-xs", status.warning ? "text-tone-warn-foreground" : "text-muted-foreground")}>
        {status.message}
      </p>
    </>
  );
}
