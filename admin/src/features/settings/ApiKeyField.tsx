import { Eye, EyeOff, Undo2, X } from "lucide-react";
import { useState } from "react";
import { useFormContext, useWatch } from "react-hook-form";
import { cn } from "@/lib/cn";
import { Button } from "@/ui/button";
import { Input } from "@/ui/input";
import { apiKeyStatus, maskedKey, type SavedKey } from "./model/apiKeyStatus";
import { keyIntent, type SettingsValues } from "./model/settingsForm";
import type { EndpointKind } from "./model/types";
import { ENVIRONMENT_PLACEHOLDER, ENDPOINT_INPUT_CLASS } from "./constants";

interface ApiKeyFieldProps {
  id: string;
  kind: EndpointKind;
  saved: SavedKey;
  readOnly: boolean;
  /** Called when the key the server would test with changes. */
  onKeyChange: () => void;
}

/**
 * The API key input. The saved key is never sent to the browser in full, so
 * the field starts empty and a status line says what saving will do with it.
 */
export function ApiKeyField({ id, kind, saved, readOnly, onKeyChange }: ApiKeyFieldProps) {
  const { register, setValue, control } = useFormContext<SettingsValues>();
  const endpoint = useWatch({ control, name: kind });
  const [visible, setVisible] = useState(false);
  const statusId = `${id}-status`;

  if (readOnly) {
    return (
      <Input
        id={id}
        readOnly
        value={saved.last4 ? maskedKey(saved.last4) : ""}
        placeholder={ENVIRONMENT_PLACEHOLDER}
        className={ENDPOINT_INPUT_CLASS}
      />
    );
  }

  const status = apiKeyStatus(saved, keyIntent(endpoint));
  function setRemoveSavedKey(remove: boolean) {
    setValue(`${kind}.removeSavedKey`, remove);
    onKeyChange();
  }

  return (
    <>
      <div className="relative">
        <Input
          id={id}
          type={visible ? "text" : "password"}
          autoComplete="off"
          spellCheck={false}
          placeholder={saved.set ? "Leave blank to keep saved key" : "Required for hosted providers"}
          aria-describedby={statusId}
          className="pr-9 font-mono"
          {...register(`${kind}.apiKey`, { onChange: () => setRemoveSavedKey(false) })}
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
      <div className="flex min-h-7 flex-wrap items-center gap-1.5">
        <p id={statusId} className={cn("text-xs", status.warning ? "text-tone-warn-foreground" : "text-muted-foreground")}>
          {status.message}
        </p>
        {status.action === "remove" ? (
          <Button type="button" variant="ghost" size="sm" aria-label="Remove saved key" onClick={() => setRemoveSavedKey(true)}>
            <X />
            Remove
          </Button>
        ) : null}
        {status.action === "undo" ? (
          <Button type="button" variant="ghost" size="sm" aria-label="Undo key removal" onClick={() => setRemoveSavedKey(false)}>
            <Undo2 />
            Undo
          </Button>
        ) : null}
      </div>
    </>
  );
}
