import { Suspense } from "react";
import { ChoiceGroup, describedBy, FormField } from "@/patterns";
import { Checkbox } from "@/ui/checkbox";
import { Input } from "@/ui/input";
import { Skeleton } from "@/ui/skeleton";
import { Switch } from "@/ui/switch";
import { TagInput } from "@/ui/tag-input";
import type { ConfigField, ConfigOption, FieldValue, ServerConfig } from "./model/configForm";
import { configWidget, type ConfigControlProps } from "./widgets";

interface ConfigFieldControlProps {
  field: ConfigField;
  /** The field's choices whose conditions hold. */
  choices: readonly ConfigOption[];
  value: FieldValue;
  onChange: (value: FieldValue) => void;
  onBlur?: () => void;
  /** What has to be corrected, from the form or from the server. */
  error?: string;
  /** A quiet line under the control, such as where a derived value came from. */
  note?: string;
  readOnly?: boolean;
  secretConfigured?: boolean;
  sourceType: string;
  sourceId?: string;
  /** The configuration the form would save, which a widget may read. */
  config: ServerConfig;
  /** Prefix of the ids, so two forms on one page do not share them. */
  idPrefix?: string;
}

const STORED_SECRET_HINT = "A value is stored. Leave this blank to keep it.";

/**
 * One configuration field: its label, help text and issue around the control
 * its `field_type` calls for, or the widget its `widget` names. The control's
 * `aria-describedby` names the help text and the issue.
 */
export function ConfigFieldControl({
  field,
  choices,
  value,
  onChange,
  onBlur,
  error,
  note,
  readOnly,
  secretConfigured,
  sourceType,
  sourceId,
  config,
  idPrefix = "config",
}: ConfigFieldControlProps) {
  const id = `${idPrefix}-${field.key}`;
  const hintId = `${id}-hint`;
  const errorId = `${id}-error`;
  const hint = [field.help_text, field.field_type === "secret" && secretConfigured ? STORED_SECRET_HINT : "", note ?? ""]
    .filter(Boolean)
    .join(" ");
  const props: ConfigControlProps = {
    field,
    id,
    labelId: `${id}-label`,
    describedBy: describedBy(error && errorId, hint && hintId),
    choices,
    value,
    onChange,
    onBlur,
    invalid: Boolean(error),
    readOnly,
    secretConfigured,
    sourceType,
    sourceId,
    config,
  };
  const widget = configWidget(field.widget, props);

  return (
    <FormField label={field.label} htmlFor={id} labelId={props.labelId} hint={hint} hintId={hintId} error={error} errorId={errorId}>
      {widget ? (
        <Suspense fallback={<Skeleton className="h-8 w-full" />}>{widget}</Suspense>
      ) : (
        <PlainControl {...props} />
      )}
    </FormField>
  );
}

/** The control for a field's type. */
function PlainControl({ field, id, labelId, describedBy, choices, value, onChange, onBlur, invalid, readOnly, secretConfigured }: ConfigControlProps) {
  const aria = { "aria-describedby": describedBy, "aria-invalid": invalid || undefined };
  switch (field.field_type) {
    case "boolean":
      return (
        <div>
          <Switch
            id={id}
            aria-labelledby={labelId}
            {...aria}
            checked={value === true}
            disabled={readOnly}
            onCheckedChange={(checked) => {
              onChange(checked);
              onBlur?.();
            }}
          />
        </div>
      );
    case "select":
      return (
        <ChoiceGroup
          id={id}
          aria-labelledby={labelId}
          {...aria}
          choices={choices.map((choice) => ({ value: choice.value, label: choice.label, description: choice.help_text || undefined }))}
          value={typeof value === "string" ? value : ""}
          disabled={readOnly}
          onValueChange={(next) => {
            onChange(next);
            onBlur?.();
          }}
        />
      );
    case "multi_select": {
      const selected = Array.isArray(value) ? value : [];
      return (
        <div id={id} role="group" aria-labelledby={labelId} {...aria} className="grid gap-2 sm:grid-cols-2">
          {choices.map((choice) => {
            const optionId = `${id}-${choice.value}`;
            return (
              <label key={choice.value} className="flex items-center gap-2 text-sm">
                <Checkbox
                  aria-labelledby={`${optionId}-label`}
                  checked={selected.includes(choice.value)}
                  disabled={readOnly}
                  onCheckedChange={(checked) => {
                    onChange(checked ? [...selected, choice.value] : selected.filter((item) => item !== choice.value));
                    onBlur?.();
                  }}
                />
                <span id={`${optionId}-label`}>{choice.label}</span>
              </label>
            );
          })}
        </div>
      );
    }
    case "tag_list":
      return (
        <TagInput
          id={id}
          aria-labelledby={labelId}
          {...aria}
          value={Array.isArray(value) ? value : []}
          placeholder={field.placeholder}
          disabled={readOnly}
          onValueChange={onChange}
          onBlur={onBlur}
        />
      );
    default:
      return (
        <Input
          id={id}
          {...aria}
          type={field.field_type === "secret" ? "password" : field.field_type === "url" ? "url" : "text"}
          inputMode={field.field_type === "integer" ? "numeric" : undefined}
          autoComplete={field.field_type === "secret" ? "new-password" : "off"}
          placeholder={field.field_type === "secret" && secretConfigured ? "Stored" : field.placeholder}
          value={typeof value === "string" ? value : ""}
          readOnly={readOnly}
          onChange={(event) => onChange(event.target.value)}
          onBlur={onBlur}
        />
      );
  }
}
