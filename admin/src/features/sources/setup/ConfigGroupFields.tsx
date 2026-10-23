import { ChevronRightIcon } from "lucide-react";
import { Suspense, useState } from "react";
import { ConfigFieldControl } from "./ConfigFieldControl";
import {
  conditionsHoldFor,
  displayValue,
  serializeConfig,
  visibleChoices,
  visibleFields,
  type ConfigField,
  type ConfigFormState,
  type ConfigGroup,
  type ConfigSchema,
  type FieldValue,
} from "./model/configForm";
import { groupFields } from "./model/setupSteps";
import { configPanel } from "./panels";

interface ConfigGroupFieldsProps {
  schema: ConfigSchema;
  group: ConfigGroup;
  state: ConfigFormState;
  sourceType: string;
  /** The edited Source; absent while a Source is created. */
  sourceId?: string;
  /** What has to be corrected, by field key. */
  issues?: Readonly<Record<string, string>>;
  /** Lines under fields, such as "Filled from the URL", by field key. */
  notes?: Readonly<Record<string, string>>;
  onChange: (key: string, value: FieldValue) => void;
  onBlur?: (key: string) => void;
}

/**
 * The fields of one of a type's groups, drawn from its form description: the
 * visible fields in order, then its `advanced` fields in a collapsed
 * "Advanced settings" that opens by itself when one of them has an issue,
 * then the group's visible panels. On an edit page a field fixed after
 * create is read-only.
 */
export function ConfigGroupFields({ schema, group, state, sourceType, sourceId, issues = {}, notes = {}, onChange, onBlur }: ConfigGroupFieldsProps) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const visible = new Set(visibleFields(schema, state).map((field) => field.key));
  const fields = groupFields(schema, group.key).filter((field) => visible.has(field.key));
  const basic = fields.filter((field) => !field.advanced);
  const advanced = fields.filter((field) => field.advanced);
  const advancedShown = advancedOpen || advanced.some((field) => issues[field.key]);
  const panels = group.panels.filter((panel) => conditionsHoldFor(schema, panel.visible_when, state));

  function control(field: ConfigField) {
    return (
      <ConfigFieldControl
        key={field.key}
        field={field}
        choices={visibleChoices(schema, field, state)}
        value={displayValue(field, state)}
        onChange={(value) => onChange(field.key, value)}
        onBlur={onBlur ? () => onBlur(field.key) : undefined}
        error={issues[field.key]}
        note={notes[field.key]}
        readOnly={sourceId !== undefined && field.fixed_after_create}
        secretConfigured={state.stored?.[`${field.key}_configured`] === true}
        sourceType={sourceType}
        sourceId={sourceId}
      />
    );
  }

  return (
    <div className="space-y-5">
      {group.description ? <p className="text-sm text-muted-foreground">{group.description}</p> : null}
      {basic.map(control)}
      {advanced.length > 0 ? (
        <details open={advancedShown} onToggle={(event) => setAdvancedOpen(event.currentTarget.open)} className="group/advanced">
          <summary className="flex cursor-pointer list-none items-center gap-1.5 text-sm font-medium text-subtle-foreground select-none [&::-webkit-details-marker]:hidden">
            <ChevronRightIcon aria-hidden className="size-4 transition-transform group-open/advanced:rotate-90" />
            Advanced settings
          </summary>
          <div className="mt-4 space-y-5">{advanced.map(control)}</div>
        </details>
      ) : null}
      {panels.map((panel) => (
        <Suspense key={panel.name} fallback={null}>
          {configPanel(panel.name, { sourceType, sourceId, config: serializeConfig(schema, state.values) })}
        </Suspense>
      ))}
    </div>
  );
}
