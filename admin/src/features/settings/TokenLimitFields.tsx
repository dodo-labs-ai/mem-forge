import { useFormContext } from "react-hook-form";
import { Input } from "@/ui/input";
import { describedBy } from "./describedBy";
import { FormField } from "./FormField";
import type { ModelFormValues } from "./model/modelForm";
import { tokenLimitField } from "./model/providerFields";

interface TokenLimitFieldsProps {
  /** The prefix of the ids of the group and its fields. */
  id: string;
  /** The limits the model may state, by the name the server gives them. */
  names: readonly string[];
  /** True when the latest validation found that the server has no limits for the model. */
  missing: boolean;
}

/**
 * The context window and the maximum output of a generation model. LiteLLM
 * knows them for most hosted models; a model it has no limits for, which is
 * common for local ones, passes validation only once they are entered here.
 */
export function TokenLimitFields({ id, names, missing }: TokenLimitFieldsProps) {
  const { register, formState } = useFormContext<ModelFormValues>();
  const headingId = `${id}-heading`;
  const purposeId = `${id}-purpose`;

  return (
    <div role="group" aria-labelledby={headingId} aria-describedby={purposeId} className="space-y-3 rounded-lg border p-3.5">
      <div className="space-y-0.5">
        <h3 id={headingId} className="text-sm font-medium text-foreground">
          Token limits
        </h3>
        <p id={purposeId} className="text-xs text-muted-foreground">
          MemForge sizes every request to these limits, so that the model is never sent more than it can read or asked
          for more than it can write.
          {missing ? " It has none for this model: take them from the model card or from your model server." : null}
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {names.map((name) => {
          const copy = tokenLimitField(name);
          const fieldId = `${id}-${name}`;
          const error = formState.errors.tokenLimits?.[name]?.message;
          return (
            <FormField
              key={name}
              label={copy.label}
              htmlFor={fieldId}
              hint={copy.hint}
              hintId={`${fieldId}-hint`}
              error={error}
              errorId={`${fieldId}-error`}
            >
              <Input
                id={fieldId}
                inputMode="numeric"
                spellCheck={false}
                aria-invalid={error ? true : undefined}
                aria-describedby={describedBy(error && `${fieldId}-error`, `${fieldId}-hint`)}
                className="font-mono"
                {...register(`tokenLimits.${name}`)}
              />
            </FormField>
          );
        })}
      </div>
    </div>
  );
}
