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
  /** True when the latest validation found that the server has no limits for the model, so they are required. */
  missing: boolean;
  /** The provider whose model list the values were filled in from, while the fields still hold them. */
  listedBy: string | null;
}

/**
 * The context window and the maximum output of a generation model. LiteLLM
 * knows them for most hosted models; a model it has no limits for, which is
 * common for local ones, passes validation only once they are entered here.
 * A provider whose model list gives the limits fills them in when a model is
 * picked from it.
 */
export function TokenLimitFields({ id, names, missing, listedBy }: TokenLimitFieldsProps) {
  const { register, formState } = useFormContext<ModelFormValues>();
  const headingId = `${id}-heading`;
  const purposeId = `${id}-purpose`;
  const sourceId = `${id}-source`;

  return (
    <div
      role="group"
      aria-labelledby={headingId}
      aria-describedby={describedBy(purposeId, listedBy && sourceId)}
      className="space-y-3 rounded-lg border p-3.5"
    >
      <div className="space-y-0.5">
        <h3 id={headingId} className="text-sm font-medium text-foreground">
          Token limits
        </h3>
        <p id={purposeId} className="text-xs text-muted-foreground">
          MemForge sizes every request to these limits, so that the model is never sent more than it can read or asked
          for more than it can write.
          {missing ? " It has none for this model: take them from the model card or from your model server." : null}
        </p>
        {listedBy ? (
          <p id={sourceId} className="text-xs text-tone-live-foreground">
            Filled in from the {listedBy} model list. You can change them.
          </p>
        ) : null}
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {names.map((name) => {
          const copy = tokenLimitField(name, { required: missing });
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
                aria-describedby={describedBy(error && `${fieldId}-error`, copy.hint && `${fieldId}-hint`)}
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
