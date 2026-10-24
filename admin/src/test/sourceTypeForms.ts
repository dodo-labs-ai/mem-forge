import type { components } from "@/api/schema.gen";
import forms from "./fixtures/sourceTypeForms.json";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];
type GeneConfigSchema = components["schemas"]["GeneConfigSchemaResponse"];

/**
 * The server's real `GET /genes` and config-schema answers for every
 * registered type, written by `scripts/export_source_type_forms.py`.
 */
export const SOURCE_TYPE_FORMS = forms as unknown as { genes: GeneMetadata[]; schemas: Record<string, GeneConfigSchema> };

/** The config-schema answer of one registered type. */
export function sourceTypeSchema(sourceType: string): GeneConfigSchema {
  const schema = SOURCE_TYPE_FORMS.schemas[sourceType];
  if (!schema) throw new Error(`No form fixture for ${sourceType}`);
  return schema;
}
