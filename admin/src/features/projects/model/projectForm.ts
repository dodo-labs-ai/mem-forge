import { z } from "zod";
import { isReservedProjectKey } from "@/api";

export const RESERVED_CODE_MESSAGE = "That code is reserved for system use. Pick a different one.";

const name = z.string().trim().min(1, "Enter a name.");

export const createProjectSchema = z.object({
  name,
  code: z
    .string()
    .trim()
    .refine((code) => !isReservedProjectKey(code.toUpperCase()), RESERVED_CODE_MESSAGE),
});

export const editProjectSchema = z.object({ name });

export type CreateProjectInput = z.input<typeof createProjectSchema>;
export type CreateProjectValues = z.output<typeof createProjectSchema>;
export type EditProjectInput = z.input<typeof editProjectSchema>;
export type EditProjectValues = z.output<typeof editProjectSchema>;

export interface CreateProjectBody {
  name: string;
  /** Left out so the server derives the code from the name. */
  key?: string;
}

export function createProjectBody({ name, code }: CreateProjectValues): CreateProjectBody {
  return code === "" ? { name } : { name, key: code };
}
