import type { ConfigField, ConfigOption, FieldValue, ServerConfig } from "./model/configForm";
import { lazyRegistry, type ComponentLoader } from "./lazyRegistry";

/** What a field's control, plain or interactive, is given. */
export interface ConfigControlProps {
  field: ConfigField;
  /** The id of the control, which the field's label points at. */
  id: string;
  /** The id of the field's label, for a control that names itself with `aria-labelledby`. */
  labelId: string;
  /** The ids of the field's help text and issue, for `aria-describedby`. */
  describedBy?: string;
  /** The field's choices whose conditions hold. */
  choices: readonly ConfigOption[];
  value: FieldValue;
  onChange: (value: FieldValue) => void;
  onBlur?: () => void;
  invalid?: boolean;
  /** The field is fixed after create and the Source exists. */
  readOnly?: boolean;
  /** A secret is stored, so a blank input keeps it. */
  secretConfigured?: boolean;
  /** The Source's type and, on an edit page, its id, for a widget that asks the server. */
  sourceType: string;
  sourceId?: string;
  /** The configuration the form would save, for a widget that reads other fields, such as the repository URL. */
  config: ServerConfig;
}

const LOADERS: Record<string, ComponentLoader<ConfigControlProps>> = {
  github_repo_paths: () => import("./widgets/GitHubRepoPathsPicker"),
  local_folder: () => import("./widgets/LocalFolderPicker"),
  teams_conversations: () => import("./widgets/TeamsConversationPicker"),
};

/**
 * Draws the interactive control a field's `widget` names, such as a repository
 * folder tree. A name with no entry returns `null`, and the field draws the
 * plain control for its type, so a form stays usable when the server names a
 * widget this admin UI does not have.
 */
export const configWidget = lazyRegistry(LOADERS);
