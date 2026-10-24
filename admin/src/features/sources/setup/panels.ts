import type { ServerConfig } from "./model/configForm";
import { lazyRegistry, type ComponentLoader } from "./lazyRegistry";

/** What an interactive panel of a group, such as a sign-in session, is given. */
export interface ConfigPanelProps {
  sourceType: string;
  /** The edited Source; absent while a Source is created. */
  sourceId?: string;
  /** The configuration the form would save. */
  config: ServerConfig;
}

const LOADERS: Record<string, ComponentLoader<ConfigPanelProps>> = {
  jira_session: () => import("./panels/JiraSessionPanel"),
  teams_session: () => import("./panels/TeamsSessionPanel"),
};

/** Draws the panel a group's `panels` entry names. A name with no entry draws nothing. */
export const configPanel = lazyRegistry(LOADERS);
