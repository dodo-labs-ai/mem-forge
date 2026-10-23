import { STANDALONE_TARGET } from "@/api/workspace";
import { APP_BASE_PATH } from "@/app/basePath";
import type { AdminExtension } from "@/app/extension";
import { renderAdminApp } from "@/app/render";

/**
 * The e2e harness for a shared workspace, built only in `--mode e2e`. Only
 * the Cloud extension selects a workspace other people use, so this page
 * renders the app with an extension that does just that, against the same
 * API as the standalone build. `?page=sources` opens that page of the app.
 */
const sharedWorkspace: AdminExtension = {
  id: "e2e-shared-workspace",
  // Setting the same target again changes nothing, so this holds on every render.
  Wrapper: ({ children, workspace }) => {
    workspace.setTarget({ ...STANDALONE_TARGET, shared: true });
    return children;
  },
};

const root = document.getElementById("root");
if (!root) throw new Error("The page has no #root element.");
const page = new URLSearchParams(window.location.search).get("page") ?? "";
window.history.replaceState(null, "", APP_BASE_PATH + page);
renderAdminApp(root, sharedWorkspace);
