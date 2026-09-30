import { useEffect } from "react";
import { useLocation } from "react-router-dom";
import { useWorkspaceTarget } from "@/api/ApiProvider";
import { useExtension } from "../extension/ExtensionProvider";
import { documentTitle, pageLabel } from "./pageTitle";

/** Keeps the browser tab title on the current page and workspace. */
export function DocumentTitle() {
  const { pathname } = useLocation();
  const extension = useExtension();
  const workspace = useWorkspaceTarget()?.label;

  useEffect(() => {
    document.title = documentTitle(pageLabel(pathname, extension), workspace);
  }, [pathname, extension, workspace]);

  return null;
}
