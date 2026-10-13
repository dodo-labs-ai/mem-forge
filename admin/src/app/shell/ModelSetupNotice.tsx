import { AlertTriangle } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useModelSettings } from "@/api";
import { Notice } from "@/patterns";
import { buttonVariants } from "@/ui/button";
import { useExtension } from "../extension/ExtensionProvider";
import { modelSetupMessage, settingsLinkTarget, showsModelSetup } from "./modelSetup";

function GenerationModelNotice() {
  const settings = useModelSettings();
  const settingsPath = settingsLinkTarget(useExtension());
  const message = settings.data ? modelSetupMessage(settings.data) : null;
  if (message === null) return null;
  return (
    <Notice
      tone="warn"
      icon={AlertTriangle}
      title={message.title}
      className="mb-5"
      action={
        settingsPath ? (
          <Link to={settingsPath} className={buttonVariants({ variant: "outline", size: "sm", className: "text-foreground" })}>
            Open Settings
          </Link>
        ) : null
      }
    >
      {message.body}
    </Notice>
  );
}

/**
 * Tells the user, above every product page but Settings, that no generation
 * model is ready. It stays silent while the settings load and when they
 * cannot be read: the page below reports its own failures.
 */
export function ModelSetupNotice() {
  const { pathname } = useLocation();
  return showsModelSetup(pathname) ? <GenerationModelNotice /> : null;
}
