import { AlertTriangle } from "lucide-react";
import { Link, useLocation } from "react-router-dom";
import { useModelSettings } from "@/api";
import { Notice } from "@/patterns";
import { buttonVariants } from "@/ui/button";
import { useExtension } from "../extension/ExtensionProvider";
import { modelSetupMessages, settingsLinkTarget, showsModelSetup } from "./modelSetup";

function ModelNotices() {
  const settings = useModelSettings();
  const settingsPath = settingsLinkTarget(useExtension());
  const messages = settings.data ? modelSetupMessages(settings.data) : [];
  if (messages.length === 0) return null;
  return (
    <div className="mb-5 space-y-3">
      {messages.map((message) => (
        <Notice
          key={message.title}
          tone="warn"
          icon={AlertTriangle}
          title={message.title}
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
      ))}
    </div>
  );
}

/**
 * Tells the user, above every product page but Settings, what about the
 * models keeps MemForge from working fully: no generation model is ready, or
 * vector search is off until the search index has been rebuilt. It stays silent
 * while the settings load and when they cannot be read: the page below
 * reports its own failures.
 */
export function ModelSetupNotice() {
  const { pathname } = useLocation();
  return showsModelSetup(pathname) ? <ModelNotices /> : null;
}
