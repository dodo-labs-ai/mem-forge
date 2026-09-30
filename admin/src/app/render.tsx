import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@/styles/index.css";
import { AdminApp } from "./AdminApp";
import { followSystemColorScheme } from "./colorScheme";
import type { AdminExtension } from "./extension/contract";

/** Renders the admin UI into `element`. The Cloud build passes its extension here. */
export function renderAdminApp(element: HTMLElement, extension?: AdminExtension) {
  followSystemColorScheme();
  createRoot(element).render(
    <StrictMode>
      <AdminApp extension={extension} />
    </StrictMode>,
  );
}
