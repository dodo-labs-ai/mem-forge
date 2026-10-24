import { Navigate, Route, Routes } from "react-router-dom";
import { SOURCES_PATH } from "@/lib/paths";
import { SourcesPage } from "./SourcesPage";
import { AddSourcePage } from "./setup/AddSourcePage";
import { NewSourcePage, SourceSettingsPage } from "./setup/SourceSetupPage";

/** The Sources list, the type picker, a new Source of one type, and one Source's settings. */
export function SourcesRoutes() {
  return (
    <Routes>
      <Route index element={<SourcesPage />} />
      <Route path="new" element={<AddSourcePage />} />
      <Route path="new/:type" element={<NewSourcePage />} />
      <Route path=":sourceId/settings" element={<SourceSettingsPage />} />
      <Route path="*" element={<Navigate to={SOURCES_PATH} replace />} />
    </Routes>
  );
}
