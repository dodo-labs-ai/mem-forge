import { Navigate, Route, Routes } from "react-router-dom";
import { ENTITIES_PATH } from "@/lib/paths";
import { EntitiesPage } from "./EntitiesPage";
import { EntityDetailPage } from "./EntityDetailPage";

/** The Entities list at `/entities` and one entity at `/entities/:entityId`. */
export function EntitiesRoutes() {
  return (
    <Routes>
      <Route index element={<EntitiesPage />} />
      <Route path=":entityId" element={<EntityDetailPage />} />
      <Route path="*" element={<Navigate to={ENTITIES_PATH} replace />} />
    </Routes>
  );
}
