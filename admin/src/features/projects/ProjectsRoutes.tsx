import { Navigate, Route, Routes } from "react-router-dom";
import { PROJECTS_PATH } from "@/lib/paths";
import { ProjectDetailPage } from "./ProjectDetailPage";
import { ProjectListPage } from "./ProjectListPage";

/** The Projects list at `/projects` and one project at `/projects/:key`. */
export function ProjectsRoutes() {
  return (
    <Routes>
      <Route index element={<ProjectListPage />} />
      <Route path=":key" element={<ProjectDetailPage />} />
      <Route path="*" element={<Navigate to={PROJECTS_PATH} replace />} />
    </Routes>
  );
}
