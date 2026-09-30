import { Route, Routes } from "react-router-dom";
import { MemoriesPage } from "./MemoriesPage";
import { MemoryDetailPage } from "./MemoryDetailPage";

/** The Memories list and, under the same segment, one memory's detail. */
export function MemoriesRoutes() {
  return (
    <Routes>
      <Route index element={<MemoriesPage />} />
      <Route path=":memoryId" element={<MemoryDetailPage />} />
    </Routes>
  );
}
