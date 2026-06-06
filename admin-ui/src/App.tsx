import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AppShell } from "@/components/layout/AppShell";
import { DEFAULT_QUERY_STALE_MS } from "@/lib/constants";
import { MemoriesPage } from "@/views/memories/MemoriesPage";
import { MemoryDetailPage } from "@/views/memories/MemoryDetailPage";
import { EntitiesPage } from "@/views/entities/EntitiesPage";
import { EntityDetailPage } from "@/views/entities/EntityDetailPage";
import { ProjectDetailPage } from "@/views/projects/ProjectDetailPage";
import { ProjectsPage } from "@/views/projects/ProjectsPage";
import { ReviewQueuePage } from "@/views/review/ReviewQueuePage";
import { ReviewDetailPage } from "@/views/review/ReviewDetailPage";
import { SourcesPage } from "@/views/sources/SourcesPage";
import { SettingsPage } from "@/views/settings/SettingsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: DEFAULT_QUERY_STALE_MS, retry: 1 },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<AppShell />}>
            <Route path="/memories" element={<MemoriesPage />} />
            <Route path="/memories/:id" element={<MemoryDetailPage />} />
            <Route path="/review" element={<ReviewQueuePage />} />
            <Route path="/review/:id" element={<ReviewDetailPage />} />
            <Route path="/entities" element={<EntitiesPage />} />
            <Route path="/entities/:id" element={<EntityDetailPage />} />
            <Route path="/sources" element={<SourcesPage />} />
            <Route path="/projects" element={<ProjectsPage />} />
            <Route path="/projects/:key" element={<ProjectDetailPage />} />
            <Route path="/settings" element={<SettingsPage />} />
            <Route path="/" element={<Navigate to="/memories" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
