import type { ComponentType } from "react";
import { Navigate, type RouteObject } from "react-router-dom";
import { extensionRoutes, reservedRedirect, type AdminExtension, type ReservedRouteSegment } from "./extension/contract";
import { HOME_SEGMENT } from "./navigation";
import { AppShell } from "./shell/AppShell";
import { NotFoundPage } from "./shell/NotFoundPage";

interface ProductRoute {
  segment: ReservedRouteSegment;
  /** Loads the page on first visit, so each page ships in its own chunk. */
  load: () => Promise<ComponentType>;
}

const PRODUCT_ROUTES: ProductRoute[] = [
  { segment: "memories", load: () => import("@/features/memories").then((module) => module.MemoriesRoutes) },
  { segment: "review", load: () => import("@/features/review").then((module) => module.ReviewPage) },
  { segment: "sources", load: () => import("@/features/sources").then((module) => module.SourcesPage) },
  { segment: "settings", load: () => import("@/features/settings").then((module) => module.SettingsPage) },
  { segment: "evaluation", load: () => import("@/features/evaluation").then((module) => module.EvaluationPage) },
  { segment: "projects", load: () => import("@/features/projects").then((module) => module.ProjectsRoutes) },
];

/** Product routes first, then extension routes; a reserved segment an extension redirects goes to its page. */
export function buildRoutes(extension: AdminExtension | undefined): RouteObject[] {
  const product: RouteObject[] = PRODUCT_ROUTES.map(({ segment, load }) => {
    const redirect = reservedRedirect(extension, segment);
    if (redirect) return { path: `${segment}/*`, element: <Navigate to={redirect} replace /> };
    return { path: `${segment}/*`, lazy: async () => ({ Component: await load() }) };
  });
  return [
    {
      element: <AppShell />,
      children: [
        { index: true, element: <Navigate to={`/${HOME_SEGMENT}`} replace /> },
        ...product,
        ...extensionRoutes(extension),
        { path: "*", element: <NotFoundPage /> },
      ],
    },
  ];
}
