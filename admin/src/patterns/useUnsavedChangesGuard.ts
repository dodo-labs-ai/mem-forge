import { useEffect } from "react";
import { useBlocker } from "react-router-dom";

/**
 * Holds navigation away from the page while it has unsaved changes. Links
 * inside the app are held by the returned blocker, which the page confirms
 * or resets; reloading, closing the tab and links that leave the app get the
 * browser's own prompt.
 */
export function useUnsavedChangesGuard(hasUnsavedChanges: boolean) {
  const blocker = useBlocker(
    ({ currentLocation, nextLocation }) => hasUnsavedChanges && currentLocation.pathname !== nextLocation.pathname,
  );

  useEffect(() => {
    if (!hasUnsavedChanges) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [hasUnsavedChanges]);

  return blocker;
}
