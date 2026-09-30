const DARK_QUERY = "(prefers-color-scheme: dark)";

/**
 * Follows the operating system's light or dark setting by toggling the `dark`
 * class that the theme tokens key on. Returns a function that stops following.
 */
export function followSystemColorScheme(root: HTMLElement = document.documentElement): () => void {
  const media = window.matchMedia(DARK_QUERY);
  const apply = () => root.classList.toggle("dark", media.matches);
  apply();
  media.addEventListener("change", apply);
  return () => media.removeEventListener("change", apply);
}
