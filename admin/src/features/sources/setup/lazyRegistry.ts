import { createElement, lazy, type ComponentType, type LazyExoticComponent, type ReactElement } from "react";

/** Loads one component on first use, so its code joins the bundle only when a form draws it. */
export type ComponentLoader<P> = () => Promise<{ default: ComponentType<P> }>;

/**
 * Draws the lazily loaded component a form description names. A name with
 * no entry, or no name, returns `null`, and the caller draws its fallback;
 * the caller also wraps the element in `Suspense`. Each name always draws
 * the same component, so React keeps its state between renders.
 */
export function lazyRegistry<P extends object>(
  loaders: Readonly<Record<string, ComponentLoader<P>>>,
): (name: string | null | undefined, props: P) => ReactElement | null {
  const components = new Map<string, LazyExoticComponent<ComponentType<P>>>();
  return (name, props) => {
    if (!name || !Object.hasOwn(loaders, name)) return null;
    let component = components.get(name);
    if (!component) {
      component = lazy(loaders[name]!);
      components.set(name, component);
    }
    return createElement(component, props);
  };
}
