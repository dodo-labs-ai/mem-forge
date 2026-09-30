# MemForge admin UI

The admin UI that replaces `admin-ui/` (the V1 admin UI). Both run side by side
until V1 is removed; see [ADR 0044](../docs/adr/0044-build-the-admin-ui-v2-beside-v1-and-remove-v1-after-a-parallel-run.md).

## Run

```bash
uv run memforge api          # the Admin API on :8765
cd admin && npm ci && npm run dev
```

Open http://localhost:5175/v2/. Pages not built here yet open in V1, so run
`npm run dev` in `admin-ui/` too if you follow those links.

| Command | What it does |
|---|---|
| `npm run lint` | ESLint, including the layer import rules |
| `npm test` | Vitest unit and component tests |
| `npm run test:e2e` | Playwright smoke tests against the production build, API stubbed |
| `npm run storybook` | The catalogue of `ui` and `patterns` |
| `npm run gen:api` | Regenerate API types after a backend change |

## Layers

```
src/styles     theme tokens (tokens.css) and global CSS
src/ui         primitives from shadcn/ui on Base UI
src/patterns   product-neutral compositions: DataTable, FilterBar, DetailDrawer, ...
src/features   one folder per product area, public API in index.ts
src/app        shell, router, extension contract
src/api        generated schema, typed client, workspace target
src/lib        pure helpers
```

A layer imports only from the layers below it, and a feature imports another
feature only through its `index.ts`. ESLint enforces both.

## Adding a page

1. Create `src/features/<area>/` with `api.ts` (query hooks), `model/` (pure
   logic and its tests) and the page component, and export the page from
   `index.ts`.
2. Add it to `PRODUCT_ROUTES` in `src/app/router.tsx` and remove the item's
   `v1Path` in `src/app/navigation.ts`.
3. If an endpoint has no response model, declare its response in
   `src/api/responses.ts` and register it in `src/api/paths.ts`.

Use colors from `tokens.css` through Tailwind utilities (`bg-surface`,
`text-tone-danger-foreground`) and never hard-code palette values. Show
related values as labeled properties (`PropertyList`), not joined into one
string.
