/**
 * Public path of the admin UI. It is `/v2/` while the V1 admin UI still owns
 * `/`, and becomes `/` when V1 is removed. Vite's `base` and the
 * router `basename` both read this constant.
 */
export const APP_BASE_PATH = "/v2/";
