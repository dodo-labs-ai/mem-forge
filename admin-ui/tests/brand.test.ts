import assert from "node:assert/strict";

import { BRAND_INITIALS, BRAND_NAME, BRAND_SUBTITLE, TAI_SEAL_LOGO_TITLE } from "../src/brand.js";

assert.equal(BRAND_NAME, "MemInception");
assert.equal(BRAND_INITIALS, "MI");
assert.equal(BRAND_SUBTITLE, "Agent memory admin");
assert.equal(TAI_SEAL_LOGO_TITLE, "MemInception logo");
