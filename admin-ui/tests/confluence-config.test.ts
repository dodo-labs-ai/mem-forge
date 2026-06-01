import assert from "node:assert/strict";

import {
  applyConfluenceUrlInference,
  confluenceSyncMode,
  isConfluenceFieldRequired,
  isConfluenceFieldVisible,
  parseConfluenceWikiUrl,
} from "../src/views/sources/confluenceConfig.js";

const corporatePage = parseConfluenceWikiUrl(
  "https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll",
);

assert.deepEqual(corporatePage, {
  normalizedBaseUrl: "https://wiki.company.example",
  apiPrefix: "/wiki",
  spaceKey: "PAY",
  pageId: "5695886009",
});

assert.deepEqual(
  applyConfluenceUrlInference({
    base_url: "https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll",
  }),
  {
    base_url: "https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll",
    spaces: ["PAY"],
    page_tree_root: "5695886009",
    sync_mode: "page_tree",
  },
);

assert.equal(confluenceSyncMode({ page_tree_root: "5695886009" }), "page_tree");
assert.equal(confluenceSyncMode({ spaces: ["PAY"] }), "space");

assert.equal(isConfluenceFieldVisible("spaces", { sync_mode: "page_tree" }), false);
assert.equal(isConfluenceFieldVisible("page_tree_root", { sync_mode: "page_tree" }), true);
assert.equal(isConfluenceFieldRequired("spaces", { sync_mode: "space" }), true);
assert.equal(isConfluenceFieldRequired("page_tree_root", { sync_mode: "page_tree" }), true);
