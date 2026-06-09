import assert from "node:assert/strict";

import { projectBindingIsComplete } from "../src/views/sources/projectBinding.js";

assert.equal(
  projectBindingIsComplete(null),
  true,
  "source creation can intentionally leave a source unmapped",
);

assert.equal(
  projectBindingIsComplete({ mode: "fixed", project_key: "" }),
  false,
  "fixed bindings require a project",
);

assert.equal(
  projectBindingIsComplete({ mode: "fixed", project_key: "PAY" }),
  true,
  "fixed bindings with a project are complete",
);
