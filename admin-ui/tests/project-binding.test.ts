import assert from "node:assert/strict";

import { projectBindingIsComplete } from "../src/views/sources/projectBinding.js";

assert.equal(
  projectBindingIsComplete(null),
  false,
  "source creation requires an explicit project binding",
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
