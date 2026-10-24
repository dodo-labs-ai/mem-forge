import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithApi } from "@/test/renderWithApi";
import { setupJobRoutes } from "@/test/setupJobs";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import type { FieldValue } from "../model/configForm";
import GitHubRepoPathsPicker from "./GitHubRepoPathsPicker";

afterEach(() => {
  vi.unstubAllGlobals();
});

const SCHEMA = sourceTypeSchema("github_repo");
const TREE = {
  source_type: "github_repo",
  ref: "main",
  count: 4,
  truncated: false,
  items: [
    { path: "docs", type: "tree" },
    { path: "docs/archive", type: "tree" },
    { path: "docs/archive/old.md", type: "blob", size: 10 },
    { path: "docs/guide.md", type: "blob", size: 10 },
    { path: "README.md", type: "blob", size: 10 },
  ],
};

function Picker({
  fieldKey,
  config,
  onChange,
  sourceId,
}: {
  fieldKey: "include_paths" | "exclude_paths";
  config: Record<string, unknown>;
  onChange: (value: FieldValue) => void;
  sourceId?: string;
}) {
  const field = SCHEMA.fields.find((candidate) => candidate.key === fieldKey)!;
  const [value, setValue] = useState<FieldValue>([]);
  return (
    <GitHubRepoPathsPicker
      field={field}
      id={`config-${fieldKey}`}
      labelId={`config-${fieldKey}-label`}
      choices={[]}
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
      sourceType="github_repo"
      sourceId={sourceId}
      config={{ ...config, [fieldKey]: value }}
    />
  );
}

test("browsing lists the repository through the server and ticking a folder includes it", async () => {
  const onChange = vi.fn();
  const { calls } = renderWithApi(
    <Picker fieldKey="include_paths" sourceId="src-repo" config={{ repo_url: "https://github.com/acme/docs" }} onChange={onChange} />,
    { "POST /api/v1/genes/github_repo/browse-tree": () => TREE },
  );

  await userEvent.click(screen.getByRole("button", { name: "Browse repository" }));
  const tree = await screen.findByRole("tree", { name: "Repository paths" });
  await userEvent.click(within(tree).getByRole("checkbox", { name: "docs" }));

  expect(onChange).toHaveBeenLastCalledWith(["docs"]);
  expect(within(tree).getByRole("checkbox", { name: "docs/guide.md" })).toHaveAttribute("aria-disabled", "true");
  const browse = calls.find((request) => request.method === "POST")!;
  expect(await browse.clone().json()).toEqual({ config: { repo_url: "https://github.com/acme/docs", include_paths: [] }, limit: 2000, source_id: "src-repo" });
});

test("the exclude picker suggests outdated folders and counts the files left to sync", async () => {
  const onChange = vi.fn();
  renderWithApi(
    <Picker fieldKey="exclude_paths" config={{ repo_url: "https://github.com/acme/docs" }} onChange={onChange} />,
    { "POST /api/v1/genes/github_repo/browse-tree": () => TREE },
  );

  await userEvent.click(screen.getByRole("button", { name: "Browse repository" }));
  expect(await screen.findByText("3 files ready to sync")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "+ docs/archive" }));

  expect(onChange).toHaveBeenLastCalledWith(["docs/archive"]);
  expect(screen.getByText("2 files ready to sync")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "+ docs/archive" })).not.toBeInTheDocument();
});

test("local sync lists the remote repository through the daemon with a job that carries no secret or path", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  const config = {
    connection_mode: "local_push",
    repo_url: "https://github.tools.example/acme/docs",
    ref: "dev",
    pat: "ghp_typed-under-cloud-access",
    exclude_paths: ["docs/archive"],
    include_extensions: ["md"],
  };
  renderWithApi(
    <Picker fieldKey="include_paths" config={config} onChange={vi.fn()} />,
    setupJobRoutes(
      {
        github_repo_preview_tree: {
          status: "succeeded",
          result: {
            operation: "github_repo_preview_tree",
            counts: { included: 2, ignored: 0 },
            items: [{ relative_path: "docs/guide.md" }, { relative_path: "README.md" }],
          },
        },
      },
      created,
    ),
  );

  await userEvent.click(screen.getByRole("button", { name: "Browse repository" }));
  const tree = await screen.findByRole("tree", { name: "Repository paths" });
  expect(within(tree).getByRole("checkbox", { name: "docs/guide.md" })).toBeInTheDocument();
  expect(created).toEqual([
    {
      source_id: "",
      source_type: "github_repo",
      operation: "github_repo_preview_tree",
      payload: { repo_url: "https://github.tools.example/acme/docs", ref: "dev", include_extensions: ["md"], limit: 2000 },
    },
  ]);
});
