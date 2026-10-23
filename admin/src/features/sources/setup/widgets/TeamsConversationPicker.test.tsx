import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithApi } from "@/test/renderWithApi";
import { setupJobRoutes, type FakeSetupJob } from "@/test/setupJobs";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import type { FieldValue } from "../model/configForm";
import TeamsConversationPicker from "./TeamsConversationPicker";

afterEach(() => {
  vi.unstubAllGlobals();
});

const FIELD = sourceTypeSchema("teams").fields.find((field) => field.key === "conversation_ids")!;
const SIGNED_IN: FakeSetupJob = {
  status: "succeeded",
  result: { operation: "teams_auth_check", region: "emea", authenticated: true, expires_in_minutes: 50, error: null },
};
const NO_SESSION: FakeSetupJob = {
  status: "failed",
  result: { operation: "teams_auth_check", region: "emea", authenticated: false, expires_in_minutes: null, error: "No Teams session found" },
};
const BROWSED: FakeSetupJob = {
  status: "succeeded",
  result: {
    operation: "teams_browse",
    region: "emea",
    favorites: [],
    teams: [{ id: "t-eng", displayName: "Engineering", channels: [{ id: "19:arch@thread.tacv2", displayName: "Architecture" }] }],
    group_chats: [{ id: "19:planning@thread.v2", topic: "Planning chat", lastActivity: null }],
    individual_chats: [{ id: "19:ada@unq.gbl.spaces", topic: "Ada Lovelace", lastActivity: null }],
    error: null,
  },
};

function Picker({ initial, onChange }: { initial: string[]; onChange: (value: FieldValue) => void }) {
  const [value, setValue] = useState<FieldValue>(initial);
  return (
    <TeamsConversationPicker
      field={FIELD}
      id="config-conversation_ids"
      labelId="config-conversation_ids-label"
      choices={[]}
      value={value}
      onChange={(next) => {
        setValue(next);
        onChange(next);
      }}
      sourceType="teams"
      config={{ region: "emea" }}
    />
  );
}

test("lists the conversations the Teams session can read and selects one", async () => {
  const onChange = vi.fn();
  renderWithApi(<Picker initial={[]} onChange={onChange} />, setupJobRoutes({ teams_auth_check: SIGNED_IN, teams_browse: BROWSED }));

  const list = await screen.findByRole("list", { name: "Teams conversations" });
  expect(within(list).getByText("Architecture")).toBeInTheDocument();
  await userEvent.type(screen.getByRole("textbox", { name: "Search conversations" }), "plan");
  expect(within(list).queryByText("Architecture")).not.toBeInTheDocument();
  await userEvent.click(within(list).getByRole("button", { name: /Planning chat/ }));

  expect(onChange).toHaveBeenLastCalledWith(["19:planning@thread.v2"]);
  expect(within(screen.getByRole("list", { name: "Selected conversations" })).getByText("Planning chat")).toBeInTheDocument();
});

test("a stored conversation shows as existing until the list names it", async () => {
  let resolve!: () => void;
  const listed = new Promise<void>((done) => (resolve = done));
  const routes = setupJobRoutes({ teams_auth_check: SIGNED_IN, teams_browse: BROWSED });
  const read = routes["GET /api/cloud/local-agent/jobs/laj-teams_browse"]!;
  routes["GET /api/cloud/local-agent/jobs/laj-teams_browse"] = async (request) => {
    await listed;
    return read(request);
  };
  renderWithApi(<Picker initial={["19:arch@thread.tacv2"]} onChange={vi.fn()} />, routes);

  const selected = screen.getByRole("list", { name: "Selected conversations" });
  expect(within(selected).getByText(/^Existing conversation · /)).toBeInTheDocument();
  resolve();
  expect(await within(selected).findByText("Engineering / Architecture")).toBeInTheDocument();
});

test("without a Teams session it asks to connect and keeps what is selected", async () => {
  const onChange = vi.fn();
  renderWithApi(<Picker initial={["19:arch@thread.tacv2"]} onChange={onChange} />, setupJobRoutes({ teams_auth_check: NO_SESSION }));

  expect(await screen.findByText(/Connect the Teams session in the Teams session step/)).toBeInTheDocument();
  expect(screen.queryByRole("list", { name: "Teams conversations" })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: /^Remove Existing conversation/ }));
  expect(onChange).toHaveBeenLastCalledWith([]);
});
