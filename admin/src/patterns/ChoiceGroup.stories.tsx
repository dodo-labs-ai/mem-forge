import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { ChoiceGroup, type Choice } from "./ChoiceGroup";

const SCOPES: Choice[] = [
  { value: "page_tree", label: "This page tree", description: "A page and the pages under it" },
  { value: "space", label: "Whole space", description: "Every page of the chosen spaces" },
];

const RETENTION: Choice[] = [
  { value: "forever", label: "Forever" },
  { value: "365", label: "1 year" },
  { value: "730", label: "2 years" },
  { value: "1095", label: "3 years" },
  { value: "30", label: "30 days" },
];

const meta = {
  component: ChoiceGroup,
  args: { id: "scope", choices: SCOPES, value: "space", onValueChange: () => {}, "aria-labelledby": "scope-label" },
} satisfies Meta<typeof ChoiceGroup>;
export default meta;
type Story = StoryObj<typeof meta>;

function Controlled({ choices, initial, label }: { choices: Choice[]; initial: string; label: string }) {
  const [value, setValue] = useState(initial);
  return (
    <div className="w-96 space-y-1.5">
      <p id="choice-label" className="text-sm font-medium">
        {label}
      </p>
      <ChoiceGroup id="choice" aria-labelledby="choice-label" choices={choices} value={value} onValueChange={setValue} />
    </div>
  );
}

export const RadioCards: Story = { render: () => <Controlled choices={SCOPES} initial="space" label="Sync scope" /> };

export const ManyChoicesAsSelect: Story = {
  render: () => <Controlled choices={RETENTION} initial="forever" label="Keep messages" />,
};
