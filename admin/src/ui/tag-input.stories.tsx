import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState } from "react";
import { TagInput } from "./tag-input";

const meta = {
  component: TagInput,
  args: { value: ["ENG", "OPS"], onValueChange: () => {}, "aria-label": "Spaces" },
} satisfies Meta<typeof TagInput>;
export default meta;
type Story = StoryObj<typeof meta>;

function Controlled({ initial }: { initial: string[] }) {
  const [value, setValue] = useState(initial);
  return (
    <div className="w-96">
      <TagInput aria-label="Spaces" value={value} onValueChange={setValue} placeholder="Space keys, such as ENG" />
    </div>
  );
}

export const WithTags: Story = { render: () => <Controlled initial={["ENG", "OPS"]} /> };

export const Empty: Story = { render: () => <Controlled initial={[]} /> };
