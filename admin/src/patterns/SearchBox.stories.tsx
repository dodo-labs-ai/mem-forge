import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState, type ComponentProps } from "react";
import { SearchBox } from "./SearchBox";

const meta = {
  component: SearchBox,
  args: { value: "", label: "Search entities", onCommit: () => {} },
} satisfies Meta<typeof SearchBox>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {};

function CommittedSearch(props: ComponentProps<typeof SearchBox>) {
  const [value, setValue] = useState(props.value);
  return (
    <div className="space-y-2">
      <SearchBox {...props} value={value} onCommit={setValue} />
      <p className="text-sm text-muted-foreground">Committed: “{value}”</p>
    </div>
  );
}

export const Interactive: Story = { render: (args) => <CommittedSearch {...args} /> };
