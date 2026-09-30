import type { Meta, StoryObj } from "@storybook/react-vite";
import { BackLink } from "./BackLink";

const meta = { component: BackLink, args: { label: "Memories", render: <a href="#memories" /> } } satisfies Meta<typeof BackLink>;
export default meta;
type Story = StoryObj<typeof meta>;

export const ToList: Story = {};
