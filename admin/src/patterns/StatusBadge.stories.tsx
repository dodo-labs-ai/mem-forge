import type { Meta, StoryObj } from "@storybook/react-vite";
import { StatusBadge } from "./StatusBadge";

const meta = { component: StatusBadge, args: { tone: "ok", children: "Up to date" } } satisfies Meta<typeof StatusBadge>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Tones: Story = {
  render: () => (
    <div className="flex flex-wrap gap-2">
      <StatusBadge tone="ok">Up to date</StatusBadge>
      <StatusBadge tone="live">Syncing now</StatusBadge>
      <StatusBadge tone="warn">Scheduled sync is overdue</StatusBadge>
      <StatusBadge tone="danger">Sign-in required</StatusBadge>
      <StatusBadge tone="idle">Paused</StatusBadge>
      <StatusBadge tone="ok" variant="dot">Local sync online</StatusBadge>
    </div>
  ),
};
