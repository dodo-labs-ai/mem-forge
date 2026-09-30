import type { Meta, StoryObj } from "@storybook/react-vite";
import { Button } from "@/ui/button";
import { DataTable, dataColumns } from "./DataTable";
import { EmptyState } from "./EmptyState";
import { GroupHeader } from "./GroupHeader";
import { StatusBadge } from "./StatusBadge";

interface Member {
  id: string;
  name: string;
  role: string;
  active: boolean;
}

const column = dataColumns<Member>();
const columns = [
  column.accessor("name", { header: "Name" }),
  column.accessor("role", { header: "Role" }),
  column.display({
    id: "status",
    header: "Status",
    cell: ({ row }) => (
      <StatusBadge tone={row.original.active ? "ok" : "idle"}>{row.original.active ? "Active" : "Invited"}</StatusBadge>
    ),
  }),
  column.display({
    id: "actions",
    header: () => <span className="sr-only">Actions</span>,
    cell: () => (
      <Button size="sm" variant="outline">
        Manage
      </Button>
    ),
    meta: { className: "w-[1%]" },
  }),
];

const admins: Member[] = [{ id: "1", name: "Ada Lovelace", role: "Workspace admin", active: true }];
const members: Member[] = [
  { id: "2", name: "Grace Hopper", role: "Member", active: true },
  { id: "3", name: "Alan Turing", role: "Viewer", active: false },
];

const meta = {
  component: DataTable<Member>,
  args: { "aria-label": "Members", columns, data: [...admins, ...members], getRowId: (row: Member) => row.id },
} satisfies Meta<typeof DataTable<Member>>;
export default meta;
type Story = StoryObj<typeof meta>;

export const Flat: Story = {};

export const Grouped: Story = {
  args: {
    data: [
      { id: "admins", header: <GroupHeader title="Admins" description="Can manage members" meta="1 person" />, rows: admins },
      { id: "members", header: <GroupHeader title="Members" meta="2 people" />, rows: members },
    ],
  },
};

export const Loading: Story = { args: { loading: true } };

export const Empty: Story = {
  args: { data: [], empty: <EmptyState title="No members yet" description="Invite someone to get started." /> },
};
