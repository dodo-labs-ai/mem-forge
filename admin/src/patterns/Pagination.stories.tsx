import type { Meta, StoryObj } from "@storybook/react-vite";
import { useState, type ComponentProps } from "react";
import { Pagination } from "./Pagination";

const meta = {
  component: Pagination,
  args: { page: 1, pageSize: 50, total: 3344, itemName: "memory", itemNamePlural: "memories", onPageChange: () => {} },
} satisfies Meta<typeof Pagination>;
export default meta;
type Story = StoryObj<typeof meta>;

export const FirstPage: Story = {};

function PagedList(props: ComponentProps<typeof Pagination>) {
  const [page, setPage] = useState(props.page);
  return <Pagination {...props} page={page} onPageChange={setPage} />;
}

export const Interactive: Story = { render: (args) => <PagedList {...args} /> };

export const SinglePage: Story = { args: { total: 4, itemName: "relation", itemNamePlural: undefined } };
