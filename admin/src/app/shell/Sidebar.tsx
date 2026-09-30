import { NavLink } from "react-router-dom";
import { cn } from "@/lib/cn";
import { Separator } from "@/ui/separator";
import { LocalSyncStatus } from "@/features/local-sync";
import { BrandMark } from "../brand/BrandMark";
import { reservedRedirect, type ExtensionNavItem } from "../extension/contract";
import { useExtension } from "../extension/ExtensionProvider";
import { NAV_GROUPS, SETTINGS_NAV_ITEM, type NavItem } from "../navigation";

const itemClass =
  "flex h-8 items-center gap-2.5 rounded-md px-2.5 text-sm text-sidebar-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground [&_svg]:size-4 [&_svg]:shrink-0 [&_svg]:text-muted-foreground";
const activeClass = "bg-sidebar-accent font-medium text-sidebar-accent-foreground [&_svg]:text-sidebar-accent-foreground";

function ProductNavLink({ item }: { item: NavItem }) {
  const extension = useExtension();
  const Icon = item.icon;
  const redirect = reservedRedirect(extension, item.segment);
  if (item.v1Path !== undefined && redirect === undefined) {
    return (
      <a href={item.v1Path} className={itemClass}>
        <Icon />
        <span className="flex-1 truncate">{item.label}</span>
      </a>
    );
  }
  return (
    <NavLink to={redirect ?? `/${item.segment}`} className={({ isActive }) => cn(itemClass, isActive && activeClass)}>
      <Icon />
      <span className="flex-1 truncate">{item.label}</span>
    </NavLink>
  );
}

function ExtensionNavLink({ item }: { item: ExtensionNavItem }) {
  const Icon = item.icon;
  return (
    <NavLink to={item.to} className={({ isActive }) => cn(itemClass, isActive && activeClass)}>
      {Icon ? <Icon /> : null}
      <span className="flex-1 truncate">{item.label}</span>
    </NavLink>
  );
}

function GroupTitle({ children }: { children: string }) {
  return (
    <div className="px-2.5 pb-1.5 text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
      {children}
    </div>
  );
}

function groupExtensionItems(items: ExtensionNavItem[]): Array<[string, ExtensionNavItem[]]> {
  const groups = new Map<string, ExtensionNavItem[]>();
  for (const item of items) groups.set(item.group, [...(groups.get(item.group) ?? []), item]);
  return [...groups.entries()];
}

export function Sidebar() {
  const extension = useExtension();
  const extensionGroups = groupExtensionItems(
    (extension?.navItems ?? []).filter((item) => item.visibleWhen?.() ?? true),
  );

  return (
    <aside className="flex h-full w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar">
      <div className="flex h-14 items-center gap-2.5 px-4">
        <BrandMark className="size-7" />
        <span className="text-[15px] font-semibold tracking-tight text-foreground">MemForge</span>
      </div>
      <nav aria-label="Main" className="flex flex-1 flex-col gap-5 overflow-y-auto px-2.5 py-3">
        {NAV_GROUPS.map((group) => (
          <div key={group.title} className="flex flex-col gap-px">
            <GroupTitle>{group.title}</GroupTitle>
            {group.items.map((item) => (
              <ProductNavLink key={item.segment} item={item} />
            ))}
          </div>
        ))}
        <div className="mt-auto flex flex-col gap-px">
          <ProductNavLink item={SETTINGS_NAV_ITEM} />
        </div>
        {extensionGroups.length > 0 ? (
          <>
            <Separator className="bg-sidebar-border" />
            {extensionGroups.map(([title, items]) => (
              <div key={title} className="flex flex-col gap-px">
                <GroupTitle>{title}</GroupTitle>
                {items.map((item) => (
                  <ExtensionNavLink key={item.to} item={item} />
                ))}
              </div>
            ))}
          </>
        ) : null}
      </nav>
      <div className="border-t border-sidebar-border p-3">
        <LocalSyncStatus />
      </div>
    </aside>
  );
}
