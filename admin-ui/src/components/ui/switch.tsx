import type * as React from "react"
import { Switch as SwitchPrimitive } from "@base-ui/react/switch"

import { cn } from "@/lib/utils"

function Switch({
  className,
  children,
  ...props
}: SwitchPrimitive.Root.Props & { children?: React.ReactNode }) {
  return (
    <SwitchPrimitive.Root
      data-slot="switch"
      className={cn(
        "group/switch inline-flex h-4 w-8 shrink-0 cursor-pointer items-center rounded-full border border-border bg-muted p-0.5 transition-colors outline-none",
        "data-[checked]:border-emerald-600 data-[checked]:bg-emerald-600",
        "focus-visible:ring-3 focus-visible:ring-ring/50",
        "data-[disabled]:cursor-not-allowed data-[disabled]:opacity-60",
        className
      )}
      {...props}
    >
      <SwitchPrimitive.Thumb
        data-slot="switch-thumb"
        className="block size-3 rounded-full bg-background shadow-sm transition-transform group-data-[checked]/switch:translate-x-4"
      />
      {children}
    </SwitchPrimitive.Root>
  )
}

export { Switch }
