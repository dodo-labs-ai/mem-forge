type NamedSourceType = {
  name: string;
};

const MANAGED_SOURCE_TYPES = new Set(["agent_session"]);

export function isManagedSourceType(sourceType: string): boolean {
  return MANAGED_SOURCE_TYPES.has(sourceType);
}

export function canConfigureSourceType(sourceType: string): boolean {
  return !isManagedSourceType(sourceType);
}

export function canDeleteSourceType(sourceType: string): boolean {
  return !isManagedSourceType(sourceType);
}

export function userConfigurableGenes<T extends NamedSourceType>(genes: readonly T[]): T[] {
  return genes.filter((gene) => canConfigureSourceType(gene.name));
}
