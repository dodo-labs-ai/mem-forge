import type { SourceRow } from "./model/sourceRows";
import { SourceNameCell } from "./SourceCells";
import { SourceRowActions } from "./SourceRowActions";
import { useSourceTable } from "./sourceTableContext";

export function NameCell({ row }: { row: SourceRow }) {
  const { typeLabels } = useSourceTable();
  return <SourceNameCell row={row} typeLabel={typeLabels[row.source.type] ?? row.source.type} />;
}

export function ActionsCell({ row }: { row: SourceRow }) {
  const { actions, onViewDetails, onDelete } = useSourceTable();
  return <SourceRowActions row={row} actions={actions} onViewDetails={onViewDetails} onDelete={onDelete} />;
}
