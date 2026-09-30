import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";
import type { EvaluationParams } from "./model/evaluationParams";

type OverviewScope = Pick<EvaluationParams, "days" | "sourceId" | "sourceType">;

export const evaluationKeys = {
  all: ["evaluation"] as const,
  overview: ({ days, sourceId, sourceType }: OverviewScope) => ["evaluation", "overview", days, sourceId, sourceType] as const,
};

/** The workspace evaluation for one window and scope. The previous scope stays on screen while the next loads. */
export function useEvaluationOverview(scope: OverviewScope) {
  const api = useApi();
  return useQuery({
    queryKey: evaluationKeys.overview(scope),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/agent-evaluations/online-overview", {
          params: { query: { days: scope.days, source_id: scope.sourceId, source_type: scope.sourceType } },
        }),
      ),
    placeholderData: keepPreviousData,
  });
}
