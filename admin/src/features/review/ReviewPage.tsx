import { Route, Routes } from "react-router-dom";
import { ReviewDetailPage } from "./ReviewDetailPage";
import { ReviewQueuePage } from "./ReviewQueuePage";

/** The Review area: the queue at `/review` and one review at `/review/:reviewId`. */
export function ReviewPage() {
  return (
    <Routes>
      <Route index element={<ReviewQueuePage />} />
      <Route path=":reviewId" element={<ReviewDetailPage />} />
    </Routes>
  );
}
