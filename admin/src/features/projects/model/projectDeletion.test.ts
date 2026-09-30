import { expect, test } from "vitest";
import { deleteProjectDescription, deletedNoticeFrom, deletedProjectNotice } from "./projectDeletion";

test("states where the memories go and which sources stop writing to the project", () => {
  expect(deleteProjectDescription(708, 2)).toBe(
    "708 memories move to the Unsorted project. 2 sources will stop writing to this project. This cannot be undone.",
  );
  expect(deleteProjectDescription(1, 1)).toBe(
    "1 memory moves to the Unsorted project. 1 source will stop writing to this project. This cannot be undone.",
  );
  expect(deleteProjectDescription(0, 0)).toBe("The project has no memories. This cannot be undone.");
});

test("reports how many memories the server moved and sources it released", () => {
  expect(deletedProjectNotice("Payroll archive", 12, 2)).toBe(
    "Project “Payroll archive” deleted. 12 memories moved to the Unsorted project. 2 sources stopped writing to this project.",
  );
  expect(deletedProjectNotice("Payroll archive", 1, 1)).toBe(
    "Project “Payroll archive” deleted. 1 memory moved to the Unsorted project. 1 source stopped writing to this project.",
  );
  expect(deletedProjectNotice("Payroll archive", 0, 0)).toBe(
    "Project “Payroll archive” deleted. 0 memories moved to the Unsorted project.",
  );
});

test("reads the notice a delete on the detail page hands to the list", () => {
  expect(deletedNoticeFrom({ deletedNotice: "Project “Pay” deleted." })).toBe("Project “Pay” deleted.");
  expect(deletedNoticeFrom(null)).toBeNull();
  expect(deletedNoticeFrom({ other: true })).toBeNull();
});
