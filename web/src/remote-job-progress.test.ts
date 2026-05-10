import { describe, expect, test } from "bun:test";
import { parseRemoteJobProgressFromLog } from "./remote-job-progress";

describe("parseRemoteJobProgressFromLog", () => {
  test("extracts live tile suggestions and summary counts from the remote runner log", () => {
    const progress = parseRemoteJobProgressFromLog(
      `
Preparing remote scan job
[   1/42] zurich-center:r12:c15 -> no_crosswalk (0.296)
[   2/42] zurich-center:r12:c16 -> crosswalk (0.234)
[   3/42] zurich-center:r13:c13 -> crosswalk (0.462)
Scene zurich-center: 42 tiles, 2 crosswalk, 1 no_crosswalk
      `,
      "server-side hybrid scan",
    );

    expect(progress.scannedTileIds).toEqual([
      "zurich-center:r12:c15",
      "zurich-center:r12:c16",
      "zurich-center:r13:c13",
    ]);
    expect(progress.results["zurich-center:r12:c15"]?.label).toBe("no_crosswalk");
    expect(progress.results["zurich-center:r12:c16"]?.label).toBe("crosswalk");
    expect(progress.results["zurich-center:r13:c13"]?.score).toBe(0.462);
    expect(progress.summary).toEqual({
      total: 3,
      crosswalk: 2,
      no_crosswalk: 1,
    });
  });
});
