import { describe, expect, it } from "vitest";

import { formatSalary, timeAgo } from "./format.js";

describe("formatSalary", () => {
  it("formats ranges and single values in LPA", () => {
    expect(formatSalary(1_200_000, 1_800_000)).toBe("12–18 LPA");
    expect(formatSalary(null, 650_000)).toBe("6.5 LPA");
    expect(formatSalary(900_000, 900_000)).toBe("9 LPA");
    expect(formatSalary(null, null)).toBeNull();
  });
});

describe("timeAgo", () => {
  const now = new Date("2026-11-24T12:00:00Z");
  it("describes recent dates", () => {
    expect(timeAgo("2026-11-24T08:00:00Z", now)).toBe("today");
    expect(timeAgo("2026-11-23T08:00:00Z", now)).toBe("yesterday");
    expect(timeAgo("2026-11-19T12:00:00Z", now)).toBe("5 days ago");
    expect(timeAgo("2026-10-27T12:00:00Z", now)).toBe("4 weeks ago");
    expect(timeAgo(null, now)).toBeNull();
    expect(timeAgo("not a date", now)).toBeNull();
  });
});
