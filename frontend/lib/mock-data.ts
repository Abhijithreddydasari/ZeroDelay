import type { Procedure } from "./types";

// Adapted from data/procedures/01-eva-prep-emu-airlock.md (EVA-PREP-001).
// Legacy demo fixture; the live step overlay now uses backend session data.
export const activeProcedure: Procedure = {
  id: "EVA-PREP-001",
  title: "EMU Suit Checkout and Airlock Egress",
  system: "EMU / Quest Airlock",
  summary:
    "Don and check out the EMU spacesuit, verify comms and pre-breathe, then depressurize the airlock and egress for a spacewalk.",
  steps: [
    {
      id: 1,
      title: "Power on EMU and verify battery",
      safetyTier: "routine",
      instruction:
        "Power on the EMU at the DCM. Confirm the battery state of charge is above 40 percent.",
    },
    {
      id: 2,
      title: "Verify comms link",
      safetyTier: "routine",
      instruction:
        "Establish and confirm two-way comms. Say \"comm check\" and confirm you hear the loopback tone.",
    },
    {
      id: 3,
      title: "Regulator and O2 concentration check",
      safetyTier: "caution",
      instruction:
        "Open the O2 actuator. Confirm suit oxygen concentration reads in the nominal band and pressure is holding at operating pressure.",
      warnings: [
        "Low O2 concentration means unsafe breathing gas. Do not continue below 95 percent.",
      ],
    },
    {
      id: 4,
      title: "CO2 scrubber baseline",
      safetyTier: "caution",
      instruction:
        "Confirm suit CO2 partial pressure is in the nominal band before you seal up for pre-breathe.",
      warnings: [
        "A high CO2 baseline before egress will only get worse under workload.",
      ],
    },
    {
      id: 5,
      title: "Pre-breathe protocol",
      safetyTier: "caution",
      instruction:
        "Begin the pure-O2 pre-breathe. Hold for the specified duration to purge nitrogen and prevent decompression sickness.",
      warnings: [
        "Skipping or shortening pre-breathe risks decompression sickness (the bends).",
      ],
    },
    {
      id: 6,
      title: "Clip safety tether",
      safetyTier: "critical",
      instruction:
        "Clip the safety tether to the structural hard point. Verify the hook is locked and load-tested.",
      warnings: [
        "An unclipped or unlocked tether means no fall protection outside. This is critical.",
      ],
      diagram: "/diagrams/emu-suit-callouts.png",
    },
    {
      id: 7,
      title: "Depressurize airlock below 2 psia BEFORE hatch open",
      safetyTier: "critical",
      instruction:
        "Depressurize the airlock. Do NOT open the hatch until pressure is confirmed below 2.0 psia. Read the gauge and cross-check against the live sensor.",
      warnings: [
        "Opening the hatch above 2 psia risks explosive decompression. This is critical.",
        "If the gauge and the sensor disagree, trust the lower-risk assumption and stop.",
      ],
      diagram: "/diagrams/airlock-valves.png",
    },
    {
      id: 8,
      title: "Open hatch and egress",
      safetyTier: "critical",
      instruction:
        "With pressure confirmed below 2 psia and tether locked, open the hatch and egress feet-first.",
      warnings: ["Re-confirm suit pressure is holding immediately after egress."],
    },
  ],
};
