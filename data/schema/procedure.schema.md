# Procedure Step Schema

Every file in `procedures/` is a Markdown document with a **YAML front-matter block**
(between `---` fences) that the agent parses into a typed state machine, followed by
human-readable prose for each step.

The prose is for humans and for embedding/retrieval context. The **YAML is the source of
truth** the agent executes.

## Top-level fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `procedure_id` | string | yes | Stable unique id, e.g. `EVA-PREP-001`. |
| `title` | string | yes | Human title. |
| `domain` | string | yes | `space` for now; generalizable (`mining`, `offshore`, `marine`). |
| `system` | string | yes | Hardware/system the procedure operates on. |
| `summary` | string | yes | One-line description used for retrieval. |
| `estimated_minutes` | int | no | Nominal duration. |
| `required_tools` | list[string] | no | Tool `part_id`s needed (checked against inventory). |
| `required_parts` | list[string] | no | Consumable/spare `part_id`s needed. |
| `sensors_watched` | list[string] | no | Sensor names monitored for the whole procedure. |
| `related_diagrams` | list[string] | no | Diagram ids used for visual verification. |
| `entry_conditions` | list[precondition] | no | Must hold before the procedure can start. |
| `steps` | list[step] | yes | Ordered steps (see below). |

## Step fields

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | int | yes | Step number; enforces order. |
| `title` | string | yes | Short step name. |
| `safety_tier` | enum | yes | `routine` \| `caution` \| `critical`. Drives confirmation strength. |
| `instruction` | string | yes | What the agent speaks to the technician. |
| `preconditions` | list[precondition] | no | Conditions gated before the step is presented. |
| `specs` | object | no | Exact values (`torque`, `pressure`, `duration_s`, etc.). |
| `required_parts` | list[string] | no | Parts consumed by this step (triggers replan if missing). |
| `verify` | object | yes | How the step is confirmed (see Verify). |
| `on_failure` | object | no | What to do if verify/preconditions fail (see On failure). |
| `warnings` | list[string] | no | Spoken warnings; mandatory for `caution`/`critical`. |
| `risk_if_skipped` | string | no | Logged silently; surfaced if step is skipped/out-of-order. |
| `branches` | list[branch] | no | Symptom-driven diagnostic branching. |

## Precondition object

Resolved by calling a reference table as a tool.

```yaml
- sensor: airlock_pressure_psia   # name from telemetry-nominal-ranges.yaml
  check: less_than                # equals | not_equals | less_than | greater_than |
                                  # in_range | out_of_range | present | absent
  value: 2.0
# OR inventory-based:
- part: QD-JMP-14                 # part_id from inventory.yaml
  check: available
  qty: 1
```

## Verify object

```yaml
verify:
  method: sensor            # sensor | visual | verbal | manual
  # sensor:
  sensor: suit_pressure_psia
  check: in_range
  # visual:
  visual_ref: airlock-valves    # diagram id; image at diagrams/<id>.png
  expect_state: "gauge reads < 2 psia; equalization valve handle vertical"
  # verbal:
  prompt: "Say 'confirm' when the tether is clipped to the structural hard point."
```

## On failure object

```yaml
on_failure:
  action: block          # block | branch | escalate | emergency | repeat
  goto_step: 7           # for action: branch
  goto_procedure: CDRA-FAULT-004  # for cross-procedure branch
  note: "Do NOT open hatch. Repressurize and recheck seal."
```

## Branch object (dynamic diagnostic branching)

```yaml
branches:
  - symptom: "white frost around QD"
    goto_step: 12
  - symptom: "white flakes / snow (ammonia release)"
    action: emergency
    note: "Ammonia release. Execute decon before ingress."
  - symptom: "no visible leak but pressure still dropping"
    goto_procedure: COOL-JMP-002
    goto_step: 20
```

## Safety tier behavior contract

| Tier | Agent behavior |
| --- | --- |
| `routine` | State the step, single confirmation to advance. |
| `caution` | Speak all `warnings`, require explicit "confirm" before advancing. |
| `critical` | Speak all `warnings` slowly, require read-back or explicit confirmation, re-check preconditions immediately before advancing; never auto-advance. |
