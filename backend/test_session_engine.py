"""State-machine safety checks that do not load the large models."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .agent.session_engine import SessionStore, fast_decision, validate_corpus
from .tools.sensor_sim import get_simulator


class SessionEngineTests(unittest.TestCase):
    def setUp(self):
        get_simulator().reset()
        self.temp = tempfile.TemporaryDirectory()
        self.store = SessionStore(Path(self.temp.name) / "sessions.sqlite")
        self.state = self.store.create()

    def tearDown(self):
        get_simulator().reset()
        self.temp.cleanup()

    def select(self, name: str):
        suggestion = fast_decision(self.state, f"select {name} procedure")
        self.assertEqual(suggestion.action, "clarify")
        first = fast_decision(self.state, "confirm procedure")
        self.assertEqual(first.step_id, self.state["step_id"])
        self.assertTrue(self.state["pending_confirmation"])

    def test_corpus_targets_and_noncontiguous_steps(self):
        validate_corpus()
        self.select("cdra")
        self.assertEqual(self.state["step_id"], 1)
        get_simulator().inject("cabin_co2_mmHg", 4.0)
        self.assertEqual(fast_decision(self.state, "acknowledged").step_id, 2)
        self.assertEqual(fast_decision(self.state, "next step").step_id, 3)
        get_simulator().inject("cdra_valve_state", "fault")
        self.assertEqual(fast_decision(self.state, "confirm").step_id, 8)

    def test_each_procedure_can_be_selected_but_not_silently_changed(self):
        for name, expected in (("airlock", "EVA-PREP-001"),
                               ("coolant", "COOL-JMP-002"),
                               ("battery", "BATT-ORU-003"),
                               ("cdra", "CDRA-FAULT-004")):
            state = self.store.create()
            fast_decision(state, f"select {name} procedure")
            fast_decision(state, "confirm procedure")
            self.assertEqual(state["procedure_id"], expected)
            if expected != "BATT-ORU-003":
                old_step = state["step_id"]
                self.assertEqual(fast_decision(state, "battery procedure").action, "clarify")
                self.assertEqual(state["step_id"], old_step)

    def test_negation_and_unsafe_sensor_cannot_advance(self):
        self.select("airlock")
        original = self.state["step_id"]
        self.assertEqual(fast_decision(self.state, "do not confirm").action, "block")
        self.assertEqual(self.state["step_id"], original)
        get_simulator().inject("suit_battery_pct", 10.0)
        self.assertIn(fast_decision(self.state, "next step").action, {"block", "emergency"})
        self.assertEqual(self.state["step_id"], original)

    def test_critical_requires_readback_and_sensor(self):
        self.select("battery")
        self.assertEqual(fast_decision(self.state, "confirm").step_id, 2)
        self.assertEqual(fast_decision(self.state, "confirm").action, "block")
        self.assertEqual(self.state["step_id"], 2)
        self.assertEqual(fast_decision(self.state, "confirm step 2").action, "block")
        self.assertEqual(self.state["step_id"], 2)

    def test_state_persists_and_turn_replay_is_unique(self):
        self.select("airlock")
        result = {"decision": fast_decision(self.state, "next step").to_dict()}
        self.store.commit(self.state, "turn-a", result)
        self.assertEqual(self.store.get(self.state["id"])["step_id"], 2)
        self.assertEqual(self.store.get_turn(self.state["id"], "turn-a"), result)
        with self.assertRaises(Exception):
            self.store.commit(self.state, "turn-a", result)

    def test_coolant_leak_classification_requires_pressure_check(self):
        self.select("coolant")
        self.assertEqual(fast_decision(self.state, "next step").step_id, 2)
        self.assertEqual(fast_decision(self.state, "next step").step_id, 3)
        self.assertEqual(fast_decision(self.state, "confirm").action, "block")
        self.assertEqual(self.state["step_id"], 3)
        self.assertEqual(fast_decision(self.state, "white frost only no flakes").step_id, 4)
        self.assertEqual(fast_decision(self.state, "confirm step 4").action, "emergency")
        self.assertEqual(self.state["step_id"], 4)

    def test_coolant_ammonia_flakes_halt_session(self):
        self.select("coolant")
        fast_decision(self.state, "next step")
        fast_decision(self.state, "next step")
        self.assertEqual(fast_decision(self.state, "white flakes").action, "emergency")
        self.assertTrue(self.state["halted"])
        self.assertEqual(fast_decision(self.state, "next step").action, "emergency")
        self.assertEqual(self.state["step_id"], 3)

    def test_required_part_blocks_battery_advance(self):
        self.select("battery")
        fast_decision(self.state, "confirm")
        get_simulator().inject("battery_voltage_v", 2.0)
        self.assertEqual(fast_decision(self.state, "confirm step 2").step_id, 3)
        from .agent import session_engine
        original = session_engine.check_inventory
        def unavailable(part):
            return {"available": False} if part == "BATT-ORU-CONN-COVER" else original(part)
        with patch.object(session_engine, "check_inventory", side_effect=unavailable):
            self.assertEqual(fast_decision(self.state, "confirm step 3").action, "block")
        self.assertEqual(self.state["step_id"], 3)

    def test_cdra_co2_emergency_uses_step_25(self):
        self.select("cdra")
        get_simulator().inject("cabin_co2_mmHg", 6.0)
        self.assertEqual(fast_decision(self.state, "next step").action, "emergency")
        self.assertEqual(self.state["step_id"], 25)
        self.assertNotIn(1, self.state["completed_step_ids"])

    def test_cdra_transition_is_not_a_successful_valve_reset(self):
        self.select("cdra")
        get_simulator().inject("cabin_co2_mmHg", 4.0)
        fast_decision(self.state, "acknowledged")
        fast_decision(self.state, "next step")
        get_simulator().inject("cdra_valve_state", "transition")
        self.assertEqual(fast_decision(self.state, "confirm").step_id, 8)
        result = fast_decision(self.state, "confirm")
        self.assertEqual(result.action, "replan")
        self.assertEqual(self.state["step_id"], 20)


if __name__ == "__main__":
    unittest.main()
