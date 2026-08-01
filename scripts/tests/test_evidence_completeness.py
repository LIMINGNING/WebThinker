import os
import sys
import unittest


SCRIPTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from evidence_completeness import (  # noqa: E402
    defer_early_complete,
    detect_benchmark_leakage,
    extract_auxiliary_summary,
    merge_evidence_packet,
    new_evidence_state,
    override_complete_with_main_search,
    parse_evidence_packet,
    question_requires_shared_time_scope,
    render_evidence_update,
    render_private_evidence_state,
    render_tiered_evidence_note,
    select_followup_query,
    split_sequence_timeout,
    validate_global_completion,
    validate_tiered_evidence,
)
from prompts.prompts import (  # noqa: E402
    get_deep_web_explorer_instruction,
    get_evidence_completeness_main_instruction,
    get_evidence_completeness_web_explorer_instruction,
)


class EvidenceCompletenessTests(unittest.TestCase):
    def test_benchmark_rule_is_shared_and_parameter_controlled(self):
        phrase = "Never use benchmark datasets"
        baseline_enabled = get_deep_web_explorer_instruction(
            "query",
            "intent",
            "documents",
            reject_benchmark_leakage=True,
        )
        optimized_enabled = get_evidence_completeness_web_explorer_instruction(
            "query",
            "intent",
            "documents",
            "prior evidence",
            reject_benchmark_leakage=True,
        )
        baseline_disabled = get_deep_web_explorer_instruction(
            "query",
            "intent",
            "documents",
            reject_benchmark_leakage=False,
        )
        optimized_disabled = get_evidence_completeness_web_explorer_instruction(
            "query",
            "intent",
            "documents",
            "prior evidence",
            reject_benchmark_leakage=False,
        )

        self.assertIn(phrase, baseline_enabled)
        self.assertIn(phrase, optimized_enabled)
        self.assertNotIn(phrase, baseline_disabled)
        self.assertNotIn(phrase, optimized_disabled)

    def test_splits_sequence_timeout_for_reserved_final_answer(self):
        self.assertEqual(split_sequence_timeout(300, 45, True), (255.0, 45.0))

    def test_timeout_reserve_is_disabled_for_baseline(self):
        self.assertEqual(split_sequence_timeout(300, 45, False), (300.0, 0.0))

    def test_rejects_reserve_that_consumes_entire_timeout(self):
        with self.assertRaises(ValueError):
            split_sequence_timeout(300, 300, True)

    def test_prompt_describes_controller_followups_as_shared_budget(self):
        instruction = get_evidence_completeness_main_instruction(
            evidence_followup_limit=4,
            evidence_delivery_mode="aux_summary",
        )
        self.assertIn("same overall search limit", instruction)
        self.assertIn("not extra searches", instruction)
        self.assertIn("do not consume the controller", instruction)

    def test_extracts_auxiliary_summary_without_private_packet(self):
        text = """**Final Information**
<MAIN_SUMMARY>
Two albums are verified, while one snippet-only title remains unverified.
</MAIN_SUMMARY>
<EVIDENCE_PACKET>
{"coverage_status": "partial", "suggested_query": "private query"}
</EVIDENCE_PACKET>"""
        summary = extract_auxiliary_summary(text)
        self.assertEqual(
            summary,
            "Two albums are verified, while one snippet-only title remains unverified.",
        )
        self.assertNotIn("coverage_status", summary)
        self.assertNotIn("private query", summary)

    def test_auxiliary_summary_fallback_never_exposes_packet(self):
        text = """**Final Information**
<EVIDENCE_PACKET>
{"coverage_status": "partial"}
</EVIDENCE_PACKET>"""
        summary = extract_auxiliary_summary(text)
        self.assertNotIn("EVIDENCE_PACKET", summary)
        self.assertNotIn("coverage_status", summary)

    def test_private_state_keeps_structured_evidence_for_auxiliary_model(self):
        state = new_evidence_state()
        state["collections"] = {"target_list": ["Album A"]}
        state["coverage_status"] = "partial"
        rendered = render_private_evidence_state(state)
        self.assertIn('"target_list": [', rendered)
        self.assertIn('"Album A"', rendered)
        self.assertIn('"coverage_status": "partial"', rendered)
        self.assertIn("do not copy this JSON into MAIN_SUMMARY", rendered)

    def test_global_requirement_prompt_includes_original_question_and_schema(self):
        instruction = get_evidence_completeness_web_explorer_instruction(
            "current subquery",
            "current intent",
            "documents",
            "prior evidence",
            original_question="Which two capitals are farthest apart?",
            use_global_requirements=True,
        )
        self.assertIn("Original question:", instruction)
        self.assertIn("Which two capitals are farthest apart?", instruction)
        self.assertIn('"requirements": [', instruction)
        self.assertIn("Completion of the Current search query alone is not sufficient", instruction)

    def test_v1_prompt_omits_global_requirement_fields(self):
        instruction = get_evidence_completeness_web_explorer_instruction(
            "current subquery",
            "current intent",
            "documents",
            "prior evidence",
        )
        self.assertNotIn("Original question:", instruction)
        self.assertNotIn('"requirements": [', instruction)

    def test_v2c_prompt_requests_atomic_evidence_and_shared_time_scope(self):
        instruction = get_evidence_completeness_web_explorer_instruction(
            "current subquery",
            "current intent",
            "documents",
            "prior evidence",
            original_question="Compare both totals as of May 2023.",
            use_global_requirements=True,
            use_risk_tiers=True,
        )
        self.assertIn('"evidence_id":', instruction)
        self.assertIn('"scope_status": "aligned|conflict|unknown|not_applicable"', instruction)
        self.assertIn("Apply one shared cutoff to every side of a comparison", instruction)
        self.assertIn("at most one focused suggested_query", instruction)
        self.assertIn("at least the top two candidates", instruction)
        self.assertIn("set_difference", instruction)
        self.assertIn("never shift a value from the following entry", instruction)
        self.assertIn("search snippet conflicts with fetched page content", instruction)

    def test_detects_benchmark_answer_dump_without_domain_rules(self):
        question = "Which class received the bug fix in the July changelog?"
        document = {
            "title": "Experiment results",
            "snippet": (
                'task_id: abc Question: Which class received the bug fix in the July '
                'changelog? Final answer: ExampleClass'
            ),
            "page_info": "",
        }
        reason = detect_benchmark_leakage(document, question)
        self.assertIsNotNone(reason)
        self.assertIn("answer", reason)

    def test_does_not_filter_normal_authoritative_page(self):
        question = "Which class received the bug fix in the July changelog?"
        document = {
            "title": "Version 1.0 changelog",
            "snippet": "Bug fix: ExampleClass now handles empty labels.",
            "page_info": "Official release notes and contributor details.",
        }
        self.assertIsNone(detect_benchmark_leakage(document, question))

    def test_detects_reproduced_question_with_benchmark_metadata(self):
        question = "Which class received the bug fix in the July changelog?"
        document = {
            "title": "Benchmark experiment",
            "snippet": (
                "GAIA benchmark question: Which class received the bug fix in the "
                "July changelog?"
            ),
            "page_info": "Model evaluation trace.",
        }
        self.assertIsNotNone(detect_benchmark_leakage(document, question))

    def test_parses_and_normalizes_packet(self):
        packet = parse_evidence_packet(
            """**Final Information**
<EVIDENCE_PACKET>
{
  "task_operation": "count",
  "requires_complete_coverage": true,
          "new_evidence": [{
    "claim": "Album A was released in 2005.",
    "source_url": "https://example.com/a",
    "source_access": "snippet_only",
    "strength": "medium",
    "supports_constraints": ["year range"],
    "collection_key": "target_list",
    "items": ["Album A"]
  }],
  "coverage_status": "partial",
  "missing_evidence": ["The complete album list"],
  "coverage_reason": "Only a snippet was available.",
  "suggested_query": "complete discography 2000 2009"
}
</EVIDENCE_PACKET>"""
        )
        self.assertTrue(packet["parse_ok"])
        self.assertEqual(packet["task_operation"], "count")
        self.assertEqual(packet["coverage_status"], "partial")
        self.assertTrue(packet["requires_complete_coverage"])
        self.assertEqual(packet["new_evidence"][0]["source_access"], "snippet_only")
        self.assertEqual(packet["new_evidence"][0]["collection_key"], "target_list")
        self.assertEqual(packet["new_evidence"][0]["items"], ["Album A"])

    def test_parses_global_answer_requirements(self):
        packet = parse_evidence_packet(
            """<EVIDENCE_PACKET>
{
  "task_operation": "comparison",
  "requires_complete_coverage": true,
  "requirements": [
    {
      "id": "R1",
      "kind": "coverage",
      "description": "Collect every candidate capital.",
      "status": "supported",
      "supporting_claims": ["Ten capitals were listed."]
    },
    {
      "id": "r2",
      "kind": "relation",
      "description": "Determine the maximum capital-to-capital distance.",
      "status": "partial",
      "supporting_claims": []
    }
  ],
  "new_evidence": [],
  "coverage_status": "complete",
  "missing_evidence": [],
  "coverage_reason": "Current lookup finished.",
  "suggested_query": "remaining distance comparison"
}
</EVIDENCE_PACKET>""",
            use_global_requirements=True,
        )
        self.assertEqual(packet["requirements"][0]["id"], "r1")
        self.assertEqual(packet["requirements"][1]["status"], "partial")
        self.assertEqual(
            packet["schema_version"],
            "evidence-completeness-v2-global-requirements",
        )

    def test_global_requirements_persist_when_later_packet_omits_one(self):
        state = new_evidence_state(use_global_requirements=True)
        first = {
            "parse_ok": True,
            "task_operation": "multi_hop",
            "requires_complete_coverage": True,
            "requirements": [
                {"id": "r1", "kind": "lookup", "description": "Find A.", "status": "supported", "supporting_claims": ["A"]},
                {"id": "r2", "kind": "lookup", "description": "Find B.", "status": "missing", "supporting_claims": []},
            ],
            "new_evidence": [],
            "coverage_status": "partial",
            "missing_evidence": ["Find B."],
            "coverage_reason": "B is missing.",
            "suggested_query": "find B",
        }
        second = {
            **first,
            "requirements": [
                {"id": "r2", "kind": "lookup", "description": "Find B.", "status": "supported", "supporting_claims": ["B"]},
            ],
            "coverage_status": "complete",
            "missing_evidence": [],
            "coverage_reason": "Both are supported.",
            "suggested_query": "",
        }
        merge_evidence_packet(state, first, use_global_requirements=True)
        merge_evidence_packet(state, second, use_global_requirements=True)
        self.assertEqual([item["id"] for item in state["requirements"]], ["r1", "r2"])
        self.assertTrue(all(item["status"] == "supported" for item in state["requirements"]))

    def test_v2c_parses_atomic_evidence_and_requirement_links(self):
        packet = parse_evidence_packet(
            """<EVIDENCE_PACKET>
{
  "task_operation": "comparison",
  "requires_complete_coverage": true,
  "requirements": [{
    "id": "r1",
    "kind": "derived_operation",
    "description": "Find the player with the most walks.",
    "status": "supported",
    "time_scope": "1977 regular season",
    "scope_status": "aligned",
    "operation": "argmax",
    "supporting_evidence_ids": ["e1", "e2"],
    "supporting_claims": ["Both candidate values were retained."]
  }],
  "new_evidence": [{
    "evidence_id": "e1",
    "subject": "Roy White",
    "relation": "walks",
    "value": "75",
    "unit": "walks",
    "time_scope": "1977 regular season",
    "supports_requirements": ["r1"],
    "claim": "Roy White had 75 walks.",
    "source_url": "https://example.com/stats",
    "source_access": "page_excerpt",
    "strength": "strong",
    "supports_constraints": ["1977 Yankees"],
    "collection_key": "candidates",
    "items": ["Roy White"]
  }],
  "coverage_status": "complete",
  "missing_evidence": [],
  "coverage_reason": "Candidate values retained.",
  "suggested_query": ""
}
</EVIDENCE_PACKET>""",
            use_global_requirements=True,
            use_risk_tiers=True,
        )
        self.assertEqual(packet["schema_version"], "evidence-completeness-v2c-risk-tiers")
        self.assertEqual(packet["new_evidence"][0]["subject"], "Roy White")
        self.assertEqual(packet["new_evidence"][0]["value"], "75")
        self.assertEqual(packet["requirements"][0]["operation"], "argmax")
        self.assertEqual(packet["requirements"][0]["supporting_evidence_ids"], ["e1", "e2"])

    def test_v2c_program_computes_argmax_from_atomic_inputs(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        packet = {
            "parse_ok": True,
            "task_operation": "comparison",
            "requires_complete_coverage": True,
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Find the largest walk total.",
                "status": "supported",
                "time_scope": "1977 regular season",
                "scope_status": "aligned",
                "operation": "argmax",
                "supporting_evidence_ids": ["e1", "e2"],
                "supporting_claims": [],
            }],
            "new_evidence": [
                {"evidence_id": "e1", "subject": "Roy White", "relation": "walks", "value": "75", "unit": "walks", "time_scope": "1977 regular season", "supports_requirements": ["r1"], "claim": "Roy White had 75 walks.", "source_url": "u", "source_access": "page_excerpt", "strength": "strong", "supports_constraints": [], "collection_key": "candidates", "items": ["Roy White"]},
                {"evidence_id": "e2", "subject": "Reggie Jackson", "relation": "walks", "value": "74", "unit": "walks", "time_scope": "1977 regular season", "supports_requirements": ["r1"], "claim": "Reggie Jackson had 74 walks.", "source_url": "u", "source_access": "page_excerpt", "strength": "strong", "supports_constraints": [], "collection_key": "candidates", "items": ["Reggie Jackson"]},
            ],
            "coverage_status": "complete",
            "missing_evidence": [],
            "coverage_reason": "Complete table.",
            "suggested_query": "",
        }
        merge_evidence_packet(
            state,
            packet,
            use_global_requirements=True,
            use_risk_tiers=True,
        )
        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        requirement = state["requirements"][0]
        self.assertEqual(requirement["validated_result"], "Roy White")
        self.assertEqual(requirement["validated_value"], "75")
        self.assertIn("Program-checked argmax", render_tiered_evidence_note(state))

    def test_v2c_explicit_time_conflict_is_blocking(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [{
                "id": "r1",
                "kind": "time_constraint",
                "description": "Use totals as of May 2023.",
                "status": "supported",
                "time_scope": "May 2023",
                "scope_status": "conflict",
                "operation": "none",
                "supporting_evidence_ids": [],
            }],
        })
        reason = validate_tiered_evidence(state, minimum_complete_rounds=1)
        self.assertEqual(reason, "tiered_blocking_gaps")
        self.assertEqual(state["coverage_status"], "partial")
        self.assertIn("time scope conflict", state["blocking_gaps"][0])

    def test_v2c_weak_source_is_soft_risk_and_gets_one_followup(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "packet_count": 2,
            "suggested_query": "official independent total",
            "records": [{
                "evidence_id": "e1",
                "claim": "A snippet reports 90 participants.",
                "source_access": "snippet_only",
                "strength": "weak",
            }],
            "requirements": [{
                "id": "r1",
                "kind": "lookup",
                "description": "Find enrollment.",
                "status": "supported",
                "time_scope": "",
                "scope_status": "not_applicable",
                "operation": "none",
                "supporting_evidence_ids": ["e1"],
            }],
        })
        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        self.assertEqual(state["coverage_status"], "complete")
        self.assertTrue(state["verification_risks"])
        query, reason = select_followup_query(
            state,
            set(),
            max_followups=4,
            stagnation_limit=2,
            use_risk_tiers=True,
            max_verification_followups=1,
        )
        self.assertEqual(query, "official independent total")
        self.assertEqual(reason, "evidence_verification_followup")
        state["verification_followups"] = 1
        self.assertEqual(
            select_followup_query(
                state,
                set(),
                4,
                2,
                use_risk_tiers=True,
                max_verification_followups=1,
            ),
            (None, "verification_limit_reached"),
        )

    def test_v2c_derived_result_without_inputs_is_blocking(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Compute the difference.",
                "status": "supported",
                "time_scope": "",
                "scope_status": "not_applicable",
                "operation": "difference",
                "supporting_evidence_ids": [],
            }],
        })
        self.assertEqual(
            validate_tiered_evidence(state, minimum_complete_rounds=1),
            "tiered_blocking_gaps",
        )
        self.assertIn("no input evidence", state["blocking_gaps"][0])

    def test_v2c_later_packet_can_refine_requirement_operation(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        base_packet = {
            "parse_ok": True,
            "task_operation": "comparison",
            "requires_complete_coverage": True,
            "requirements": [{
                "id": "r1",
                "kind": "other",
                "description": "Determine the largest value.",
                "status": "partial",
                "time_scope": "",
                "scope_status": "unknown",
                "operation": "none",
                "supporting_evidence_ids": [],
                "supporting_claims": [],
            }],
            "new_evidence": [],
            "coverage_status": "partial",
            "missing_evidence": ["candidate values"],
            "coverage_reason": "Inputs are missing.",
            "suggested_query": "candidate values",
        }
        refined_packet = {
            **base_packet,
            "requirements": [{
                **base_packet["requirements"][0],
                "kind": "derived_operation",
                "status": "supported",
                "operation": "argmax",
                "supporting_evidence_ids": ["e1", "e2"],
            }],
            "new_evidence": [
                {"evidence_id": "e1", "subject": "A", "relation": "score", "value": "10", "claim": "A scored 10."},
                {"evidence_id": "e2", "subject": "B", "relation": "score", "value": "12", "claim": "B scored 12."},
            ],
            "coverage_status": "complete",
            "missing_evidence": [],
            "suggested_query": "",
        }

        merge_evidence_packet(
            state, base_packet, use_global_requirements=True, use_risk_tiers=True
        )
        merge_evidence_packet(
            state, refined_packet, use_global_requirements=True, use_risk_tiers=True
        )

        requirement = state["requirements"][0]
        self.assertEqual(requirement["kind"], "derived_operation")
        self.assertEqual(requirement["operation"], "argmax")
        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        self.assertEqual(requirement["validated_result"], "B")

    def test_v2c_count_accepts_non_numeric_atomic_members(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "packet_count": 2,
            "records": [
                {"evidence_id": "e1", "subject": "Album A", "value": "Album A", "claim": "Album A qualifies.", "source_access": "full_page", "strength": "strong"},
                {"evidence_id": "e2", "subject": "Album B", "value": "Album B", "claim": "Album B qualifies.", "source_access": "full_page", "strength": "strong"},
            ],
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Count qualifying albums.",
                "status": "supported",
                "time_scope": "",
                "scope_status": "not_applicable",
                "operation": "count",
                "supporting_evidence_ids": ["e1", "e2"],
            }],
        })

        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        self.assertEqual(state["requirements"][0]["validated_result"], "2")

    def test_v2c_set_difference_accepts_non_numeric_atomic_inputs(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "packet_count": 2,
            "records": [
                {"evidence_id": "e1", "subject": "all albums", "value": "A, B", "claim": "Albums are A and B."},
                {"evidence_id": "e2", "subject": "graded albums", "value": "A", "claim": "Album A was graded."},
            ],
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Find albums without a grade.",
                "status": "supported",
                "time_scope": "",
                "scope_status": "not_applicable",
                "operation": "set_difference",
                "supporting_evidence_ids": ["e1", "e2"],
            }],
        })

        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        self.assertFalse(state["blocking_gaps"])

    def test_v2c_blocker_generates_generic_fallback_query(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Identify the player with the most walks.",
                "status": "supported",
                "time_scope": "1977 regular season",
                "scope_status": "aligned",
                "operation": "argmax",
                "supporting_evidence_ids": ["e1"],
            }],
            "records": [{
                "evidence_id": "e1",
                "subject": "Player A",
                "value": "74",
                "claim": "Player A had 74 walks.",
            }],
        })

        self.assertEqual(
            validate_tiered_evidence(state, minimum_complete_rounds=1),
            "tiered_blocking_gaps",
        )
        self.assertIn("complete table all candidate values", state["suggested_query"])

    def test_v2c_detects_question_wide_temporal_comparison_cutoff(self):
        self.assertTrue(question_requires_shared_time_scope(
            "As of the end of season 44, how many more winners were there compared to Idol?"
        ))
        self.assertFalse(question_requires_shared_time_scope(
            "Which 2018 and 2019 studies used a common model?"
        ))

    def test_v2c_shared_cutoff_blocks_an_undated_comparison_input(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [
                {"id": "r1", "kind": "coverage", "description": "Count A.", "status": "supported", "time_scope": "through May 2023", "scope_status": "aligned", "operation": "none", "supporting_evidence_ids": []},
                {"id": "r2", "kind": "coverage", "description": "Count B.", "status": "supported", "time_scope": "", "scope_status": "not_applicable", "operation": "none", "supporting_evidence_ids": []},
            ],
        })

        self.assertEqual(
            validate_tiered_evidence(
                state,
                minimum_complete_rounds=1,
                require_shared_time_scope=True,
            ),
            "tiered_blocking_gaps",
        )
        self.assertIn("shared time cutoff", " ".join(state["blocking_gaps"]))
        self.assertIn("through May 2023", state["suggested_query"])

    def test_v2c_count_does_not_count_list_evidence_records(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "records": [
                {"evidence_id": "e1", "subject": "show winners", "relation": "complete list", "value": "Alice (S1), Bob (S2)", "claim": "Two winner names are listed."},
                {"evidence_id": "e2", "subject": "show winners", "relation": "partial list", "value": "Carol (S3)", "claim": "Another winner is listed."},
            ],
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Count unique winners.",
                "status": "supported",
                "time_scope": "",
                "scope_status": "not_applicable",
                "operation": "count",
                "supporting_evidence_ids": ["e1", "e2"],
            }],
        })

        self.assertEqual(
            validate_tiered_evidence(state, minimum_complete_rounds=1),
            "tiered_blocking_gaps",
        )
        self.assertNotIn("validated_result", state["requirements"][0])
        self.assertIn("authoritative total", " ".join(state["blocking_gaps"]))

    def test_v2c_count_accepts_one_authoritative_numeric_total(self):
        state = new_evidence_state(use_global_requirements=True, use_risk_tiers=True)
        state.update({
            "coverage_status": "complete",
            "records": [{
                "evidence_id": "e1",
                "subject": "show winners",
                "relation": "unique winner count",
                "value": "42 unique winners",
                "claim": "The authoritative total is 42 unique winners.",
            }],
            "requirements": [{
                "id": "r1",
                "kind": "derived_operation",
                "description": "Count unique winners.",
                "status": "supported",
                "time_scope": "through season 44",
                "scope_status": "aligned",
                "operation": "count",
                "supporting_evidence_ids": ["e1"],
            }],
        })

        self.assertIsNone(validate_tiered_evidence(state, minimum_complete_rounds=1))
        self.assertEqual(state["requirements"][0]["validated_result"], "42")

    def test_requirement_progress_resets_stagnation_without_new_record(self):
        state = new_evidence_state(use_global_requirements=True)
        state["stagnant_rounds"] = 1
        packet = {
            "parse_ok": True,
            "task_operation": "multi_hop",
            "requires_complete_coverage": True,
            "requirements": [
                {"id": "r1", "kind": "lookup", "description": "Find A.", "status": "supported", "supporting_claims": ["A was found."]},
            ],
            "new_evidence": [],
            "coverage_status": "complete",
            "missing_evidence": [],
            "coverage_reason": "A was found.",
            "suggested_query": "",
        }
        merge_evidence_packet(state, packet, use_global_requirements=True)
        self.assertEqual(state["stagnant_rounds"], 0)

    def test_global_completion_rejects_unsupported_requirement(self):
        state = new_evidence_state(use_global_requirements=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [
                {"id": "r1", "description": "Find both comparison inputs.", "status": "supported"},
                {"id": "r2", "description": "Compare the inputs.", "status": "partial"},
            ],
            "missing_evidence": [],
        })
        reason = validate_global_completion(state)
        self.assertEqual(reason, "global_requirements_unsupported")
        self.assertEqual(state["coverage_status"], "partial")
        self.assertIn("Compare the inputs.", state["missing_evidence"])

    def test_global_completion_accepts_supported_requirements(self):
        state = new_evidence_state(use_global_requirements=True)
        state.update({
            "coverage_status": "complete",
            "requirements": [
                {"id": "r1", "description": "Find A.", "status": "supported"},
                {"id": "r2", "description": "Find B.", "status": "supported"},
            ],
            "missing_evidence": [],
        })
        self.assertIsNone(validate_global_completion(state))
        self.assertEqual(state["coverage_status"], "complete")

    def test_global_completion_rejects_empty_requirement_list(self):
        state = new_evidence_state(use_global_requirements=True)
        state["coverage_status"] = "complete"
        reason = validate_global_completion(state)
        self.assertEqual(reason, "global_requirements_missing")
        self.assertEqual(state["coverage_status"], "partial")

    def test_merge_deduplicates_claims(self):
        state = new_evidence_state()
        packet = {
            "parse_ok": True,
            "task_operation": "count",
            "requires_complete_coverage": True,
            "new_evidence": [
                {"claim": "Album A, 2005", "source_url": "u", "source_access": "full_page", "strength": "strong", "supports_constraints": []},
                {"claim": "Album A, 2005", "source_url": "u", "source_access": "full_page", "strength": "strong", "supports_constraints": []},
            ],
            "coverage_status": "partial",
            "missing_evidence": ["remaining years"],
            "coverage_reason": "partial list",
            "suggested_query": "remaining albums 2000 2009",
        }
        added = merge_evidence_packet(state, packet)
        self.assertEqual(added, 1)
        self.assertEqual(len(state["records"]), 1)
        self.assertEqual(state["stagnant_rounds"], 0)
        merge_evidence_packet(state, packet)
        self.assertEqual(state["stagnant_rounds"], 1)

    def test_unicode_claims_are_retained_and_deduplicated(self):
        state = new_evidence_state()
        packet = {
            "parse_ok": True,
            "task_operation": "single_fact",
            "requires_complete_coverage": False,
            "new_evidence": [
                {"claim": "专辑甲发行于2005年", "source_url": "u", "source_access": "page_excerpt", "strength": "strong", "supports_constraints": []},
                {"claim": "专辑甲发行于2005年", "source_url": "u", "source_access": "page_excerpt", "strength": "strong", "supports_constraints": []},
            ],
            "coverage_status": "complete",
            "missing_evidence": [],
            "coverage_reason": "direct fact",
            "suggested_query": "",
        }
        self.assertEqual(merge_evidence_packet(state, packet), 1)

    def test_merges_complementary_list_items_across_rounds(self):
        state = new_evidence_state()
        base_record = {
            "source_url": "u",
            "source_access": "page_excerpt",
            "strength": "strong",
            "supports_constraints": [],
            "collection_key": "target_list",
        }
        first = {
            "parse_ok": True,
            "task_operation": "count",
            "requires_complete_coverage": True,
            "new_evidence": [{
                **base_record,
                "claim": "First partial station list.",
                "items": ["Back Bay", "Ruggles", "Hyde Park"],
            }],
            "coverage_status": "partial",
            "missing_evidence": ["remaining stations"],
            "coverage_reason": "partial",
            "suggested_query": "complete schedule",
        }
        second = {
            **first,
            "new_evidence": [{
                **base_record,
                "claim": "Second complementary station list.",
                "items": ["Ruggles", "Forest Hills", "Hyde Park"],
            }],
            "coverage_status": "complete",
            "missing_evidence": [],
            "coverage_reason": "complete schedule",
            "suggested_query": "",
        }
        merge_evidence_packet(state, first)
        merge_evidence_packet(state, second)
        self.assertEqual(
            state["collections"]["target_list"],
            ["Back Bay", "Ruggles", "Hyde Park", "Forest Hills"],
        )
        rendered = render_evidence_update(state, added_count=1)
        self.assertIn("Merged target_list (4 items)", rendered)

    def test_defers_first_round_complete_count(self):
        state = new_evidence_state()
        state.update({
            "task_operation": "count",
            "requires_complete_coverage": True,
            "coverage_status": "complete",
            "packet_count": 1,
        })
        deferred = defer_early_complete(state, "route station list", minimum_rounds=2)
        self.assertTrue(deferred)
        self.assertEqual(state["coverage_status"], "partial")
        self.assertIn("independent confirmation", state["suggested_query"])

    def test_does_not_defer_single_fact_or_second_round(self):
        state = new_evidence_state()
        state.update({
            "task_operation": "single_fact",
            "coverage_status": "complete",
            "packet_count": 1,
        })
        self.assertFalse(defer_early_complete(state, "query", minimum_rounds=2))

        state["task_operation"] = "list"
        state["packet_count"] = 2
        self.assertFalse(defer_early_complete(state, "query", minimum_rounds=2))

    def test_partial_count_can_request_limited_followup(self):
        state = new_evidence_state()
        state.update({
            "requires_complete_coverage": True,
            "coverage_status": "partial",
            "suggested_query": "complete album list",
        })
        query, reason = select_followup_query(state, set(), max_followups=2, stagnation_limit=2)
        self.assertEqual(query, "complete album list")
        self.assertEqual(reason, "evidence_completeness_followup")

        state["forced_followups"] = 2
        query, reason = select_followup_query(state, set(), max_followups=2, stagnation_limit=2)
        self.assertIsNone(query)
        self.assertEqual(reason, "followup_limit_reached")

    def test_main_model_followups_do_not_consume_controller_budget(self):
        state = new_evidence_state()
        state.update({
            "requires_complete_coverage": True,
            "coverage_status": "partial",
            "suggested_query": "complete album list",
            "completeness_followups": 99,
            "forced_followups": 0,
        })

        query, reason = select_followup_query(
            state,
            set(),
            max_followups=2,
            stagnation_limit=2,
        )

        self.assertEqual(query, "complete album list")
        self.assertEqual(reason, "evidence_completeness_followup")

    def test_complete_or_stagnant_evidence_does_not_force_search(self):
        state = new_evidence_state()
        state.update({
            "requires_complete_coverage": True,
            "coverage_status": "complete",
            "suggested_query": "unnecessary query",
        })
        self.assertEqual(select_followup_query(state, set(), 2, 2), (None, None))

        state["coverage_status"] = "partial"
        state["stagnant_rounds"] = 2
        query, reason = select_followup_query(state, set(), 2, 2)
        self.assertIsNone(query)
        self.assertEqual(reason, "evidence_stagnated")

    def test_distinct_main_search_overrides_complete_within_budget(self):
        state = new_evidence_state()
        state.update({
            "coverage_status": "complete",
            "coverage_reason": "The first lookup is complete.",
            "missing_evidence": [],
        })

        overridden = override_complete_with_main_search(
            state,
            "second entity required relation",
            {"first entity lookup"},
            search_count=1,
            max_search_limit=15,
        )

        self.assertTrue(overridden)
        self.assertEqual(state["coverage_status"], "partial")
        self.assertEqual(state["suggested_query"], "second entity required relation")
        self.assertIn("Main-model follow-up requested", state["missing_evidence"][0])
        self.assertEqual(
            state["last_overridden_coverage_reason"],
            "The first lookup is complete.",
        )

    def test_duplicate_main_search_does_not_override_complete(self):
        state = new_evidence_state()
        state["coverage_status"] = "complete"

        overridden = override_complete_with_main_search(
            state,
            "Repeated Query",
            {"repeated query"},
            search_count=1,
            max_search_limit=15,
        )

        self.assertFalse(overridden)
        self.assertEqual(state["coverage_status"], "complete")

    def test_main_search_cannot_override_complete_without_budget(self):
        state = new_evidence_state()
        state["coverage_status"] = "complete"

        overridden = override_complete_with_main_search(
            state,
            "new focused query",
            set(),
            search_count=14,
            max_search_limit=15,
        )

        self.assertFalse(overridden)
        self.assertEqual(state["coverage_status"], "complete")

    def test_malformed_packet_has_safe_fallback(self):
        packet = parse_evidence_packet("not json", fallback_text="A possibly relevant fact.")
        self.assertFalse(packet["parse_ok"])
        self.assertEqual(packet["coverage_status"], "unknown")
        self.assertEqual(packet["new_evidence"][0]["strength"], "weak")

        state = new_evidence_state()
        added = merge_evidence_packet(state, packet)
        rendered = render_evidence_update(state, added)
        self.assertIn("A possibly relevant fact", rendered)
        self.assertIn("Coverage: unknown", rendered)

    def test_recovers_claim_from_truncated_packet(self):
        packet = parse_evidence_packet(
            '<EVIDENCE_PACKET>{"task_operation":"count",'
            '"requires_complete_coverage":true,"new_evidence":['
            '{"claim":"Stations include Hyde Park and Forest Hills.",'
            '"source_url":"https://example.com/schedule"'
        )
        self.assertFalse(packet["parse_ok"])
        self.assertEqual(packet["task_operation"], "count")
        self.assertTrue(packet["requires_complete_coverage"])
        self.assertEqual(
            packet["new_evidence"][0]["claim"],
            "Stations include Hyde Park and Forest Hills.",
        )

    def test_malformed_packet_preserves_previous_coverage_state(self):
        state = new_evidence_state()
        state.update({
            "task_operation": "count",
            "requires_complete_coverage": True,
            "coverage_status": "partial",
            "missing_evidence": ["one station"],
        })
        packet = parse_evidence_packet(
            '<EVIDENCE_PACKET>{"task_operation":"count",'
            '"requires_complete_coverage":true,"new_evidence":['
            '{"claim":"Forest Hills is in the sequence."}'
        )
        merge_evidence_packet(state, packet)
        self.assertEqual(state["coverage_status"], "partial")
        self.assertTrue(state["requires_complete_coverage"])
        self.assertEqual(state["missing_evidence"], ["one station"])


if __name__ == "__main__":
    unittest.main()
