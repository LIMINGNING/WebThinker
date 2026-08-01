import json
import re
from typing import Dict, List, Optional, Set, Tuple


PACKET_START = "<EVIDENCE_PACKET>"
PACKET_END = "</EVIDENCE_PACKET>"
SUMMARY_START = "<MAIN_SUMMARY>"
SUMMARY_END = "</MAIN_SUMMARY>"

VALID_OPERATIONS = {
    "single_fact",
    "list",
    "count",
    "intersection",
    "comparison",
    "multi_hop",
    "other",
}
VALID_ACCESS_LEVELS = {
    "full_page",
    "page_excerpt",
    "snippet_only",
    "search_result_only",
    "inaccessible",
}
VALID_STRENGTHS = {"strong", "medium", "weak"}
VALID_COVERAGE = {"complete", "partial", "unknown"}
VALID_COLLECTION_KEYS = {"target_list", "left_set", "right_set", "candidates", "none"}
VALID_REQUIREMENT_KINDS = {
    "lookup",
    "relation",
    "coverage",
    "calculation_input",
    "derived_operation",
    "time_constraint",
    "source_constraint",
    "output_constraint",
    "other",
}
VALID_REQUIREMENT_STATUSES = {"supported", "partial", "missing"}
VALID_SCOPE_STATUSES = {"aligned", "conflict", "unknown", "not_applicable"}
VALID_DERIVED_OPERATIONS = {
    "none",
    "argmax",
    "argmin",
    "count",
    "difference",
    "set_difference",
    "comparison",
    "calculation",
}

BENCHMARK_LEAK_MARKERS = {
    "benchmark",
    "final answer",
    "gold answer",
    "reference answer",
    "annotator_metadata",
    "annotator metadata",
    "task_id",
    "benchmark answer",
    "pred_answer",
}


def split_sequence_timeout(
    sequence_timeout: float,
    final_answer_reserve_seconds: float,
    enabled: bool,
) -> Tuple[float, float]:
    total = float(sequence_timeout or 0)
    reserve = float(final_answer_reserve_seconds or 0)
    if total <= 0 or not enabled or reserve <= 0:
        return total, 0.0
    if reserve >= total:
        raise ValueError(
            "Final-answer reserve must be smaller than the sequence timeout."
        )
    return total - reserve, reserve


def _clip(value: object, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def extract_auxiliary_summary(text: str, fallback_text: str = "") -> str:
    for candidate in (text, fallback_text):
        match = re.search(
            rf"{re.escape(SUMMARY_START)}\s*(.*?)\s*{re.escape(SUMMARY_END)}",
            str(candidate or ""),
            flags=re.DOTALL | re.IGNORECASE,
        )
        if match:
            summary = _clip(match.group(1), 5000)
            if summary:
                return summary

    for candidate in (text, fallback_text):
        candidate = str(candidate or "")
        packet_index = candidate.find(PACKET_START)
        if packet_index >= 0:
            candidate = candidate[:packet_index]
        candidate = re.sub(
            r"^\s*\*{0,2}Final Information\*{0,2}\s*:?\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        candidate = candidate.replace(SUMMARY_START, "").replace(SUMMARY_END, "")
        summary = _clip(candidate, 5000)
        if summary:
            return summary

    return "The auxiliary reader did not produce a reliable natural-language summary for this search."


def _word_tokens(value: object) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
        if len(token) >= 3
    }


def detect_benchmark_leakage(document: Dict, question: str) -> Optional[str]:
    text = " ".join(
        str(document.get(field, ""))
        for field in ("title", "snippet", "page_info")
    ).lower()
    matched_markers = sorted(marker for marker in BENCHMARK_LEAK_MARKERS if marker in text)
    if not matched_markers:
        return None

    question_tokens = _word_tokens(question)
    document_tokens = _word_tokens(text)
    overlap = (
        len(question_tokens & document_tokens) / len(question_tokens)
        if question_tokens
        else 0.0
    )
    normalized_question = re.sub(r"\s+", " ", question.lower()).strip()
    normalized_text = re.sub(r"\s+", " ", text)
    exact_question = len(normalized_question) >= 40 and normalized_question in normalized_text

    answer_dump = (
        ("task_id" in matched_markers or "annotator_metadata" in matched_markers)
        and any("answer" in marker for marker in matched_markers)
    )
    if answer_dump:
        return "benchmark answer dump markers: " + ", ".join(matched_markers)
    benchmark_metadata = any(
        marker in matched_markers
        for marker in ("benchmark", "task_id", "annotator_metadata", "annotator metadata")
    )
    if exact_question and (benchmark_metadata or any("answer" in marker for marker in matched_markers)):
        return "exact question text appears beside benchmark or answer metadata"
    if overlap >= 0.65 and (benchmark_metadata or any("answer" in marker for marker in matched_markers)):
        return f"question-token overlap {overlap:.2f} beside benchmark or answer metadata"
    return None


def _string_list(value: object, item_limit: int = 240, max_items: int = 8) -> List[str]:
    if not isinstance(value, list):
        return []
    items = []
    for item in value[:max_items]:
        text = _clip(item, item_limit)
        if text:
            items.append(text)
    return items


def _parse_requirements(
    value: object,
    max_items: int = 6,
    use_risk_tiers: bool = False,
) -> List[Dict]:
    if not isinstance(value, list):
        return []

    requirements = []
    seen_ids = set()
    for index, item in enumerate(value[:max_items], start=1):
        if not isinstance(item, dict):
            continue
        description = _clip(item.get("description"), 320)
        if not description:
            continue
        requirement_id = _clip(item.get("id"), 32).lower() or f"r{index}"
        requirement_id = re.sub(r"[^a-z0-9_-]", "", requirement_id)
        if not requirement_id or requirement_id in seen_ids:
            continue
        kind = str(item.get("kind", "other")).strip().lower()
        if kind not in VALID_REQUIREMENT_KINDS:
            kind = "other"
        status = str(item.get("status", "missing")).strip().lower()
        if status not in VALID_REQUIREMENT_STATUSES:
            status = "missing"
        requirement = {
            "id": requirement_id,
            "kind": kind,
            "description": description,
            "status": status,
            "supporting_claims": _string_list(
                item.get("supporting_claims"),
                item_limit=320,
                max_items=6,
            ),
        }
        if use_risk_tiers:
            scope_status = str(
                item.get("scope_status", "not_applicable")
            ).strip().lower()
            if scope_status not in VALID_SCOPE_STATUSES:
                scope_status = "unknown"
            operation = str(item.get("operation", "none")).strip().lower()
            if operation not in VALID_DERIVED_OPERATIONS:
                operation = "none"
            requirement.update({
                "time_scope": _clip(item.get("time_scope"), 160),
                "scope_status": scope_status,
                "operation": operation,
                "supporting_evidence_ids": _string_list(
                    item.get("supporting_evidence_ids"),
                    item_limit=32,
                    max_items=16,
                ),
            })
        requirements.append(requirement)
        seen_ids.add(requirement_id)
    return requirements


def _extract_json_object(text: str) -> Optional[Dict]:
    if not text:
        return None

    blocks = re.findall(
        rf"{re.escape(PACKET_START)}\s*(.*?)\s*{re.escape(PACKET_END)}",
        text,
        flags=re.DOTALL,
    )
    candidates = blocks if blocks else [text]
    for candidate in reversed(candidates):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate.strip(), flags=re.DOTALL)
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end <= start:
            continue
        try:
            parsed = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _recover_truncated_packet(text: str) -> Dict:
    def extract_scalar(field: str) -> str:
        match = re.search(rf'"{re.escape(field)}"\s*:\s*"([^"\r\n]*)', text)
        return match.group(1).strip() if match else ""

    claims = []
    for match in re.finditer(r'"claim"\s*:\s*"((?:\\.|[^"\\])*)"', text):
        try:
            claim = json.loads(f'"{match.group(1)}"')
        except json.JSONDecodeError:
            claim = match.group(1)
        claim = _clip(claim, 900)
        if claim:
            claims.append({
                "claim": claim,
                "source_url": "",
                "source_access": "search_result_only",
                "strength": "weak",
                "supports_constraints": [],
            })

    required_match = re.search(r'"requires_complete_coverage"\s*:\s*(true|false)', text, re.IGNORECASE)
    return {
        "task_operation": extract_scalar("task_operation"),
        "requires_complete_coverage": (
            required_match.group(1).lower() == "true" if required_match else False
        ),
        "new_evidence": claims,
        "coverage_status": extract_scalar("coverage_status"),
        "missing_evidence": [],
        "coverage_reason": "",
        "suggested_query": extract_scalar("suggested_query"),
        "requirements": [],
    }


def parse_evidence_packet(
    text: str,
    fallback_text: str = "",
    use_global_requirements: bool = False,
    use_risk_tiers: bool = False,
) -> Dict:
    raw = _extract_json_object(text)
    parse_ok = raw is not None
    raw = raw or _recover_truncated_packet(text)

    operation = str(raw.get("task_operation", "other")).strip().lower()
    if operation not in VALID_OPERATIONS:
        operation = "other"

    coverage_status = str(raw.get("coverage_status", "unknown")).strip().lower()
    if coverage_status not in VALID_COVERAGE:
        coverage_status = "unknown"

    new_evidence = []
    raw_evidence = raw.get("new_evidence", [])
    if isinstance(raw_evidence, list):
        for item in raw_evidence:
            if not isinstance(item, dict):
                continue
            claim = _clip(item.get("claim"), 900)
            if not claim:
                continue
            access = str(item.get("source_access", "search_result_only")).strip().lower()
            if access not in VALID_ACCESS_LEVELS:
                access = "search_result_only"
            strength = str(item.get("strength", "weak")).strip().lower()
            if strength not in VALID_STRENGTHS:
                strength = "weak"
            collection_key = str(item.get("collection_key", "none")).strip().lower()
            if collection_key not in VALID_COLLECTION_KEYS:
                collection_key = "none"
            record = {
                "claim": claim,
                "source_url": _clip(item.get("source_url"), 500),
                "source_access": access,
                "strength": strength,
                "supports_constraints": _string_list(item.get("supports_constraints"), max_items=6),
                "collection_key": collection_key,
                "items": _string_list(item.get("items"), item_limit=180, max_items=30),
            }
            if use_risk_tiers:
                record.update({
                    "evidence_id": _clip(item.get("evidence_id"), 32).lower(),
                    "subject": _clip(item.get("subject"), 180),
                    "relation": _clip(item.get("relation"), 180),
                    "value": _clip(item.get("value"), 240),
                    "unit": _clip(item.get("unit"), 80),
                    "time_scope": _clip(item.get("time_scope"), 160),
                    "supports_requirements": _string_list(
                        item.get("supports_requirements"),
                        item_limit=32,
                        max_items=8,
                    ),
                })
            new_evidence.append(record)

    if not parse_ok and not new_evidence and fallback_text:
        fallback = _clip(fallback_text, 800)
        if fallback:
            new_evidence = [{
                "claim": fallback,
                "source_url": "",
                "source_access": "search_result_only",
                "strength": "weak",
                "supports_constraints": [],
                "collection_key": "none",
                "items": [],
            }]

    suggested_query = _clip(raw.get("suggested_query"), 300)
    requires_complete_coverage = bool(raw.get("requires_complete_coverage", False))
    if operation in {"list", "count", "intersection", "comparison"}:
        requires_complete_coverage = True

    return {
        "schema_version": (
            "evidence-completeness-v2c-risk-tiers"
            if use_risk_tiers
            else (
                "evidence-completeness-v2-global-requirements"
                if use_global_requirements
                else "evidence-completeness-v1"
            )
        ),
        "parse_ok": parse_ok,
        "task_operation": operation,
        "requires_complete_coverage": requires_complete_coverage,
        "requirements": (
            _parse_requirements(
                raw.get("requirements"),
                use_risk_tiers=use_risk_tiers,
            )
            if use_global_requirements
            else []
        ),
        "new_evidence": new_evidence,
        "coverage_status": coverage_status,
        "missing_evidence": _string_list(raw.get("missing_evidence"), max_items=6),
        "coverage_reason": _clip(raw.get("coverage_reason"), 500),
        "suggested_query": suggested_query,
    }


def new_evidence_state(
    use_global_requirements: bool = False,
    use_risk_tiers: bool = False,
) -> Dict:
    return {
        "schema_version": (
            "evidence-completeness-v2c-risk-tiers"
            if use_risk_tiers
            else (
                "evidence-completeness-v2-global-requirements"
                if use_global_requirements
                else "evidence-completeness-v1"
            )
        ),
        "records": [],
        "collections": {},
        "requirements": [],
        "task_operation": "other",
        "requires_complete_coverage": False,
        "coverage_status": "unknown",
        "missing_evidence": [],
        "coverage_reason": "",
        "suggested_query": "",
        "blocking_gaps": [],
        "verification_risks": [],
        "shared_time_scope_required": False,
        "shared_time_scope": "",
        "forced_followups": 0,
        "verification_followups": 0,
        "completeness_followups": 0,
        "stagnant_rounds": 0,
        "packet_count": 0,
        "parse_failures": 0,
    }


def _claim_key(record: Dict) -> str:
    claim = str(record.get("claim", "")).lower()
    return "".join(character for character in claim if character.isalnum())


def _item_key(item: str) -> str:
    return "".join(character for character in item.lower() if character.isalnum())


def _normalize_evidence_id(value: object) -> str:
    return re.sub(r"[^a-z0-9_-]", "", str(value or "").strip().lower())[:32]


def _next_evidence_id(used_ids: Set[str]) -> str:
    index = 1
    while f"e{index}" in used_ids:
        index += 1
    return f"e{index}"


def _merge_global_requirements(
    state: Dict,
    incoming_requirements: List[Dict],
    max_requirements: int = 6,
) -> int:
    requirements = state.setdefault("requirements", [])
    existing = {item.get("id"): item for item in requirements if item.get("id")}
    updates = 0
    for incoming in incoming_requirements:
        requirement_id = incoming.get("id")
        if requirement_id in existing:
            current = existing[requirement_id]
            next_status = incoming.get("status", current.get("status", "missing"))
            if next_status != current.get("status"):
                updates += 1
            current["status"] = next_status
            incoming_kind = incoming.get("kind", "other")
            if incoming_kind != "other" and incoming_kind != current.get("kind"):
                current["kind"] = incoming_kind
                updates += 1
            current["description"] = (
                current.get("description") or incoming.get("description", "")
            )
            claims = current.setdefault("supporting_claims", [])
            claim_keys = {_item_key(claim) for claim in claims}
            for claim in incoming.get("supporting_claims", []):
                key = _item_key(claim)
                if key and key not in claim_keys:
                    claims.append(claim)
                    claim_keys.add(key)
                    updates += 1
            current["supporting_claims"] = claims[-6:]
            incoming_time_scope = incoming.get("time_scope", "")
            if incoming_time_scope and incoming_time_scope != current.get("time_scope"):
                current["time_scope"] = incoming_time_scope
                updates += 1
            incoming_operation = incoming.get("operation", "none")
            if (
                incoming_operation != "none"
                and incoming_operation != current.get("operation")
            ):
                current["operation"] = incoming_operation
                updates += 1
            if incoming.get("scope_status"):
                if incoming["scope_status"] != current.get("scope_status"):
                    updates += 1
                current["scope_status"] = incoming["scope_status"]
            evidence_ids = current.setdefault("supporting_evidence_ids", [])
            known_ids = set(evidence_ids)
            for evidence_id in incoming.get("supporting_evidence_ids", []):
                if evidence_id and evidence_id not in known_ids:
                    evidence_ids.append(evidence_id)
                    known_ids.add(evidence_id)
                    updates += 1
            current["supporting_evidence_ids"] = evidence_ids[-16:]
            continue
        if len(requirements) >= max_requirements:
            continue
        added = dict(incoming)
        added["supporting_claims"] = list(incoming.get("supporting_claims", []))
        requirements.append(added)
        existing[requirement_id] = added
        updates += 1
    return updates


def validate_global_completion(state: Dict) -> Optional[str]:
    """Require every global answer requirement before accepting complete."""
    state["requires_complete_coverage"] = True
    requirements = state.get("requirements", [])
    reason = None
    gaps = []

    if not requirements:
        reason = "global_requirements_missing"
        gaps = ["Global answer requirements have not been established"]
    else:
        unsupported = [
            requirement
            for requirement in requirements
            if requirement.get("status") != "supported"
        ]
        if unsupported:
            reason = "global_requirements_unsupported"
            gaps = [requirement.get("description", "") for requirement in unsupported]
        elif state.get("missing_evidence"):
            reason = "global_missing_evidence_present"

    if reason is None:
        return None

    missing = list(state.get("missing_evidence", []))
    missing_keys = {_item_key(item) for item in missing}
    for gap in gaps:
        gap = _clip(gap, 240)
        key = _item_key(gap)
        if gap and key not in missing_keys:
            missing.append(gap)
            missing_keys.add(key)
    state["missing_evidence"] = missing[-6:]

    if state.get("coverage_status") == "complete":
        state["coverage_status"] = "partial"
        if reason == "global_requirements_missing":
            state["coverage_reason"] = (
                "Complete was rejected because the original question's global answer "
                "requirements were not established."
            )
        elif reason == "global_requirements_unsupported":
            state["coverage_reason"] = (
                "Complete was rejected because one or more global answer requirements "
                "remain partial or missing."
            )
        else:
            state["coverage_reason"] = (
                "Complete was rejected because unresolved evidence gaps remain."
            )
        return reason
    return None


def _append_unique(values: List[str], value: object, limit: int = 8) -> None:
    text = _clip(value, 240)
    if text and _item_key(text) not in {_item_key(item) for item in values}:
        values.append(text)
    if len(values) > limit:
        del values[:-limit]


def _numeric_value(value: object) -> Optional[float]:
    text = str(value or "").replace(",", "").strip()
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if len(matches) != 1:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def question_requires_shared_time_scope(question: str) -> bool:
    """Detect comparison questions whose leading cutoff applies to every operand."""
    normalized = re.sub(r"\s+", " ", str(question or "")).lower()
    has_cutoff = bool(re.search(
        r"\b(as of|by the end of|through|up to|截至|截止|到.+为止)\b",
        normalized,
    ))
    has_comparison = bool(re.search(
        r"\b(compared to|compare|difference|more than|less than|how many more|相比|差值|多多少|少多少)\b",
        normalized,
    ))
    return has_cutoff and has_comparison


def validate_tiered_evidence(
    state: Dict,
    minimum_complete_rounds: int = 2,
    require_shared_time_scope: bool = False,
) -> Optional[str]:
    """Classify logical blockers separately from limited verification risks."""
    state["requires_complete_coverage"] = True
    requirements = state.get("requirements", [])
    records = state.get("records", [])
    records_by_id = {
        _normalize_evidence_id(record.get("evidence_id")): record
        for record in records
        if _normalize_evidence_id(record.get("evidence_id"))
    }
    blocking = []
    risks = []
    shared_time_scope = next((
        requirement.get("time_scope", "")
        for requirement in requirements
        if requirement.get("time_scope")
    ), "")
    state["shared_time_scope_required"] = require_shared_time_scope
    state["shared_time_scope"] = shared_time_scope

    if not requirements:
        blocking.append("Global answer requirements have not been established")

    for requirement in requirements:
        requirement_id = requirement.get("id", "requirement")
        description = requirement.get("description", requirement_id)
        kind = requirement.get("kind", "other")
        status = requirement.get("status", "missing")
        scope_status = requirement.get("scope_status", "not_applicable")
        time_scope = requirement.get("time_scope", "")
        evidence_ids = [
            _normalize_evidence_id(value)
            for value in requirement.get("supporting_evidence_ids", [])
            if _normalize_evidence_id(value)
        ]
        missing_ids = [value for value in evidence_ids if value not in records_by_id]

        if scope_status == "conflict":
            blocking.append(f"{requirement_id}: explicit time scope conflict for {description}")
        elif time_scope and scope_status == "unknown":
            risks.append(f"{requirement_id}: time scope is not verified for {description}")

        if kind == "output_constraint":
            continue
        if (
            require_shared_time_scope
            and kind in {"lookup", "relation", "coverage", "calculation_input"}
            and not time_scope
        ):
            blocking.append(
                f"{requirement_id}: comparison input lacks the shared time cutoff for {description}"
            )
        if status == "missing":
            if kind in {"coverage", "source_constraint"}:
                risks.append(f"{requirement_id}: coverage remains unverified for {description}")
            else:
                blocking.append(f"{requirement_id}: required input is missing for {description}")
        elif status == "partial":
            if kind in {"calculation_input", "derived_operation", "time_constraint"}:
                blocking.append(f"{requirement_id}: required input is incomplete for {description}")
            else:
                risks.append(f"{requirement_id}: evidence is only partial for {description}")

        if missing_ids:
            blocking.append(
                f"{requirement_id}: referenced evidence does not exist ({', '.join(missing_ids)})"
            )

        if kind == "calculation_input" and not evidence_ids:
            blocking.append(f"{requirement_id}: calculation input has no atomic evidence")

        operation = requirement.get("operation", "none")
        if kind == "derived_operation" or operation != "none":
            if not evidence_ids:
                blocking.append(f"{requirement_id}: derived result has no input evidence")
                continue
            input_records = [records_by_id[value] for value in evidence_ids if value in records_by_id]
            if operation in {"comparison", "set_difference"} and len(input_records) < 2:
                blocking.append(
                    f"{requirement_id}: {operation} lacks 2 atomic evidence inputs"
                )
                continue

            numeric_inputs = []
            if operation in {"argmax", "argmin", "difference", "calculation"}:
                numeric_inputs = [
                    (record, _numeric_value(record.get("value")))
                    for record in input_records
                ]
                numeric_inputs = [item for item in numeric_inputs if item[1] is not None]
                minimum_inputs = (
                    2
                    if operation in {"argmax", "argmin", "difference"}
                    else 1
                )
                if len(numeric_inputs) < minimum_inputs:
                    blocking.append(
                        f"{requirement_id}: {operation} lacks {minimum_inputs} numeric evidence inputs"
                    )
                    continue
            if operation in {"argmax", "argmin"}:
                selector = max if operation == "argmax" else min
                selected_record, selected_value = selector(
                    numeric_inputs,
                    key=lambda item: item[1],
                )
                requirement["validated_result"] = (
                    selected_record.get("subject")
                    or selected_record.get("claim")
                )
                requirement["validated_inputs"] = [
                    f"{record.get('subject') or record.get('evidence_id')}={value:g}"
                    for record, value in numeric_inputs
                ]
                requirement["validated_value"] = f"{selected_value:g}"
            elif operation == "difference" and len(numeric_inputs) >= 2:
                result = numeric_inputs[0][1] - numeric_inputs[1][1]
                requirement["validated_result"] = f"{result:g}"
                requirement["validated_inputs"] = [
                    f"{record.get('subject') or record.get('evidence_id')}={value:g}"
                    for record, value in numeric_inputs[:2]
                ]
            elif operation == "count":
                direct_totals = [
                    _numeric_value(record.get("value"))
                    for record in input_records
                    if any(
                        marker in str(record.get("relation", "")).lower()
                        for marker in ("count", "total", "number")
                    )
                ]
                direct_totals = [value for value in direct_totals if value is not None]
                distinct_totals = set(direct_totals)
                merged_items = {
                    _item_key(item): item
                    for record in input_records
                    for item in record.get("items", [])
                    if _item_key(item)
                }
                atomic_subjects = {
                    _item_key(record.get("subject")): record.get("subject")
                    for record in input_records
                    if record.get("subject")
                    and not record.get("items")
                    and not any(
                        marker in str(record.get("relation", "")).lower()
                        for marker in ("list", "count", "total", "number")
                    )
                    and not re.search(r"[,;]", str(record.get("value", "")))
                }
                if len(distinct_totals) == 1:
                    result = next(iter(distinct_totals))
                    requirement["validated_result"] = f"{result:g}"
                    requirement["validated_inputs"] = evidence_ids
                elif len(distinct_totals) > 1:
                    blocking.append(
                        f"{requirement_id}: authoritative count evidence conflicts"
                    )
                elif merged_items and all(record.get("items") for record in input_records):
                    requirement["validated_result"] = str(len(merged_items))
                    requirement["validated_inputs"] = evidence_ids
                elif atomic_subjects and len(atomic_subjects) == len(input_records):
                    requirement["validated_result"] = str(len(atomic_subjects))
                    requirement["validated_inputs"] = evidence_ids
                else:
                    blocking.append(
                        f"{requirement_id}: count lacks an authoritative total or complete atomic members"
                    )

        supporting_records = [records_by_id[value] for value in evidence_ids if value in records_by_id]
        if supporting_records and all(
            record.get("strength") == "weak"
            or record.get("source_access") in {"snippet_only", "search_result_only", "inaccessible"}
            for record in supporting_records
        ):
            risks.append(f"{requirement_id}: support is limited to weak or snippet-level evidence")

    if state.get("missing_evidence"):
        for gap in state["missing_evidence"][:3]:
            risks.append(f"reported evidence gap: {gap}")

    if (
        state.get("task_operation") in {"list", "count", "intersection", "comparison"}
        and state.get("coverage_status") == "complete"
        and state.get("packet_count", 0) < minimum_complete_rounds
    ):
        risks.append("completeness-sensitive answer has only one search round")

    state["blocking_gaps"] = []
    state["verification_risks"] = []
    for gap in blocking:
        _append_unique(state["blocking_gaps"], gap)
    for risk in risks:
        _append_unique(state["verification_risks"], risk)

    if state["blocking_gaps"]:
        if state.get("coverage_status") == "complete":
            state["coverage_status"] = "partial"
        state["coverage_reason"] = (
            "Answer readiness was rejected because required inputs or explicit scope "
            "constraints are unresolved."
        )
        if not str(state.get("suggested_query", "")).strip():
            blocking_requirement_ids = {
                gap.split(":", 1)[0]
                for gap in blocking
                if ":" in gap
            }
            blocked_requirements = [
                requirement
                for requirement in requirements
                if requirement.get("id") in blocking_requirement_ids
            ]
            if blocked_requirements:
                requirement = blocked_requirements[0]
                operation = requirement.get("operation", "none")
                prefix = (
                    "complete table all candidate values for"
                    if operation in {"argmax", "argmin"}
                    else "verify required evidence for"
                )
                state["suggested_query"] = _clip(
                    " ".join(filter(None, (
                        prefix,
                        requirement.get("description", ""),
                        requirement.get("time_scope", "") or shared_time_scope,
                    ))),
                    300,
                )
        return "tiered_blocking_gaps"
    return None


def render_tiered_evidence_note(state: Dict, max_records: int = 8) -> str:
    """Render private structured checks as concise natural language for the main model."""
    lines = ["Evidence validation note:"]
    atomic_records = [
        record for record in state.get("records", [])
        if record.get("subject") or record.get("value")
    ][-max_records:]
    if atomic_records:
        lines.append("Retained atomic facts:")
        for record in atomic_records:
            subject = record.get("subject") or "unspecified subject"
            relation = record.get("relation") or "fact"
            value = record.get("value") or record.get("claim")
            unit = f" {record.get('unit')}" if record.get("unit") else ""
            time_scope = (
                f"; time scope: {record.get('time_scope')}"
                if record.get("time_scope")
                else ""
            )
            lines.append(f"- {subject}: {relation} = {value}{unit}{time_scope}.")

    if state.get("shared_time_scope_required"):
        lines.append(
            "All comparison inputs must use the shared cutoff: "
            + (state.get("shared_time_scope") or "not yet established")
            + "."
        )

    validated = [
        requirement for requirement in state.get("requirements", [])
        if requirement.get("validated_result") is not None
    ]
    for requirement in validated:
        inputs = ", ".join(requirement.get("validated_inputs", []))
        lines.append(
            f"Program-checked {requirement.get('operation')} using {inputs}: "
            f"{requirement.get('validated_result')}."
        )

    if state.get("blocking_gaps"):
        lines.append("Blocking gaps: " + "; ".join(state["blocking_gaps"]))
    elif state.get("verification_risks"):
        lines.append(
            "Verification risks (not hard blockers): "
            + "; ".join(state["verification_risks"])
        )
    else:
        lines.append("No structural blocking gap was detected.")
    return "\n".join(lines)


def merge_evidence_packet(
    state: Dict,
    packet: Dict,
    max_records: int = 12,
    use_global_requirements: bool = False,
    use_risk_tiers: bool = False,
) -> int:
    existing_records = state.get("records", [])
    existing_keys = {_claim_key(record) for record in existing_records}
    record_by_key = {_claim_key(record): record for record in existing_records}
    used_ids = {
        _normalize_evidence_id(record.get("evidence_id"))
        for record in existing_records
        if _normalize_evidence_id(record.get("evidence_id"))
    }
    id_remap = {}
    added = 0
    for incoming_record in packet.get("new_evidence", []):
        record = dict(incoming_record)
        collection_key = record.get("collection_key", "none")
        if collection_key in VALID_COLLECTION_KEYS - {"none"}:
            collection = state["collections"].setdefault(collection_key, [])
            existing_items = {_item_key(item) for item in collection}
            for item in record.get("items", []):
                key = _item_key(item)
                if key and key not in existing_items:
                    collection.append(item)
                    existing_items.add(key)

        key = _claim_key(record)
        incoming_id = _normalize_evidence_id(record.get("evidence_id"))
        if not key:
            continue
        if key in existing_keys:
            existing_id = _normalize_evidence_id(
                record_by_key[key].get("evidence_id")
            )
            if incoming_id and existing_id:
                id_remap[incoming_id] = existing_id
            continue
        if use_risk_tiers:
            evidence_id = incoming_id
            if not evidence_id or evidence_id in used_ids:
                evidence_id = _next_evidence_id(used_ids)
            record["evidence_id"] = evidence_id
            used_ids.add(evidence_id)
            if incoming_id:
                id_remap[incoming_id] = evidence_id
        state["records"].append(record)
        record_by_key[key] = record
        existing_keys.add(key)
        added += 1

    if max_records > 0 and len(state["records"]) > max_records:
        state["records"] = state["records"][-max_records:]

    requirement_updates = 0
    if packet.get("parse_ok", False):
        if use_global_requirements:
            incoming_requirements = []
            for requirement in packet.get("requirements", []):
                normalized = dict(requirement)
                if use_risk_tiers:
                    normalized["supporting_evidence_ids"] = [
                        id_remap.get(_normalize_evidence_id(evidence_id), _normalize_evidence_id(evidence_id))
                        for evidence_id in requirement.get("supporting_evidence_ids", [])
                        if _normalize_evidence_id(evidence_id)
                    ]
                incoming_requirements.append(normalized)
            requirement_updates = _merge_global_requirements(
                state,
                incoming_requirements,
            )
        state["task_operation"] = packet.get("task_operation", "other")
        state["requires_complete_coverage"] = packet.get("requires_complete_coverage", False)
        state["coverage_status"] = packet.get("coverage_status", "unknown")
        state["missing_evidence"] = packet.get("missing_evidence", [])
        state["coverage_reason"] = packet.get("coverage_reason", "")
        state["suggested_query"] = packet.get("suggested_query", "")
    else:
        if packet.get("task_operation") != "other":
            state["task_operation"] = packet["task_operation"]
        state["requires_complete_coverage"] = (
            state.get("requires_complete_coverage", False)
            or packet.get("requires_complete_coverage", False)
        )
    state["packet_count"] += 1
    if not packet.get("parse_ok", False):
        state["parse_failures"] += 1

    if added or requirement_updates:
        state["stagnant_rounds"] = 0
    else:
        state["stagnant_rounds"] += 1
    return added


def render_prior_evidence(state: Dict, max_records: int = 10) -> str:
    records = state.get("records", [])[-max_records:]
    if not records and not state.get("requirements") and not state.get("collections"):
        return "No prior evidence has been retained."

    lines = ["Previously retained evidence:"]
    if state.get("requirements"):
        lines.append("Global answer requirements:")
        for requirement in state["requirements"]:
            lines.append(
                f"- {requirement.get('id')}: {requirement.get('description')} "
                f"[status={requirement.get('status')}]"
            )
    for collection_key, items in state.get("collections", {}).items():
        lines.append(
            f"Merged {collection_key} ({len(items)} items): "
            + "; ".join(items)
        )
    for index, record in enumerate(records, start=1):
        source = record.get("source_url") or "source URL unavailable"
        lines.append(
            f"{index}. {_clip(record.get('claim'), 320)} "
            f"[access={record.get('source_access')}, strength={record.get('strength')}, "
            f"source={_clip(source, 180)}]"
        )
    lines.append(f"Previous cumulative coverage: {state.get('coverage_status', 'unknown')}.")
    if state.get("missing_evidence"):
        lines.append("Previously missing: " + "; ".join(state["missing_evidence"]))
    return "\n".join(lines)


def render_private_evidence_state(state: Dict, max_records: int = 10) -> str:
    payload = {
        "schema_version": state.get("schema_version", "evidence-completeness-v1"),
        "task_operation": state.get("task_operation", "other"),
        "requires_complete_coverage": state.get("requires_complete_coverage", False),
        "collections": state.get("collections", {}),
        "requirements": state.get("requirements", []),
        "records": state.get("records", [])[-max_records:],
        "coverage_status": state.get("coverage_status", "unknown"),
        "missing_evidence": state.get("missing_evidence", []),
        "coverage_reason": state.get("coverage_reason", ""),
        "suggested_query": state.get("suggested_query", ""),
        "blocking_gaps": state.get("blocking_gaps", []),
        "verification_risks": state.get("verification_risks", []),
        "shared_time_scope_required": state.get("shared_time_scope_required", False),
        "shared_time_scope": state.get("shared_time_scope", ""),
        "verification_followups": state.get("verification_followups", 0),
    }
    return (
        "Private structured evidence state. Use it to maintain cumulative evidence; "
        "do not copy this JSON into MAIN_SUMMARY:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def render_evidence_update(state: Dict, added_count: int) -> str:
    records = state.get("records", [])[-8:]
    lines = ["Evidence update (maintained by the program; do not reproduce this block):"]
    if state.get("requirements"):
        lines.append("- Global answer requirements:")
        for requirement in state["requirements"]:
            lines.append(
                f"  {requirement.get('id')}: {requirement.get('description')} "
                f"[status={requirement.get('status')}]"
            )
    for collection_key, items in state.get("collections", {}).items():
        lines.append(
            f"- Merged {collection_key} ({len(items)} items): "
            + "; ".join(items)
        )
    if records:
        for index, record in enumerate(records, start=1):
            source = record.get("source_url") or "source unavailable"
            lines.append(
                f"- E{index}: {_clip(record.get('claim'), 650)} "
                f"[access={record.get('source_access')}; strength={record.get('strength')}; "
                f"source={_clip(source, 180)}]"
            )
    else:
        lines.append("- No usable factual evidence was extracted.")

    lines.append(
        "Coverage: "
        f"{state.get('coverage_status', 'unknown')} "
        f"(complete coverage required: {str(state.get('requires_complete_coverage', False)).lower()}; "
        f"new records this round: {added_count})."
    )
    if state.get("coverage_reason"):
        lines.append("Reason: " + state["coverage_reason"])
    if state.get("missing_evidence"):
        lines.append("Missing: " + "; ".join(state["missing_evidence"]))
    if state.get("suggested_query"):
        lines.append("Focused follow-up query: " + state["suggested_query"])
    if state.get("blocking_gaps"):
        lines.append("Blocking gaps: " + "; ".join(state["blocking_gaps"]))
    if state.get("verification_risks"):
        lines.append("Verification risks: " + "; ".join(state["verification_risks"]))

    if state.get("requires_complete_coverage") and state.get("coverage_status") != "complete":
        lines.append(
            "Decision: do not treat this partial collection as a complete list or exact count. "
            "Use one focused follow-up if it can fill the stated gap; otherwise give the best-supported answer."
        )
    else:
        lines.append("Decision: continue reasoning and answer when the retained evidence supports the requested result.")
    return "\n".join(lines)


def select_followup_query(
    state: Dict,
    executed_queries: Set[str],
    max_followups: int,
    stagnation_limit: int,
    use_risk_tiers: bool = False,
    max_verification_followups: int = 1,
) -> Tuple[Optional[str], Optional[str]]:
    followup_reason = "evidence_completeness_followup"
    if use_risk_tiers:
        has_blocking_gap = bool(state.get("blocking_gaps"))
        has_verification_risk = bool(state.get("verification_risks"))
        if not has_blocking_gap and not has_verification_risk:
            return None, None
        if has_blocking_gap:
            followup_reason = "evidence_blocking_followup"
        else:
            if state.get("verification_followups", 0) >= max_verification_followups:
                return None, "verification_limit_reached"
            followup_reason = "evidence_verification_followup"

    if not state.get("requires_complete_coverage"):
        return None, None
    if state.get("coverage_status") == "complete" and not use_risk_tiers:
        return None, None
    if state.get("forced_followups", 0) >= max_followups:
        return None, "followup_limit_reached"
    if state.get("stagnant_rounds", 0) >= stagnation_limit:
        return None, "evidence_stagnated"

    query = re.sub(r"\s+", " ", str(state.get("suggested_query", ""))).strip().strip("\"'`")
    if len(query) <= 5 or len(query) > 300:
        return None, "no_valid_suggested_query"
    normalized_executed = {re.sub(r"\s+", " ", q).strip().lower() for q in executed_queries}
    if query.lower() in normalized_executed:
        return None, "suggested_query_already_executed"
    return query, followup_reason


def override_complete_with_main_search(
    state: Dict,
    current_query: str,
    executed_queries: Set[str],
    search_count: int,
    max_search_limit: int,
) -> bool:
    """Downgrade complete evidence when the main model requests a new search."""
    if state.get("coverage_status") != "complete":
        return False

    query = re.sub(r"\s+", " ", str(current_query or "")).strip().strip("\"'`")
    if len(query) <= 5 or len(query) > 300:
        return False

    normalized_executed = {
        re.sub(r"\s+", " ", str(value)).strip().lower()
        for value in executed_queries
    }
    if query.lower() in normalized_executed:
        return False

    # The inherited loop increments search_count before applying its strict limit.
    if search_count + 1 >= max_search_limit:
        return False

    previous_reason = state.get("coverage_reason", "")
    state["coverage_status"] = "partial"
    state["coverage_reason"] = (
        "The auxiliary reader marked the evidence complete, but the main model "
        f"identified another unresolved information need: {query}"
    )
    missing = list(state.get("missing_evidence", []))
    main_model_gap = f"Main-model follow-up requested: {query}"
    if main_model_gap not in missing:
        missing.append(main_model_gap)
    state["missing_evidence"] = missing[-6:]
    state["suggested_query"] = query
    state["last_overridden_coverage_reason"] = previous_reason
    return True


def defer_early_complete(
    state: Dict,
    current_query: str,
    minimum_rounds: int,
) -> bool:
    completeness_sensitive = {"list", "count", "intersection", "comparison"}
    if minimum_rounds <= 1:
        return False
    if state.get("task_operation") not in completeness_sensitive:
        return False
    if state.get("coverage_status") != "complete":
        return False
    if state.get("packet_count", 0) >= minimum_rounds:
        return False

    state["coverage_status"] = "partial"
    state["coverage_reason"] = (
        f"A completeness-sensitive result needs at least {minimum_rounds} independent "
        "search rounds before complete coverage is accepted."
    )
    missing = list(state.get("missing_evidence", []))
    verification_gap = "Independent confirmation of the complete bounded collection"
    if verification_gap not in missing:
        missing.append(verification_gap)
    state["missing_evidence"] = missing
    if not state.get("suggested_query"):
        state["suggested_query"] = _clip(
            f"{current_query} official complete list independent confirmation",
            300,
        )
    return True
