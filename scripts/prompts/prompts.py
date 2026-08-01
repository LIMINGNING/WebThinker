
def get_gpqa_web_thinker_instruction(MAX_SEARCH_LIMIT=15):
    return """You are a reasoning assistant with the ability to perform web searches to help you answer the user's question accurately. You have special tools:

- To perform a search: write <|begin_search_query|>your query here<|end_search_query|>.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.

You can repeat the search process multiple times if necessary. Once you have all the information you need, continue your reasoning.

Example:
Question: "What is the energy range of pp III neutrinos?"
Thinking steps:
- I might need to look up details about pp III neutrinos.

<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>

(System returns processed information from relevant web pages)

Continues reasoning with the new information...

Remember:
- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.
- When done searching, continue your reasoning.
"""





def _benchmark_leakage_rule(enabled):
    if not enabled:
        return ""
    return (
        "- Never use benchmark datasets, answer keys, evaluation logs, solution dumps, "
        "or pages that reproduce the question with a reference answer as evidence.\n"
    )


def get_deep_web_explorer_instruction(
    search_query,
    search_intent,
    search_result,
    reject_benchmark_leakage=False,
):
    benchmark_leakage_rule = _benchmark_leakage_rule(reject_benchmark_leakage)
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query and search intent.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.
{benchmark_leakage_rule}

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent.
"""


def get_evidence_completeness_web_explorer_instruction(
    search_query,
    search_intent,
    search_result,
    prior_evidence,
    original_question="",
    max_evidence_records=8,
    evidence_delivery_mode="structured_update",
    reject_benchmark_leakage=False,
    use_global_requirements=False,
    use_risk_tiers=False,
):
    benchmark_leakage_rule = _benchmark_leakage_rule(reject_benchmark_leakage)
    summary_rules = ""
    summary_output = ""
    global_requirement_rules = ""
    global_requirement_schema = ""
    atomic_evidence_schema = ""
    risk_summary_rule = ""
    original_question_section = ""
    if use_global_requirements:
        global_requirement_rules = """
Global answer requirement rules:
- Judge completeness against the Original Question, not merely the Current search query.
- On the first packet, create 1 to 6 requirements that jointly cover every fact, relation, bounded collection, calculation input, source/version constraint, and output constraint needed to answer the Original Question.
- Use stable IDs r1, r2, and so on. In later packets, preserve every prior requirement ID and description. Update its status and supporting_claims; never delete an unresolved prior requirement merely because the current query concerns another subgoal.
- Set a requirement to supported only when cumulative retained evidence directly supports it. Use partial when only part of it is supported and missing when no usable support exists.
- Raw inputs do not by themselves support a derived comparison, intersection, count, or calculation requirement. Keep the derived requirement partial until the required operation can be completed from supported inputs.
- coverage_status may be complete only when requirements is non-empty, every requirement is supported, and missing_evidence is empty. Completion of the Current search query alone is not sufficient.
"""
        global_requirement_schema = """  "requirements": [
    {
      "id": "r1",
      "kind": "lookup|relation|coverage|calculation_input|source_constraint|output_constraint|other",
      "description": "one stable requirement needed to answer the original question",
      "status": "supported|partial|missing",
      "supporting_claims": ["short retained fact that supports this requirement"]
    }
  ],
"""
        if use_risk_tiers:
            global_requirement_rules = """
Global answer requirement rules:
- Judge completeness against the Original Question, not merely the Current search query.
- On the first packet, create 1 to 6 requirements that jointly cover every required fact, relation, bounded collection, calculation input, explicit time/source constraint, and output constraint.
- Use stable IDs r1, r2, and so on. Preserve every prior requirement ID and description in later packets; update its status and evidence links instead of deleting it.
- Set a requirement to supported only when cumulative retained evidence directly supports it. Use partial when only part is supported and missing when no usable support exists.

Risk-tier and scope rules:
- Preserve explicit or relative time cutoffs from the Original Question. Apply one shared cutoff to every side of a comparison. Do not replace a historical cutoff with current data.
- Phrases such as "as of", "through", and "by the end of" define a question-wide cutoff for every compared quantity, even when the second quantity does not repeat the date. Give every comparison input that same time_scope; current totals are invalid for a historical cutoff.
- Do not invent a historical cutoff when the Original Question has none. In that case use time_scope="" and scope_status=not_applicable.
- Use kind=time_constraint for an explicit cutoff and kind=derived_operation for max/min/count/difference results.
- Set scope_status=conflict when evidence is outside an explicit required time scope, unknown when its date cannot be checked, aligned when it matches, and not_applicable when no time scope applies.
- A derived_operation may be supported only when supporting_evidence_ids names every atomic input used. Use difference only for numeric subtraction and set_difference for collection exclusion; other operations are argmax, argmin, count, comparison, and calculation.
- For argmax or argmin over a complete table, scan all visible rows and retain separate atomic facts for at least the top two candidates by the requested field. Never infer the winner from row order or retain only the presumed winner.
- For other tables and rankings, extract separate atomic facts for every visible candidate that could change the result. Do not hide candidate values inside a prose conclusion.
- Treat missing required facts, missing calculation operands, explicit scope conflicts, and unsupported derivations as blocking gaps. Treat weak sources, uncertain collection coverage, and single-source confirmation as verification risks, not permanent blockers.
- A blocking gap must produce one focused suggested_query targeting the missing input. Verification risk should produce at most one focused suggested_query. If a verification query has already failed, leave suggested_query empty and provide the best-supported evidence with a caveat.
- coverage_status may be complete when no blocking gap remains. A source-strength or coverage verification risk may coexist with complete after one focused verification attempt.
"""
            global_requirement_schema = """  "requirements": [
    {
      "id": "r1",
      "kind": "lookup|relation|coverage|calculation_input|derived_operation|time_constraint|source_constraint|output_constraint|other",
      "description": "one stable requirement needed to answer the original question",
      "status": "supported|partial|missing",
      "time_scope": "explicit shared cutoff or empty",
      "scope_status": "aligned|conflict|unknown|not_applicable",
      "operation": "none|argmax|argmin|count|difference|set_difference|comparison|calculation",
      "supporting_evidence_ids": ["evidence IDs used by this requirement"],
      "supporting_claims": ["short retained fact that supports this requirement"]
    }
  ],
"""
            atomic_evidence_schema = """      "evidence_id": "stable e1, e2, and so on; never reuse an ID for a different fact",
      "subject": "entity whose atomic fact is recorded",
      "relation": "measured or asserted relation",
      "value": "atomic value as text, with no derived max/min/count conclusion",
      "unit": "unit or empty",
      "time_scope": "date or period supported by this fact, or empty",
      "supports_requirements": ["requirement IDs directly supported"],
"""
            risk_summary_rule = (
                "- Summarize retained atomic facts in natural language. Do not replace "
                "their values with an unsupported derived conclusion.\n"
            )
        original_question_section = f"""Original question:
{original_question}

"""
    if evidence_delivery_mode == "aux_summary":
        summary_rules = f"""
- Maintain the structured evidence packet as private working state. It will be retained for your next search round but will not be shown to the main model.
- Before the private packet, write a concise natural-language summary for the main model inside <MAIN_SUMMARY> tags.
- The summary must synthesize cumulative prior evidence plus useful current evidence. State relevant facts, source limitations, conflicts, and whether the requested scope is still incomplete.
- Do not expose JSON field names, collection keys, confidence labels, suggested_query, or the evidence packet in the summary.
- Do not present a weak or snippet-only candidate as verified. If it remains useful, describe it explicitly as an unverified lead rather than an answer item.
{risk_summary_rule}
"""
        summary_output = """<MAIN_SUMMARY>
Concise natural-language evidence summary for the main model. Do not include the private evidence packet.
</MAIN_SUMMARY>
"""
    return f"""You are an evidence extractor for a web-search question-answering agent.
Analyze only the supplied search pages and prior retained evidence. Do not start another search and do not click links. Put any useful next search in suggested_query instead.

Your job is to preserve facts while explicitly checking whether the cumulative evidence is complete enough for the requested answer operation.

Evidence rules:
- If Detailed Search Intent conflicts with Current Search Query, follow the Current Search Query and the Original Question.
- Copy the supplied provided_access value when it is present. page_excerpt means that only a local excerpt around the search snippet was supplied, not the full page.
- Use source_access=snippet_only when the claim comes from a snippet because the page could not be fetched.
- When a search snippet conflicts with fetched page content, treat the snippet as a weak lead and use the page content for retained facts.
- In flattened tables, catalogues, and rankings, first separate entry boundaries. Attach a grade, number, date, or label only to the subject in the same row or entry; never shift a value from the following entry to the preceding one.
- An empty field may support "not assigned" only when an authoritative, apparently exhaustive page uses a clear repeated schema and the subject's entry ends before the next entry without that field. State this layout basis in the claim. Otherwise keep the relation unresolved.
- Snippet-only evidence normally cannot prove that a list is complete. An exception is a snippet that directly states an authoritative total or clearly contains the entire bounded list.
- For list or count questions, complete means the full requested set or an authoritative direct total is available.
- For intersections, both input sets must be complete.
- For comparisons, all compared candidates and the requested comparison field must be available.
- Check explicit constraints such as time range, entity type, inclusion/exclusion, version, source, and output unit.
- Keep a fact in new_evidence only when it directly establishes a relation needed by the answer. For list and count tasks, a candidate must directly satisfy both membership/type and range constraints before it can be counted.
- Treat awards, category membership, commercial catalogue hits, related-page mentions, and chronology hints as leads unless they directly establish the required answer relation. Put the unresolved relation in missing_evidence instead of presenting the lead as an answer item.
{benchmark_leakage_rule}- Do not turn absent information into a negative fact. Do not infer omitted list members.
- coverage_status describes all prior retained evidence plus the new pages, not just the current page.
{global_requirement_rules}
- Return no more than {max_evidence_records} concise new evidence records. Do not repeat prior evidence.
- Keep each claim below 300 characters, coverage_reason below 160 characters, and missing_evidence to at most three short items. Never paste raw page text into a field.
- For list/count evidence, set collection_key=target_list and put each directly verified member in items. For intersections use left_set and right_set; for comparisons use candidates. Use collection_key=none and an empty items list for ordinary facts.
- For count operations, prefer an authoritative numeric total recorded as one atomic count fact. If deriving a count from members, put every complete member in items or emit one atomic record per member; never use the number of evidence records as the answer count.
- Keep item names canonical and concise. Include only directly supported members, preserve meaningful source order, and do not place explanatory sentences in items.
- suggested_query must target one concrete unresolved relation. Do not merely paraphrase the current query or repeat a broad query that has already failed. When a completeness-sensitive list first appears complete but no prior evidence exists, suggest one independent verification query.
{summary_rules}

Output exactly this structure without a Markdown code fence:

**Final Information**
{summary_output}<EVIDENCE_PACKET>
{{
  "task_operation": "single_fact|list|count|intersection|comparison|multi_hop|other",
  "requires_complete_coverage": true,
{global_requirement_schema}  "new_evidence": [
    {{
{atomic_evidence_schema}      "claim": "one short verifiable fact or one compact complete list",
      "source_url": "URL, or empty if unavailable",
      "source_access": "full_page|page_excerpt|snippet_only|search_result_only|inaccessible",
      "strength": "strong|medium|weak",
      "supports_constraints": ["question constraint supported by this fact"],
      "collection_key": "target_list|left_set|right_set|candidates|none",
      "items": ["directly supported canonical list member"]
    }}
  ],
  "coverage_status": "complete|partial|unknown",
  "missing_evidence": ["specific fact or coverage gap still needed"],
  "coverage_reason": "one concise reason for the status",
  "suggested_query": "one focused query for the most important gap, or empty when no useful query remains"
}}
</EVIDENCE_PACKET>

{original_question_section}Current search query:
{search_query}

Detailed search intent:
{search_intent}

Prior retained evidence:
{prior_evidence}

Current searched web pages:
{search_result}
"""




def get_web_page_reader_instruction(query, document):
    return f"""{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""

def get_detailed_web_page_reader_instruction(query, search_intent, document):
    return f"""Please provide all content related to the following search query and search intent from this document in markdown format.

Search Query: 
{query}

Search Intent: 
{search_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the search query and intent, do not omit any relevant information.
- Include any relevant links from the source
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the query/intent
"""


def get_search_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_click_intent_instruction(prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""



def get_query_plan_instruction(question):
    return f"""You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries.

Question: {question}

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""









def get_gpqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"What is the energy range of pp III neutrinos?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up details about pp III neutrinos.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>pp III neutrino energy spectrum<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_math_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"How do you compute the integral of e^(x^2) dx?\"\n"
        "Assistant thinking steps:\n"
        "- I might need to look up techniques for integrating e^(x^2).\n\n"
        "Assistant:\n"
        "<|begin_search_query|>methods to integrate e^(x^2)<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_code_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Find the minimum number of vertices in a Steiner tree that includes all specified vertices in a given tree.\"\n"
        "Assistant thinking steps:\n"
        "- I need to understand what a Steiner tree is and how to compute the minimum number of vertices required to include all specified vertices in a given tree.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Minimum Steiner Tree problem in trees<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_singleqa_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        "Example:\n"
        "Question: \"Who got the first Nobel Prize in Physics?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who was awarded the first Nobel Prize in Physics.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>first Nobel Prize in Physics winner<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_evidence_board_instruction():
    return (
        "Evidence Board requirement:\n"
        "- Maintain a concise structured evidence board throughout your reasoning.\n"
        "- Initialize it before your first search or before final answering if no search is needed.\n"
        "- After every <|begin_search_result|> ... <|end_search_result|>, update the board before continuing.\n"
        "- Keep at most 5 subgoals and at most 8 evidence cards. Merge duplicates.\n"
        "- Use this exact XML-like wrapper and valid JSON:\n\n"
        "<EVIDENCE_BOARD>\n"
        "{\n"
        "  \"subgoals\": [\n"
        "    {\"id\": \"S1\", \"description\": \"...\", \"status\": \"pending/done/missing\"}\n"
        "  ],\n"
        "  \"evidence_cards\": [\n"
        "    {\n"
        "      \"subgoal_id\": \"S1\",\n"
        "      \"fact\": \"short verifiable fact extracted from search results\",\n"
        "      \"source_url\": \"source URL if available, otherwise Web Page id/title\",\n"
        "      \"support_status\": \"supporting/partial/missing/conflicting/invalid\",\n"
        "      \"note\": \"why this evidence is useful or limited\"\n"
        "    }\n"
        "  ],\n"
        "  \"missing_evidence\": [\n"
        "    {\n"
        "      \"subgoal_id\": \"S1\",\n"
        "      \"missing\": \"key fact still needed before answering\",\n"
        "      \"next_query\": \"specific next search query to fill the gap\"\n"
        "    }\n"
        "  ],\n"
        "  \"answer_ready\": false,\n"
        "  \"final_answer_basis\": \"which evidence cards directly support the final answer\"\n"
        "}\n"
        "</EVIDENCE_BOARD>\n\n"
        "Before giving the final boxed answer, inspect the latest Evidence Board. "
        "If any key missing_evidence remains and search attempts remain, do not guess; issue the next search query from missing_evidence.next_query. "
        "When you need another search, do not leave the query only inside JSON; immediately after </EVIDENCE_BOARD>, output <|begin_search_query|>missing_evidence.next_query<|end_search_query|>. "
        "Only set answer_ready to true when the final answer can be directly derived from evidence_cards and required calculations/format constraints are satisfied. "
        "If the search limit is reached, answer only from the best supported evidence and do not invent missing facts.\n\n"
    )

def get_evidence_completeness_main_instruction(
    evidence_followup_limit=2,
    evidence_delivery_mode="structured_update",
    use_risk_tiers=False,
):
    if evidence_delivery_mode == "aux_summary":
        tiered_guidance = ""
        if use_risk_tiers:
            tiered_guidance = (
                "- The auxiliary summary is followed by a natural-language evidence validation note. "
                "Use its atomic values and program-checked operations when they conflict with an unsupported prose conclusion.\n"
                "- A blocking gap means a required operand or explicit scope constraint is unresolved. "
                "A verification risk is softer and receives at most one controller check; it does not justify endless searching.\n"
                "- Apply one explicit or relative time cutoff to every side of a comparison. "
                "Do not mix current evidence with a historical cutoff.\n"
            )
        return (
            "Evidence completeness guidance:\n"
            "- Search results are concise natural-language summaries written by the auxiliary web reader. Its private structured evidence state is intentionally not shown to you.\n"
            "- Use the summarized facts for reasoning, but preserve any stated uncertainty, conflict, source limitation, or incomplete coverage.\n"
            "- Do not assume that examples from a snippet form a complete list. Request another search when the summary identifies a material gap and search budget remains.\n"
            f"{tiered_guidance}"
            f"- The controller may insert at most {evidence_followup_limit} automatic follow-up searches for unresolved evidence gaps. These searches consume the same overall search limit stated above; they are not extra searches.\n"
            "- Searches you request also consume the overall search limit, but do not consume the controller's automatic follow-up sub-budget.\n"
            "- When search cannot improve the evidence or the overall search budget is exhausted, give a concrete best-supported short answer. You may briefly note uncertainty, but do not use 'unable to determine' as the final answer and do not emit another search request.\n\n"
        )
    return (
        "Evidence completeness guidance:\n"
        "- The program will provide compact Evidence update blocks after searches. Read them but do not reproduce them.\n"
        "- Distinguish complete evidence from partial or unknown coverage. A few examples from a snippet are not a complete list.\n"
        "- For exact lists, counts, intersections, and comparisons, verify that all required candidates and question constraints are covered before treating the result as exact.\n"
        "- Count only retained facts that directly satisfy the requested entity type, time range, and membership relation. Do not convert awards, catalogue appearances, or indirect mentions into answer items.\n"
        "- When the Evidence update contains Merged target_list, left_set, right_set, or candidates, use the merged collections across all rounds. Do not discard earlier members merely because a later source repeats only part of a list.\n"
        "- If coverage is partial and a focused follow-up query is provided, use it when it can resolve the key gap.\n"
        f"- The controller may insert at most {evidence_followup_limit} automatic follow-up searches for unresolved evidence gaps. These searches consume the same overall search limit stated above; they are not extra searches.\n"
        "- Searches you request also consume the overall search limit, but do not consume the controller's automatic follow-up sub-budget.\n"
        "- When search cannot improve the evidence or the overall search budget is exhausted, give a concrete best-supported short answer. You may briefly note uncertainty, but do not use 'unable to determine' as the final answer and do not emit another search request.\n\n"
    )


def get_multiqa_search_o1_instruction(
    MAX_SEARCH_LIMIT,
    use_evidence_board=False,
    use_evidence_completeness=False,
    evidence_followup_limit=2,
    evidence_delivery_mode="structured_update",
    use_risk_tiers=False,
):
    evidence_board_instruction = get_evidence_board_instruction() if use_evidence_board else ""
    evidence_completeness_instruction = (
        get_evidence_completeness_main_instruction(
            evidence_followup_limit,
            evidence_delivery_mode=evidence_delivery_mode,
            use_risk_tiers=use_risk_tiers,
        )
        if use_evidence_completeness
        else ""
    )
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you answer the user's question accurately. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        f"You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {MAX_SEARCH_LIMIT}.\n\n"
        "Once you have all the information you need, continue your reasoning.\n\n"
        f"{evidence_board_instruction}"
        f"{evidence_completeness_instruction}"
        "Example:\n"
        "Question: \"Alice David is the voice of Lara Croft in a video game developed by which company?\"\n"
        "Assistant thinking steps:\n"
        "- I need to find out who voices Lara Croft in the video game.\n"
        "- Then, I need to determine which company developed that video game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n\n"
    )

def get_timeline_search_o1_instruction(MAX_SEARCH_LIMIT):
    return (
        "You are a reasoning assistant with the ability to perform web searches to help "
        "you create an accurate chronological timeline summary. You have special tools:\n\n"
        "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
        "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
        "You should perform multiple searches to gather comprehensive information until you believe you have enough details.\n"
        "Finally, provide a comprehensive timeline that includes all relevant events in chronological order.\n\n"
        "Example:\n"
        "Text: \"Create a timeline of key events in the Apollo 11 mission.\"\n"
        "Assistant thinking steps:\n"
        "- I need to find key dates and events of the Apollo 11 mission.\n\n"
        "Assistant:\n"
        "<|begin_search_query|>Apollo 11 mission timeline key events dates<|end_search_query|>\n\n"
        "(System returns processed information from relevant web pages)\n\n"
        "Assistant continues reasoning with the new information...\n\n"
        "Remember:\n"
        "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
        "- When done searching, continue your reasoning.\n"
        "- You should perform as many searches as possible to gather comprehensive information.\n\n"
    )



def get_naive_rag_instruction(question, documents):
    return (
        "You are a knowledgeable assistant that uses the provided documents to answer the user's question.\n\n"
        "Question:\n"
        f"{question}\n"
        "Documents:\n"
        f"{documents}\n"
    )



def get_task_instruction_openqa(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following question.\n\n'
            'Provide your final answer in the format **ANSWER: {YOUR_ANSWER}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_math(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following math question. '
            'You should provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following math question.\n\n'
            'Provide your final answer in the format **ANSWER: YOUR_ANSWER**.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following math question. You should think step by step to solve it.\n\n'
            'Provide your final answer in the format \\boxed{YOUR_ANSWER}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_multi_choice(question, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Please answer the following multiple-choice question. '
            'You should provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'dpsk':
        user_prompt = (
            'Please answer the following multiple-choice question.\n\n'
            'Provide your final choice in the format **ANSWER: {YOUR_CHOICE}**.\n\n'
            f'Question:\n{question}\n\n'
        )
    elif model_name == 'llama':
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}. Your final choice should be one of the letters A, B, C, or D, DO NOT include any answer content.\n\n'
            f'Question:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'Please answer the following multiple-choice question. You should think step by step to solve it.\n\n'
            'Provide your final choice in the format \\boxed{YOUR_CHOICE}.\n\n'
            f'Question:\n{question}\n\n'
        )
    return user_prompt

def get_task_instruction_code(question, question_title=None, model_name=None):
    if model_name == 'qwq':
        user_prompt = (
            'Generate a correct Python program that passes all tests for the given problem. '
            'You should provide your final code within a Python code block using triple backticks (```python\n'
            'YOUR_CODE\n'
            '```).\n\n'
            f'Problem Title: {question_title}\n\n'
            f'Problem Statement:\n{question}\n\n'
        )
    else:
        user_prompt = (
            'You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. '
            f'You should think step by step to solve it.\n\nQuestion:\n{question}\n\n'
            'Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows.\n\n'
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return user_prompt

def get_task_instruction_timeline(text, model_name=None):
    # Common format template for both cases
    format_template = '- [DATE/TIME]: Event description\n\n'
    # Base prompt that's shared between both cases
    base_prompt = f'Text:\n{text}\n\n'
    if model_name == 'qwq':
        return (
            'Now it is March 14, 2025. Please create a comprehensive timeline based on the given text.'
            f'Format each event as:\n{format_template}'
            'Ensure events are ordered chronologically and include specific dates/times when available.\n\n'
            f'{base_prompt}'
        )
    else:
        return (
            'Please summarize the key events from the text in chronological order. '
            'For each event, include the date/time (if available) and a clear description.\n\n'
            f'Format your timeline as:\n{format_template}'
            f'{base_prompt}'
        )

