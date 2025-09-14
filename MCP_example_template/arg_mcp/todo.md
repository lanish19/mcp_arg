# TODO.md — Argumentation Analysis MCP Connector

**Objective:** Deliver a robust, predictable, developer-friendly argument analysis service. This checklist provides complete, unambiguous instructions for an AI-coder to implement. Follow items in priority order.

---

## 0) Global Goals & Constraints

* **No silent failures.** All endpoints must return structured errors with remediation hints.
* **Traceability.** Every detected claim/pattern must include source text spans.
* **Stable I/O.** Versioned JSON Schemas for requests and responses; never omit top-level fields.
* **Useful by default.** When inputs are incomplete, apply safe fallbacks (auto pattern detection, minimal probe plans).
* **Ontology clarity.** Correct encoding, consistent hierarchy, synonyms coverage.

---

## 1) API VERSIONING & SCHEMAS

### 1.1 Introduce `/v1` base path and deprecate legacy

* Create **aliases** so existing routes continue to work for one minor version.
* Add a `version` field in all responses, e.g., `"version": "v1.1.0"`.

### 1.2 JSON Schemas

Create **schema files** under `schemas/`:

* `schemas/v1/analyze_argument_comprehensive.response.json`
* `schemas/v1/decompose_argument_structure.response.json`
* `schemas/v1/detect_argument_patterns.response.json`
* `schemas/v1/generate_missing_assumptions.response.json`
* `schemas/v1/orchestrate_probe_analysis.response.json`
* `schemas/v1/construct_argument_graph.response.json`
* `schemas/v1/validate_argument_graph.response.json`
* `schemas/v1/identify_reasoning_weaknesses.response.json`
* `schemas/v1/assess_argument_quality.response.json`
* `schemas/v1/ontology_*.response.json`
* For each request, mirror a `.request.json`.

**Common object types (re-use via `$ref`):**

```json
// schemas/v1/common/Node.json
{
  "$id": "Node",
  "type": "object",
  "required": ["node_id", "node_type", "content", "confidence"],
  "properties": {
    "node_id": {"type":"string"},
    "node_type": {"type":"string", "enum": ["STATEMENT","ASSUMPTION","MAIN_CLAIM","EVIDENCE","OTHER"]},
    "primary_subtype": {"type":["string","null"]},
    "content": {"type":"string"},
    "confidence": {"type":"number", "minimum": 0, "maximum": 1},
    "source_text_span": {
      "type": ["object","null"],
      "properties": {"start":{"type":"integer"},"end":{"type":"integer"}},
      "required": ["start","end"]
    },
    "assumptions": {"type":"array", "items":{"type":"string"}}
  }
}
```

```json
// schemas/v1/common/Pattern.json
{
  "$id": "Pattern",
  "type": "object",
  "required": ["pattern_id","pattern_type","label","confidence"],
  "properties": {
    "pattern_id":{"type":"string"},
    "pattern_type":{"type":"string","enum":["authority","causal","analogical","normative","other"]},
    "label":{"type":"string"},
    "confidence":{"type":"number","minimum":0,"maximum":1},
    "details":{
      "type":"object",
      "properties":{
        "scheme":{"type":["string","null"]},
        "score":{"type":["number","null"]},
        "trigger":{"type":["string","null"]},      // short text snippet or regex
        "roles":{"type":["object","null"]}         // role→text mapping if applicable
      }
    },
    "source_text_span": {"$ref":"Node#/properties/source_text_span"}
  }
}
```

**Response envelope contract (all endpoints):**

```json
{
  "version": "v1.1.0",
  "data": { /* endpoint-specific payload */ },
  "metadata": {
    "schema_url": "https://<host>/schemas/v1/<endpoint>.response.json",
    "elapsed_ms": 0,
    "warnings": [],
    "next_steps": []
  },
  "error": null
}
```

On error:

```json
{
  "version":"v1.1.0",
  "data": null,
  "metadata": { "elapsed_ms": 3, "warnings": [] },
  "error": {
    "code": "ENGINE_METHOD_MISSING",
    "message": "InferenceEngine.analyze is not implemented",
    "hint": "Use AnalysisEngine.stage3_infer or implement a compatibility wrapper",
    "where": "generate_missing_assumptions"
  }
}
```

---

## 2) CRITICAL BUGFIXES

### 2.1 `generate_missing_assumptions` crash

**Problem:** `'InferenceEngine' object has no attribute 'analyze'`.

**Fix (choose one and apply consistently):**

* **Option A (Wrapper):** Implement `InferenceEngine.analyze()` that delegates to `AnalysisEngine.stage3_infer(components, ...)`.
* **Option B (Refactor callers):** Replace all invocations of `GapAnalyzer.analyze(...)` with `AnalysisEngine.stage3_infer(...)`.

**Acceptance tests:**

* Given only `{ main_claim, supporting_claims[] }` → returns non-empty `assumptions[]` or a helpful empty-set explanation listing what inputs are missing.
* Mixed inputs (argument text + patterns) → merges both.
* Invalid shapes → structured error (see schema above).

### 2.2 `identify_reasoning_weaknesses` returns `[]` on valid inputs

**Fixes:**

1. **Auto-fallback:** If `argument_analysis.patterns` missing but `argument_text` present → auto run `detect_argument_patterns`.
2. **Threshold controls:** Add optional `sensitivity` param (`"low" | "default" | "high"`), adjusting confidence cutoffs (e.g., 0.1 / 0.2 / 0.3).
3. **Stable fields:** The analyzer must **not** depend on `details.match`; change to `details.trigger` (short snippet) and always attempt to populate it.
4. **At minimum** return candidate weaknesses with `"confidence":"low"` instead of empty set.

**Acceptance tests:**

* For text featuring **limited options** and **heavy analogy**, the weaknesses include at least: `False Dilemma/Trilemma`, `Analogy Risk`, `Causal Necessity Overstatement`, with node references and rationales.

### 2.3 `orchestrate_probe_analysis` yields empty `probe_plan`

**Fixes:**

* Implement **default probe sets** keyed to top-level pattern types. If no pattern passes threshold, return:

  * `["Assumption Audit", "Counterexample Search", "Standards-of-Proof Gatekeeper"]`
* Add parameter: `"max_probes": 5` and `"min_probes": 3`.

**Acceptance tests:**

* With any single detected pattern, returns ≥3 probes, each with `why`, `how`, `targets: [node_ids]`.

---

## 3) TRACEABILITY (MANDATORY)

### 3.1 Preserve `source_text_span` everywhere

* In detection/decomposition engines, ensure `(start,end)` char indices are computed and **retained**.
* **DO NOT** discard spans during compaction.
* For multi-sentence patterns, return the span of the **trigger** and the span of the **full segment** (array allowed).

**Acceptance tests:**

* Every `node` and `pattern` has a non-null span referencing the original `argument_text`.
* Spans map correctly when rendered over the input text (no off-by-one).

---

## 4) ONTOLOGY DATA HYGIENE

### 4.1 Fix UTF-8 encoding & punctuation

* Normalize all ontology CSV/JSON inputs to **UTF-8 (no BOM)**.
* Replace mojibake (e.g., `â`, `Ò...Ó`) with proper characters.
* On load, run a sanitizer:

  * Convert smart quotes to ASCII `' "`.
  * Collapse whitespace.
  * Lowercase **search index** fields.

**Acceptance tests:**

* Round-trip a few entries containing quotes/apostrophes.
* Searching for “fallacist’s fallacy” works; no mojibake remains.

### 4.2 Disentangle bucket/category

* Some buckets combine multiple categories. Implement:

  * `ontology_list_buckets(dimension, category)` → returns buckets **scoped** to the category.
  * `ontology_bucket_detail(bucket_name, category?)` → if category given, **filter** entries to that category.

**Acceptance tests:**

* `dimension=Fallacy, category=False Dilemma` does **not** return Straw Man or Slippery Slope items.

### 4.3 Synonym map & query normalization

* Add `synonyms.json`, e.g.:

```json
{
  "false trilemma": "false dilemma",
  "appeal to authority": "ad verecundiam",
  "post hoc": "post hoc ergo propter hoc"
}
```

* Preprocess all ontology queries via synonyms.
* For `ontology_semantic_search`, return `metadata.applied_synonyms`.

**Acceptance tests:**

* Searching “false trilemma” returns False Dilemma reliably (score ≥ 0.6, tunable).

---

## 5) INPUT FALLBACKS & ERGONOMICS

### 5.1 Auto pattern detection

* Any endpoint receiving `argument_text` but lacking `patterns` should auto-compute patterns internally (visible in `metadata.warnings` as an implicit step).

### 5.2 Non-empty defaults

* `orchestrate_probe_analysis` must **never** return empty `probe_plan` (unless a hard error occurred).
* `identify_reasoning_weaknesses` must prefer **low-confidence** items to empty arrays (opt-out with `sensitivity:"high"`).

### 5.3 Case/punctuation-insensitive tool lookup

* Normalize names to slugs internally (e.g., `the assumption audit™` → `assumption-audit`).
* `tools_get` accepts any case/punctuation variant.

---

## 6) ENDPOINT ADDITIONS

### 6.1 `map_assumptions_to_nodes` (NEW)

**Route:** `POST /v1/argument/map_assumptions_to_nodes`
**Request:**

```json
{
  "analysis_results": { /* from decompose/analyze */ },
  "assumptions": [{ "text": "Frozen conflicts are inherently unstable." }],
  "strategy": "best-match"  // or "strict"
}
```

**Response:**

```json
{
  "data": {
    "mappings": [
      {"assumption":"...", "node_id":"N_c0727256", "score":0.83}
    ],
    "unmapped": []
  },
  "metadata": { "next_steps": ["export_graph"] },
  "error": null
}
```

### 6.2 `export_graph` (NEW)

**Route:** `GET /v1/argument/export_graph?format=mermaid|graphviz|jsonld&analysis_id=...`
**Response:** `{ "data": {"format":"mermaid","content":"graph TD; ..."} }`

### 6.3 `analyze_and_probe` (NEW one-shot)

**Route:** `POST /v1/argument/analyze_and_probe`
**Request:**

```json
{
  "argument_text": "...",
  "analysis_depth": "comprehensive",
  "audience": "policy",
  "goal": "strengthen",
  "confidence_threshold": 0.2
}
```

**Response:** returns unified `structure`, `patterns`, `assumptions`, `weaknesses`, `probe_plan`, plus `next_steps`.

---

## 7) VALIDATION & QUALITY ASSESSMENT

### 7.1 `validate_argument_graph` improvements

* Add checks:

  * Missing or multiple main claims.
  * Orphan nodes; cycles; duplicate edges.
  * No `source_text_span` on nodes/patterns.
  * No assumptions / no evidence types.
* Return `suggestions[]` and `next_steps[]`.

### 7.2 `assess_argument_quality` scoring

* Compute sub-scores (0–1):

  * **Coverage** (claims vs. supports vs. assumptions present)
  * **Balance** (presence of counter-considerations)
  * **Clarity** (node length/complexity; jargon density)
  * **Support rigor** (evidence type diversity; warrant articulation)
* Include rubric and weights in `metadata`.

---

## 8) ANALYSIS QUALITY IMPROVEMENTS

### 8.1 Pattern details completeness

* Always set `details.scheme`, `details.trigger`, and `source_text_span`.
* For analogies, include `"roles": {"base": "...", "target": "..."}` when feasible.

### 8.2 Assumption generation breadth

Expand checks in the assumption engine to include:

* **Limited options** (False Dilemma/Trilemma) → “Are hybrid/third options considered?”
* **Analogy** → “List similarity/difference dimensions; scope limits.”
* **Causality** → “Mechanism present?”, “Temporal order?”, “Confounders addressed?”
* **Leadership ⇒ Stability** necessity claims → “Could leadership increase confrontation?”
* **Frozen conflict ⇒ Instability** inevitability → “What stabilizing pathways exist?”

**Output shape example:**

```json
{
  "assumptions": [
    {
      "text": "Hybrid solutions beyond reform/new/force are negligible.",
      "linked_patterns": ["scheme_22"],
      "impact": "high",
      "confidence": 0.5,
      "tests": ["List hybrid models", "Check regional coalition cases"]
    }
  ]
}
```

---

## 9) PROBE PLANNING (DEFAULTS)

### 9.1 Default probe sets

* If any **dilemma/limited options** pattern:

  * `Assumption Audit`, `Counterexample Search`, `MECE Pyramid Structurer`
* If **analogy** pattern:

  * `Analogy Fidelity Filter`, `Standards-of-Proof Gatekeeper`, `External Validity Stress Test`
* If **authority** pattern:

  * `Credibility Graph Builder`, `Standards-of-Proof Gatekeeper`, `Disconfirmation Bias Buster`
* If **causal** pattern:

  * `Causal ID Checklist`, `Counterfactual Contrast Test`, `Boundary Condition Stress Test`

Include `targets: [node_ids]` and `why/how` fields.

---

## 10) ONTOLOGY & SEARCH

### 10.1 `ontology_list_dimensions` / `categories` / `buckets`

* Return objects, not bare strings:

```json
{"data":{"dimensions":[{"name":"Fallacy","count":123}, ...]}}
```

* For categories/buckets include counts and **parent references**.

### 10.2 `tools_list` / `tools_search` / `tools_get`

* Add structured tags: `phase`, `theme`, `requires` (prereqs).
* `tools_search` supports `tags:any/all`, `text`, and synonyms.
* `tools_get` accepts **slug** (normalized) and case-insensitive names.

---

## 11) ERROR MODEL & LOGGING

### 11.1 Error codes (standardize)

* `ENGINE_METHOD_MISSING`
* `INVALID_INPUT_SHAPE`
* `MISSING_ARGUMENT_TEXT`
* `ONTOLOGY_LOOKUP_FAILURE`
* `UNSUPPORTED_FORMAT`
* `INTERNAL_ERROR`

### 11.2 Logging

* Add structured logs (JSON) with `endpoint`, `elapsed_ms`, `input_size`, and `result_size`.
* Redact large `argument_text` after hashing for correlation.

---

## 12) TESTS

### 12.1 Unit tests

* Engines: pattern detection, assumption generation, probe planner.
* Ontology I/O: encoding sanitizer, synonyms expansion.
* Graph: construct, validate (with all new checks), export.

### 12.2 Integration tests (goldens)

* **Geopolitical sample** (the argument from our earlier testing):

  * Must produce non-empty assumptions: false trilemma, analogy scope, causal necessity.
  * `identify_reasoning_weaknesses` returns ≥3 items, each linked to `node_id`.
  * `orchestrate_probe_analysis` returns ≥3 probes.

### 12.3 Fuzz/error tests

* Missing fields, wrong types → structured errors with hints.
* Extremely long texts → performance budget under N ms (set a benchmark).

---

## 13) PERFORMANCE

* Cache ontology vectorizer/embeddings in memory (recompute only on reload).
* Add `reload_ontology` (dev-only) endpoint.
* Limit top-N outputs (`patterns ≤ 8`, `probes ≤ 5`) unless `debug=true`.

---

## 14) DOCUMENTATION

* Publish a **walkthrough**: text → decompose → assumptions → map → weaknesses → probe → export (with copy-paste payloads).
* Add a **Contracts** page linking to the JSON Schemas.
* Include **migration notes** describing breaking changes (e.g., `details.match → details.trigger`).

---

## 15) ROLLOUT PLAN

1. **Phase 1 (Core fixes):** §2 (crashes), §3 (spans), §5 (fallbacks), §9 (default probes).
2. **Phase 2 (Ontology):** §4 (encoding, buckets, synonyms).
3. **Phase 3 (APIs):** §1 (versioning), §6 (new endpoints), §7 (validation/quality).
4. **Phase 4 (Docs & Tests):** §12 (goldens), §14 (docs).
5. **Phase 5 (Perf & polish):** §13 (caching), error model (§11).

---

## 16) DONE WHEN (acceptance checklist)

* [ ] No endpoint returns empty top-level arrays unless `sensitivity:"high"` was requested.
* [ ] All nodes/patterns include valid `source_text_span`.
* [ ] `generate_missing_assumptions` never crashes; returns structured assumptions or actionable error.
* [ ] `identify_reasoning_weaknesses` surfaces expected items for the geopolitical sample.
* [ ] `orchestrate_probe_analysis` always returns ≥3 probes (unless error).
* [ ] Ontology responses are UTF-8 clean; bucket/category scoping works.
* [ ] Searching “false trilemma” yields False Dilemma.
* [ ] JSON Schemas published; all responses include `schema_url` and `version`.
* [ ] `export_graph` produces valid Mermaid output for any analysis id.
* [ ] `validate_argument_graph` flags structural issues & suggests next steps.
* [ ] CI runs unit + integration + fuzz tests and enforces the above.

---

### Notes for the AI-Coder

* Use the provided schemas as source of truth; reject or coerce invalid inputs.
* Prefer pure functions in engines; keep I/O at the boundary layer.
* Keep **deterministic** ordering (sort by score then by node id) for test stability.
* Include **confidence** everywhere; calibrate thresholds and document defaults.

This plan is intentionally exhaustive; implement in the specified order and you’ll convert the current tool into a production-grade, reliable analysis service.
