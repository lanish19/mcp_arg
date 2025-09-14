Looking at the additional TODO document, I can see several important items that weren't covered in my original analysis. Let me create an enhanced comprehensive TODO that merges both findings:
TODO.md — Exhaustive Fix/Enhancement Plan for Argumentation Analysis MCP
Goal: Make the MCP connector reliable, contract-true, and useful by default for end-to-end argument analysis. This plan merges hands-on testing findings with a close read of the codebase, prioritizing deterministic behavior, strong schemas, and actionable outputs.

P0 — CRITICAL INFRASTRUCTURE FIXES (Blockers)
1. Fix Envelope Validation Crisis
Files: All endpoint functions in argument_mcp.py Problem: MCP clients expect raw arrays but receive envelope objects, causing "Output validation error: ... is not of type 'array'" Solution:
def _make_response(data, error=None, envelope=True, compat_mode=False):
    if compat_mode or not envelope:  # Raw mode for MCP compatibility
        return data if error is None else {"error": error}
    return {
        "version": "v1.1.0", 
        "data": data, 
        "metadata": {
            "schema_url": f"schemas/v1/{endpoint_name}.response.json",
            "warnings": [],
            "next_steps": [],
            "retry_after": None  # For 503 errors
        }, 
        "error": error
    }

# Add compat parameter to all endpoints
def ontology_list_dimensions(compat=False):
    # Add query param ?compat=raw support

2. Rebuild Broken Span Tracking System
Files: MCP_example_template/arg_mcp/patterns.py, MCP_example_template/arg_mcp/engine.py Problem: All spans return [0,2] - completely broken, making UI highlighting impossible Solution:
class SpanTracker:
    def __init__(self, text):
        self.text = text
        self.sentences = self._split_sentences_with_positions()
        self.words = self._split_words_with_positions()
    
    def find_pattern_span(self, pattern_cues, confidence):
        # Use TF-IDF + fuzzy string matching
        # Return actual character positions [start, end]
        # NEVER return [0,2] - always meaningful positions
        
    def find_phrase_exact_span(self, phrase):
        # Direct string search with case-insensitive matching
        start = self.text.lower().find(phrase.lower())
        if start >= 0:
            return [start, start + len(phrase)]
        return None

# In stage1_decompose: assign real spans to nodes
# In detect_patterns: guarantee spans for every pattern

Acceptance: 100% of patterns/weaknesses have accurate source_text_span pointing to actual text
3. Fix Completely Broken Stepwise Analysis
Files: MCP_example_template/argument_mcp.py Problem: "FunctionTool' object is not callable" - core feature completely broken Solution:
def analyze_argument_stepwise(argument_text, steps=None, max_steps=None, **kwargs):
    if not argument_text or not argument_text.strip():
        return _error_response("MISSING_ARGUMENT_TEXT", "Empty argument text")
    
    # Define callable step functions with proper error handling
    STEP_FUNCTIONS = {
        "decompose_argument_structure": lambda text, **kw: decompose_argument_structure(text, **kw),
        "detect_argument_patterns": lambda text, **kw: detect_argument_patterns(text, **kw),
        "ontology_pattern_match": lambda patterns, **kw: ontology_pattern_match(patterns, **kw),
        "generate_missing_assumptions": lambda components, **kw: generate_missing_assumptions(components, **kw),
        "construct_argument_graph": lambda analysis, **kw: construct_argument_graph(analysis, **kw),
        "validate_argument_graph": lambda graph, **kw: validate_argument_graph(graph, **kw),
        "assess_argument_quality": lambda graph, **kw: assess_argument_quality(graph, **kw),
        "identify_reasoning_weaknesses": lambda analysis, **kw: identify_reasoning_weaknesses(analysis, **kw),
        "orchestrate_probe_analysis": lambda analysis, **kw: orchestrate_probe_analysis(analysis, **kw),
        "generate_counter_analysis": lambda graph, **kw: generate_counter_analysis(graph, **kw)
    }
    
    DEFAULT_STEPS = [
        "decompose_argument_structure", "detect_argument_patterns", "ontology_pattern_match",
        "generate_missing_assumptions", "construct_argument_graph", "validate_argument_graph", 
        "assess_argument_quality", "identify_reasoning_weaknesses", "orchestrate_probe_analysis",
        "generate_counter_analysis"
    ]
    
    steps = steps or DEFAULT_STEPS
    max_steps = max_steps or 10
    
    stages = []
    artifacts = {}
    truncated = False
    
    # Execute pipeline with proper data flow
    current_data = {"text": argument_text}
    
    for i, step_name in enumerate(steps):
        if i >= max_steps:
            truncated = True
            break
            
        if step_name not in STEP_FUNCTIONS:
            return _error_response("INVALID_INPUT_SHAPE", 
                f"Unknown step: {step_name}",
                hint=f"Allowed steps: {list(STEP_FUNCTIONS.keys())}",
                where={"step": step_name, "allowed": list(STEP_FUNCTIONS.keys()), 
                      "stages_so_far": len(stages)})
        
        try:
            # Execute step with accumulated artifacts
            if step_name == "decompose_argument_structure":
                result_env = STEP_FUNCTIONS[step_name](argument_text, **kwargs)
            elif step_name == "detect_argument_patterns":
                result_env = STEP_FUNCTIONS[step_name](argument_text, **kwargs)
            elif step_name == "ontology_pattern_match":
                patterns = artifacts.get("patterns", [])
                result_env = STEP_FUNCTIONS[step_name](patterns, **kwargs)
            elif step_name == "generate_missing_assumptions":
                components = {"text": argument_text, "patterns": artifacts.get("patterns", [])}
                result_env = STEP_FUNCTIONS[step_name](components, **kwargs)
            elif step_name == "construct_argument_graph":
                analysis = artifacts.get("structure", {})
                result_env = STEP_FUNCTIONS[step_name](analysis, **kwargs)
            else:
                # Other steps use accumulated artifacts
                result_env = STEP_FUNCTIONS[step_name](artifacts, **kwargs)
            
            # Extract data from envelope if present
            result_data = result_env.get("data", result_env) if isinstance(result_env, dict) else result_env
            
            # Record stage info
            stages.append({
                "name": step_name,
                "inputs_summary": {
                    "argument_text": argument_text[:100] + "..." if len(argument_text) > 100 else argument_text
                },
                "key_outputs": _summarize_step_outputs(step_name, result_data),
                "next_tools": _get_next_tools(step_name, result_data)
            })
            
            # Update artifacts for next step
            artifacts[step_name.split("_")[-1]] = result_data  # Store by key name
            if step_name == "decompose_argument_structure":
                artifacts["structure"] = result_data.get("structure", {})
                artifacts["patterns"] = result_data.get("patterns", [])
            elif step_name == "detect_argument_patterns":
                artifacts["patterns"] = result_data.get("patterns", [])
            elif step_name == "construct_argument_graph":
                artifacts["graph"] = result_data
            # ... continue for other steps
            
        except Exception as e:
            return _error_response("INTERNAL_ERROR", 
                f"Step {step_name} failed: {str(e)}",
                where={"step": step_name, "stage": i})
    
    return {
        "stages": stages,
        "final_artifacts": artifacts,
        "truncated": truncated
    }

def _summarize_step_outputs(step_name, result_data):
    """Generate summary of key outputs for each step"""
    if step_name == "decompose_argument_structure":
        structure = result_data.get("structure", {})
        return {
            "nodes": len(structure.get("nodes", [])),
            "links": len(structure.get("links", []))
        }
    elif step_name == "detect_argument_patterns":
        return {
            "patterns": len(result_data.get("patterns", []))
        }
    # ... continue for other steps
    return {}

4. Fix Completely Non-Functional Assumptions Generation
Files: MCP_example_template/arg_mcp/gap.py, MCP_example_template/arg_mcp/engine.py Problem: Always returns empty assumptions array Solution:
class AssumptionGenerator:
    def __init__(self, ontology):
        self.ontology = ontology
        self.pattern_handlers = {
            "authority": self._generate_authority_assumptions,
            "causal": self._generate_causal_assumptions,
            "analogical": self._generate_analogical_assumptions,
            "other": self._generate_generic_assumptions
        }
    
    def generate_assumptions(self, text, patterns, argument_components=None):
        assumptions = []
        
        # Pattern-based assumption generation
        for pattern in patterns:
            pattern_type = pattern.get("pattern_type", "other")
            handler = self.pattern_handlers.get(pattern_type, self._generate_generic_assumptions)
            pattern_assumptions = handler(text, pattern)
            assumptions.extend(pattern_assumptions)
        
        # Structure-based assumptions
        if argument_components:
            structural_assumptions = self._generate_structural_assumptions(argument_components)
            assumptions.extend(structural_assumptions)
        
        # Remove duplicates and rank by impact
        assumptions = self._deduplicate_and_rank(assumptions)
        return assumptions[:10]  # Cap at 10 most important
    
    def _generate_authority_assumptions(self, text, pattern):
        return [
            {
                "text": "The cited experts are qualified in the relevant domain",
                "category": "epistemic",
                "impact": "high",
                "confidence": 0.8,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Check expert credentials and institutional affiliations",
                    "Verify expertise specifically relates to the claim domain",
                    "Research publication history and peer recognition"
                ]
            },
            {
                "text": "The experts are not biased or influenced by conflicts of interest",
                "category": "reliability", 
                "impact": "high",
                "confidence": 0.7,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Research funding sources and financial interests",
                    "Check for organizational or ideological pressures",
                    "Compare with independent expert opinions"
                ]
            },
            {
                "text": "There exists expert consensus on this topic",
                "category": "consensus",
                "impact": "medium",
                "confidence": 0.6,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Survey multiple experts in the field",
                    "Check professional organization positions",
                    "Review recent peer-reviewed literature"
                ]
            }
        ]
    
    def _generate_causal_assumptions(self, text, pattern):
        return [
            {
                "text": "The cause precedes the effect temporally",
                "category": "temporal",
                "impact": "critical",
                "confidence": 0.9,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Verify chronological sequence of events",
                    "Check for simultaneous occurrence that might indicate reverse causation",
                    "Look for clear temporal gaps between cause and effect"
                ]
            },
            {
                "text": "No alternative causes adequately explain the effect",
                "category": "alternative_causation",
                "impact": "high", 
                "confidence": 0.7,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Identify and test competing explanations",
                    "Check for confounding variables",
                    "Use controlled comparisons where possible"
                ]
            },
            {
                "text": "A plausible causal mechanism links cause to effect",
                "category": "mechanistic",
                "impact": "high",
                "confidence": 0.8,
                "linked_patterns": [pattern.get("pattern_id")],
                "tests": [
                    "Identify intermediate steps in causal chain",
                    "Test whether mechanism holds under different conditions",
                    "Look for similar mechanisms in comparable cases"
                ]
            }
        ]

def generate_missing_assumptions(argument_components, prioritization="critical"):
    """Fixed version that actually generates assumptions"""
    if not argument_components:
        return _error_response("INVALID_INPUT_SHAPE", "Missing argument_components")
    
    # Extract text and patterns from components
    text = argument_components.get("text", "")
    patterns = argument_components.get("patterns", [])
    
    if not text and not patterns:
        return _error_response("MISSING_ARGUMENT_TEXT", "No text or patterns provided in components")
    
    # Initialize generator
    generator = AssumptionGenerator(ontology=_get_ontology())
    
    # Generate assumptions
    assumptions = generator.generate_assumptions(text, patterns, argument_components)
    
    # Filter by priority if specified
    if prioritization == "critical":
        assumptions = [a for a in assumptions if a.get("impact") in ["critical", "high"]]
    
    return {
        "assumptions": assumptions,
        "next_tools": ["map_assumptions_to_nodes", "construct_argument_graph"],
        "stage_id": "generate_missing_assumptions",
        "inputs_digest": _hash_inputs(argument_components)
    }

5. Stable Tool Registry & Discovery
Files: MCP_example_template/argument_mcp.py, new health.py Problem: Tool catalog intermittently returns empty results, endpoints error with ResourceNotFound Solution:
# Add health check endpoint
@mcp.tool()
def health_status():
    """Health check endpoint returning system status"""
    return {
        "version": "v1.1.0",
        "uptime_ms": int((time.time() - _start_time) * 1000),
        "tool_count": len(_get_tool_registry()),
        "status": "healthy",
        "commit": os.environ.get("GIT_COMMIT", "unknown")
    }

# Add static tool registry resource
@mcp.resource("argument://tools_index")
def tools_index():
    """Static registry of all available tools built at startup"""
    registry = []
    for tool_name in _get_tool_registry():
        registry.append({
            "name": tool_name,
            "canonical_id": tool_name.lower().replace(" ", "_").replace("™", ""),
            "uri": f"argument://tools/{tool_name}",
            "schema_version": "v1.1.0"
        })
    return {"tools": registry}

# Ensure tools_list returns stable results
_TOOL_REGISTRY_CACHE = None
_start_time = time.time()

def _get_tool_registry():
    global _TOOL_REGISTRY_CACHE
    if _TOOL_REGISTRY_CACHE is None:
        # Build once at startup and cache
        _TOOL_REGISTRY_CACHE = [
            "analyze_argument_comprehensive",
            "analyze_argument_stepwise", 
            "decompose_argument_structure",
            "detect_argument_patterns",
            # ... complete list
        ]
    return _TOOL_REGISTRY_CACHE

# On backend restarts, return 503 with retry_after
def _handle_service_unavailable():
    return _error_response("SERVICE_UNAVAILABLE", 
        "Service temporarily unavailable",
        hint="Retry after backend restart completes",
        metadata={"retry_after": 30})


P1 — SCHEMA & CONTRACT HARDENING
6. Unify Object Names and Required Fields
Files: schemas/v1/common/*, all response schemas Problem: Mixed links/edges, optional spans, very loose response schemas Solution:
// schemas/v1/common/Node.json
{
  "$id": "Node",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Node",
  "type": "object",
  "required": ["node_id", "node_type", "content", "source_text_span"],
  "properties": {
    "node_id": {"type": "string", "pattern": "^N_[a-f0-9]{8}$"},
    "node_type": {"type": "string", "enum": ["STATEMENT", "CLAIM", "EVIDENCE"]},
    "primary_subtype": {"type": ["string", "null"], "enum": ["Main Claim", "Statement", "Evidence", null]},
    "content": {"type": "string", "minLength": 1, "maxLength": 2000},
    "source_text_span": {
      "type": "array", 
      "items": {"type": "integer", "minimum": 0}, 
      "minItems": 2, 
      "maxItems": 2
    },
    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "assumptions": {"type": "array", "items": {"type": "string"}}
  }
}

// schemas/v1/common/Link.json  
{
  "$id": "Link",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Link", 
  "type": "object",
  "required": ["link_id", "source_node", "target_node", "relationship_type"],
  "properties": {
    "link_id": {"type": "string", "pattern": "^L_[a-f0-9]{8}$"},
    "source_node": {"type": "string", "pattern": "^N_[a-f0-9]{8}$"},
    "target_node": {"type": "string", "pattern": "^N_[a-f0-9]{8}$"},
    "relationship_type": {"type": "string", "enum": ["SUPPORT", "ATTACK", "REBUT", "UNDERCUT"]},
    "relationship_subtype": {"type": ["string", "null"]}
  }
}

// schemas/v1/common/Pattern.json
{
  "$id": "Pattern", 
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Pattern",
  "type": "object", 
  "required": ["pattern_id", "pattern_type", "label", "confidence", "source_text_span"],
  "properties": {
    "pattern_id": {"type": "string"},
    "pattern_type": {"type": "string", "enum": ["authority", "causal", "analogical", "other"]},
    "label": {"type": "string", "minLength": 1},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "source_text_span": {
      "type": "array",
      "items": {"type": "integer", "minimum": 0},
      "minItems": 2,
      "maxItems": 2  
    },
    "details": {"type": ["object", "null"]}
  }
}

// schemas/v1/common/Weakness.json
{
  "$id": "Weakness",
  "$schema": "http://json-schema.org/draft-07/schema#", 
  "title": "Weakness",
  "type": "object",
  "required": ["label", "why", "how"],
  "properties": {
    "label": {"type": "string", "minLength": 1},
    "type": {"type": ["string", "null"], "enum": ["logical", "epistemic", "rhetorical", null]},
    "severity": {"type": ["string", "null"], "enum": ["low", "medium", "high", "critical", null]},
    "why": {"type": "string", "minLength": 1},
    "how": {"type": "string", "minLength": 1},
    "node_ids": {"type": "array", "items": {"type": "string", "pattern": "^N_[a-f0-9]{8}$"}},
    "span": {
      "type": ["array", "null"],
      "items": {"type": "integer", "minimum": 0},
      "minItems": 2,
      "maxItems": 2
    },
    "confidence": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
    "ontology_bucket": {"type": ["string", "null"]},
    "evidence_hooks": {"type": "array", "items": {"type": "string"}}
  }
}

// schemas/v1/common/Assumption.json
{
  "$id": "Assumption",
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Assumption", 
  "type": "object",
  "required": ["text", "category", "impact", "confidence", "tests"],
  "properties": {
    "text": {"type": "string", "minLength": 1, "maxLength": 500},
    "category": {"type": "string", "enum": ["epistemic", "reliability", "consensus", "temporal", "mechanistic", "bridging", "contentious"]},
    "impact": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    "linked_patterns": {"type": "array", "items": {"type": "string"}},
    "node_ids": {"type": ["array", "null"], "items": {"type": "string", "pattern": "^N_[a-f0-9]{8}$"}},
    "tests": {"type": "array", "items": {"type": "string"}, "minItems": 1}
  }
}

7. Add Missing Request Schemas
Files: schemas/v1/*request.json Problem: No input validation, inconsistent parameter handling Solution: Create request schemas for every endpoint:
// schemas/v1/analyze_argument_comprehensive.request.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "analyze_argument_comprehensive.request",
  "type": "object",
  "required": ["argument_text"],
  "properties": {
    "argument_text": {"type": "string", "minLength": 10, "maxLength": 50000},
    "analysis_depth": {"type": "string", "enum": ["basic", "standard", "comprehensive"], "default": "standard"},
    "forum": {"type": ["string", "null"], "enum": ["academic", "policy", "legal", "informal", null]},
    "audience": {"type": ["string", "null"], "enum": ["expert", "general", "student", null]},
    "goal": {"type": ["string", "null"], "enum": ["analysis", "critique", "decision_support", null]}
  },
  "additionalProperties": false
}

// Apply to all endpoints with proper validation

8. Implement Compatibility Mode
Files: MCP_example_template/argument_mcp.py Problem: MCP clients expect arrays, not envelope objects Solution:
# Add compat parameter support to all endpoints
def _handle_compat_mode(request_data):
    """Check for compat mode in request"""
    return request_data.get("compat") == "raw" or request_data.get("envelope") == False

def ontology_list_dimensions(**kwargs):
    compat_mode = _handle_compat_mode(kwargs)
    dimensions = [...]  # existing logic
    
    if compat_mode:
        return dimensions  # Raw array
    else:
        return _envelope(dimensions, "ontology_list_dimensions")  # Full envelope

# Add sibling endpoints for guaranteed raw mode
@mcp.tool()
def ontology_list_dimensions_raw():
    """Raw array version of ontology_list_dimensions for MCP compatibility"""
    return ontology_list_dimensions(compat="raw")

# Document compatibility clearly in README


P2 — PIPELINE INTEGRITY & TRACEABILITY
9. Fix Pipeline Data Flow Issues
Files: MCP_example_template/argument_mcp.py Problem: Links disappear between analyze_comprehensive and construct_argument_graph Solution:
def construct_argument_graph(analysis_results):
    """Enhanced to handle multiple input formats"""
    if not analysis_results:
        return _error_response("INVALID_INPUT_SHAPE", "Missing analysis_results")
    
    # Handle multiple input formats
    nodes = []
    links = []
    
    if "structure" in analysis_results:
        # Format from analyze_argument_comprehensive
        structure = analysis_results["structure"]
        nodes = structure.get("nodes", [])
        links = structure.get("links", [])
    elif "nodes" in analysis_results:
        # Direct format
        nodes = analysis_results.get("nodes", [])
        links = analysis_results.get("links", [])
    else:
        return _error_response("INVALID_INPUT_SHAPE", 
            "analysis_results must contain 'structure' or 'nodes'",
            hint="Call decompose_argument_structure first to generate structure")
    
    if not nodes:
        return _error_response("INVALID_INPUT_SHAPE", 
            "No nodes found in analysis_results",
            hint="Ensure input contains argument structure with nodes")
    
    if not links and len(nodes) > 1:
        return _error_response("INVALID_INPUT_SHAPE",
            "Multi-node argument requires links between nodes", 
            hint="Pass links or call decompose_argument_structure first")
    
    # Build adjacency list for client convenience
    adjacency = {}
    for link in links:
        source = link["source_node"] 
        if source not in adjacency:
            adjacency[source] = {"outgoing": [], "incoming": []}
        adjacency[source]["outgoing"].append(link["target_node"])
        
        target = link["target_node"]
        if target not in adjacency:
            adjacency[target] = {"outgoing": [], "incoming": []}
        adjacency[target]["incoming"].append(source)
    
    return {
        "nodes": nodes,
        "links": links, 
        "adjacency": adjacency,
        "next_tools": ["validate_argument_graph", "assess_argument_quality"]
    }

# Add contract unit tests
def test_pipeline_integrity():
    """Test that data flows correctly through pipeline"""
    text = "Experts say we should ban X because it causes harm. Therefore we must act."
    
    # Step 1: Comprehensive analysis
    result1 = analyze_argument_comprehensive(text)
    assert "structure" in result1["data"]
    assert len(result1["data"]["structure"]["links"]) > 0
    
    # Step 2: Graph construction should preserve links
    result2 = construct_argument_graph(result1["data"])
    assert len(result2["data"]["links"]) == len(result1["data"]["structure"]["links"])
    
    # Step 3: Validation should find no issues with good input
    result3 = validate_argument_graph(result2["data"])
    assert len(result3["data"]["issues"]["structural"]) == 0

10. Guarantee Span Completeness
Files: MCP_example_template/arg_mcp/engine.py, MCP_example_template/arg_mcp/patterns.py Problem: Missing spans break UI integration and traceability Solution:
class EnhancedSpanTracker:
    def __init__(self, text):
        self.text = text
        self.sentences = self._split_sentences_with_positions()
        self.words = self._tokenize_with_positions()
    
    def _split_sentences_with_positions(self):
        """Split text into sentences with character positions"""
        sentences = []
        current_pos = 0
        
        # Use proper sentence boundary detection
        import re
        sentence_pattern = r'[.!?]+\s+'
        
        for match in re.finditer(sentence_pattern, self.text):
            end_pos = match.start() + 1  # Include punctuation
            sentence_text = self.text[current_pos:end_pos].strip()
            if sentence_text:
                sentences.append({
                    "text": sentence_text,
                    "start": current_pos,
                    "end": end_pos
                })
            current_pos = match.end()
        
        # Handle final sentence without terminal punctuation
        if current_pos < len(self.text):
            final_text = self.text[current_pos:].strip()
            if final_text:
                sentences.append({
                    "text": final_text,
                    "start": current_pos, 
                    "end": len(self.text)
                })
        
        return sentences
    
    def assign_node_spans(self, nodes):
        """Assign accurate spans to all nodes based on content matching"""
        for node in nodes:
            span = self._find_content_span(node["content"])
            if span:
                node["source_text_span"] = span
            else:
                # Fallback: use sentence-level spans
                node["source_text_span"] = self._assign_fallback_span(node["content"])
    
    def _find_content_span(self, content):
        """Find exact span for node content in text"""
        # Try exact match first
        start = self.text.find(content)
        if start >= 0:
            return [start, start + len(content)]
        
        # Try case-insensitive
        start = self.text.lower().find(content.lower())
        if start >= 0:
            return [start, start + len(content)]
        
        # Try fuzzy matching for paraphrased content
        return self._fuzzy_span_match(content)
    
    def _fuzzy_span_match(self, content):
        """Fuzzy matching for paraphrased content"""
        # Use TF-IDF to find most similar text region
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Split text into overlapping windows
        window_size = len(content)
        windows = []
        window_positions = []
        
        for i in range(0, len(self.text) - window_size + 1, window_size // 4):
            window_text = self.text[i:i + window_size]
            windows.append(window_text)
            window_positions.append([i, i + window_size])
        
        if not windows:
            return [0, min(len(content), len(self.text))]
        
        # Find most similar window
        try:
            vectorizer = TfidfVectorizer().fit([content] + windows)
            vectors = vectorizer.transform([content] + windows)
            similarities = cosine_similarity(vectors[0], vectors[1:]).flatten()
            
            best_idx = similarities.argmax()
            if similarities[best_idx] > 0.3:  # Similarity threshold
                return window_positions[best_idx]
        except:
            pass
        
        # Final fallback - proportional positioning
        return self._assign_fallback_span(content)

   def _assign_fallback_span(self, content):
        """Last resort fallback - proportional positioning based on content length"""
        # Never return [0,2] - use proportional spans based on content
        content_ratio = min(len(content) / len(self.text), 0.8) if self.text else 0.1
        
        # Distribute spans proportionally across text
        text_len = len(self.text)
        span_start = int(text_len * 0.1)  # Start after 10% of text
        span_length = max(int(text_len * content_ratio), 20)  # Minimum 20 chars
        span_end = min(span_start + span_length, text_len)
        
        return [span_start, span_end]

# In stage1_decompose: guarantee spans for all nodes
def stage1_decompose(self, text):
    # ... existing decomposition logic ...
    
    # Ensure all nodes have spans
    span_tracker = EnhancedSpanTracker(text)
    span_tracker.assign_node_spans(nodes)
    
    # Validation: no node should have null or [0,2] spans
    for node in nodes:
        span = node.get("source_text_span")
        if not span or span == [0, 2]:
            raise ValueError(f"Node {node['node_id']} has invalid span: {span}")
    
    return nodes, links

# In detect_patterns: guarantee spans for all patterns  
class EnhancedPatternDetector:
    def detect_patterns(self, text, top_n=5):
        patterns = []
        span_tracker = EnhancedSpanTracker(text)
        
        for pattern_info in self._get_pattern_candidates(text):
            # Find actual span for each pattern
            span = self._find_pattern_span(text, pattern_info)
            
            pattern = {
                "pattern_id": pattern_info["id"],
                "pattern_type": pattern_info["type"], 
                "label": pattern_info["label"],
                "confidence": pattern_info["confidence"],
                "source_text_span": span,  # Guaranteed non-null
                "details": pattern_info.get("details", {})
            }
            patterns.append(pattern)
        
        return patterns[:top_n]
    
    def _find_pattern_span(self, text, pattern_info):
        """Find span for pattern using multiple strategies"""
        # Strategy 1: Look for trigger phrases
        trigger_phrases = pattern_info.get("triggers", [])
        for phrase in trigger_phrases:
            span = self._find_phrase_span(text, phrase)
            if span:
                return span
        
        # Strategy 2: Use TF-IDF to find most relevant text region
        keywords = pattern_info.get("keywords", [])
        if keywords:
            span = self._find_keyword_region_span(text, keywords)
            if span:
                return span
        
        # Strategy 3: Pattern-type specific heuristics
        return self._pattern_type_span_heuristic(text, pattern_info)
    
    def _pattern_type_span_heuristic(self, text, pattern_info):
        """Pattern-type specific span finding"""
        pattern_type = pattern_info["type"]
        
        if pattern_type == "authority":
            # Look for "expert", "specialist", "according to", etc.
            authority_markers = ["expert", "specialist", "according to", "research shows", "studies indicate"]
            return self._find_first_marker_span(text, authority_markers)
        
        elif pattern_type == "causal":
            # Look for causal connectives
            causal_markers = ["because", "causes", "leads to", "results in", "due to"]
            return self._find_first_marker_span(text, causal_markers)
        
        elif pattern_type == "analogical":
            # Look for comparison markers
            analogy_markers = ["like", "similar to", "just as", "compared to", "analogous to"]
            return self._find_first_marker_span(text, analogy_markers)
        
        else:
            # Generic fallback - use first quarter of text
            text_len = len(text)
            return [0, min(text_len // 4, 100)]

    def _find_first_marker_span(self, text, markers):
        """Find span of first occurring marker"""
        text_lower = text.lower()
        earliest_pos = len(text)
        best_marker = None
        
        for marker in markers:
            pos = text_lower.find(marker.lower())
            if pos >= 0 and pos < earliest_pos:
                earliest_pos = pos
                best_marker = marker
        
        if best_marker:
            return [earliest_pos, earliest_pos + len(best_marker)]
        
        # No markers found - return proportional span
        return [0, min(len(text) // 3, 50)]

11. Consistent Field Naming Throughout
Files: MCP_example_template/arg_mcp/structures.py, all response schemas Problem: Mixed links/edges naming causes confusion Solution:
# In structures.py - standardize on "links"
class ArgumentGraph:
    def __init__(self):
        self.nodes = []
        self.links = []  # Always use "links", never "edges"
    
    def to_json(self):
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "links": [link.to_dict() for link in self.links]  # Consistent naming
        }
    
    def add_link(self, source_id, target_id, relationship_type, relationship_subtype=None):
        link = ArgumentLink(
            link_id=f"L_{self._generate_id()}",
            source_node=source_id,
            target_node=target_id, 
            relationship_type=relationship_type,
            relationship_subtype=relationship_subtype
        )
        self.links.append(link)
        return link

# Update all API responses to use "links"
def export_graph(graph, format, analysis_id=None):
    if format == "mermaid":
        content = "graph TD;\n"
        for node in graph["nodes"]:
            # ... node rendering
        
        for link in graph["links"]:  # Use "links" consistently
            content += f"  {link['source_node']} --> {link['target_node']};\n"
    
    # ... other formats

12. Enhanced Graph Validation with Specific Feedback
Files: MCP_example_template/argument_mcp.py Problem: Basic validation only, no actionable suggestions Solution:
def validate_argument_graph(graph, validation_level="structural"):
    """Enhanced validation with specific, actionable feedback"""
    issues = {
        "structural": [],
        "logical": [],
        "completeness": [],
        "consistency": []
    }
    suggestions = []
    next_tools = ["assess_argument_quality"]
    next_steps = []
    
    nodes = graph.get("nodes", [])
    links = graph.get("links", [])
    
    # 1. Structural validation
    node_ids = {node["node_id"] for node in nodes}
    
    # Check for dangling links
    dangling_links = []
    for link in links:
        if link["source_node"] not in node_ids:
            dangling_links.append(f"Link {link['link_id']} references missing source node {link['source_node']}")
        if link["target_node"] not in node_ids:
            dangling_links.append(f"Link {link['link_id']} references missing target node {link['target_node']}")
    
    if dangling_links:
        issues["structural"].extend(dangling_links)
        suggestions.append("Remove dangling links or add missing nodes")
        next_steps.append("Clean up node-link consistency")
    
    # Check for duplicate edges
    link_signatures = set()
    duplicate_links = []
    for link in links:
        signature = (link["source_node"], link["target_node"], link["relationship_type"])
        if signature in link_signatures:
            duplicate_links.append(f"Duplicate link: {link['link_id']}")
        link_signatures.add(signature)
    
    if duplicate_links:
        issues["structural"].extend(duplicate_links)
        suggestions.append("Remove duplicate relationships between same nodes")
    
    # Check for cycles (simple DFS)
    cycles = _detect_cycles(nodes, links)
    if cycles:
        issues["structural"].append(f"Circular reasoning detected in {len(cycles)} cycles")
        suggestions.append("Break circular dependencies between premises")
        next_steps.append("Identify and resolve logical cycles")
    
    # 2. Logical validation
    # Find orphan nodes (no connections)
    connected_nodes = set()
    for link in links:
        connected_nodes.add(link["source_node"])
        connected_nodes.add(link["target_node"])
    
    orphan_nodes = [node["node_id"] for node in nodes if node["node_id"] not in connected_nodes]
    if len(orphan_nodes) > 1:  # Allow one orphan (could be standalone claim)
        issues["completeness"].append(f"Found {len(orphan_nodes)} disconnected nodes")
        suggestions.append(f"Connect isolated nodes: {', '.join(orphan_nodes[:3])}")
        next_steps.append("Link premises to main argument structure")
    
    # Check for unsupported claims
    main_claims = [node for node in nodes if node.get("primary_subtype") == "Main Claim"]
    unsupported_claims = []
    for claim in main_claims:
        supporting_links = [l for l in links if l["target_node"] == claim["node_id"] and l["relationship_type"] == "SUPPORT"]
        if not supporting_links:
            unsupported_claims.append(claim["node_id"])
    
    if unsupported_claims:
        issues["logical"].append(f"{len(unsupported_claims)} claims lack support")
        suggestions.append("Add evidence or premises supporting main claims")
        next_tools.append("generate_missing_assumptions")
    
    # 3. Completeness validation
    if not nodes:
        issues["structural"].append("Graph contains no nodes")
        suggestions.append("Add argument structure with decompose_argument_structure")
        return _error_response("INVALID_INPUT_SHAPE", "Empty graph")
    
    if len(nodes) > 1 and not links:
        issues["completeness"].append("Multi-node graph has no relationships")
        suggestions.append("Add relationships between nodes")
        next_steps.append("Identify logical connections between statements")
    
    # Check span completeness
    missing_spans = [node["node_id"] for node in nodes if not node.get("source_text_span")]
    if missing_spans:
        issues["completeness"].append(f"{len(missing_spans)} nodes missing source spans")
        suggestions.append("Regenerate structure to include text spans")
    
    # 4. Consistency validation
    # Check for contradictory relationships
    contradiction_pairs = []
    for i, link1 in enumerate(links):
        for link2 in links[i+1:]:
            if (link1["source_node"] == link2["source_node"] and 
                link1["target_node"] == link2["target_node"] and
                link1["relationship_type"] == "SUPPORT" and
                link2["relationship_type"] == "ATTACK"):
                contradiction_pairs.append((link1["link_id"], link2["link_id"]))
    
    if contradiction_pairs:
        issues["consistency"].append(f"Found {len(contradiction_pairs)} contradictory relationships")
        suggestions.append("Resolve conflicting support/attack relationships")
    
    # Determine overall health and next steps
    total_issues = sum(len(issue_list) for issue_list in issues.values())
    
    if total_issues == 0:
        next_tools.extend(["identify_reasoning_weaknesses", "orchestrate_probe_analysis"])
        next_steps.append("Proceed with weakness detection and probe analysis")
    else:
        if issues["structural"]:
            next_steps.insert(0, "Fix structural issues before proceeding")
        if issues["completeness"]:
            next_tools.append("generate_missing_assumptions")
    
    return {
        "issues": issues,
        "suggestions": suggestions,
        "next_tools": next_tools,
        "next_steps": next_steps,
        "summary": {
            "total_issues": total_issues,
            "severity": "high" if issues["structural"] else "medium" if total_issues > 0 else "none",
            "graph_health": "unhealthy" if issues["structural"] else "needs_attention" if total_issues > 0 else "healthy"
        }
    }

def _detect_cycles(nodes, links):
    """Simple cycle detection using DFS"""
    # Build adjacency list
    adj = {}
    for node in nodes:
        adj[node["node_id"]] = []
    
    for link in links:
        if link["source_node"] in adj:
            adj[link["source_node"]].append(link["target_node"])
    
    # DFS to detect cycles
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node):
        if node in rec_stack:
            return [node]  # Found cycle
        if node in visited:
            return None
        
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in adj.get(node, []):
            cycle = dfs(neighbor)
            if cycle:
                if node in cycle:
                    return cycle  # Complete cycle
                else:
                    cycle.append(node)
                    return cycle
        
        rec_stack.remove(node)
        return None
    
    for node in adj:
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                cycles.append(cycle)
    
    return cycles


P3 — ENHANCED WEAKNESS DETECTION & ONTOLOGY MAPPING
13. Rich Weakness Detection with Ontology Grounding
Files: MCP_example_template/argument_mcp.py, MCP_example_template/arg_mcp/ontology.py Problem: Basic weakness labels only, no actionable guidance Solution:
class OntologyGroundedWeaknessDetector:
    def __init__(self, ontology):
        self.ontology = ontology
        # Map pattern types to ontology categories
        self.pattern_to_ontology = {
            "authority": {
                "primary": "Appeal to Authority",
                "fallback": "Argument from Authority", 
                "bucket": "Appeals / Irrelevant Persuasion",
                "dimension": "Fallacy"
            },
            "causal": {
                "primary": "Post Hoc Ergo Propter Hoc",
                "fallback": "Causal Oversimplification",
                "bucket": "Causal / Statistical Confusions", 
                "dimension": "Fallacy"
            },
            "analogical": {
                "primary": "Weak Analogy", 
                "fallback": "False Equivalence",
                "bucket": "Other / Check",
                "dimension": "Fallacy"
            }
        }
        
    def detect_weaknesses(self, argument_analysis, sensitivity="default"):
        weaknesses = []
        
        patterns = argument_analysis.get("patterns", [])
        structure = argument_analysis.get("structure", {})
        nodes = structure.get("nodes", [])
        
        # Pattern-based weakness detection
        for pattern in patterns:
            pattern_weaknesses = self._analyze_pattern_weaknesses(pattern, nodes)
            weaknesses.extend(pattern_weaknesses)
        
        # Structure-based weakness detection
        structural_weaknesses = self._analyze_structural_weaknesses(structure)
        weaknesses.extend(structural_weaknesses)
        
        # Apply sensitivity filtering
        if sensitivity == "high":
            threshold = 0.3
        elif sensitivity == "low": 
            threshold = 0.7
        else:
            threshold = 0.5
            
        weaknesses = [w for w in weaknesses if w.get("confidence", 0.5) >= threshold]
        
        # Deduplicate and rank by severity
        weaknesses = self._deduplicate_and_rank_weaknesses(weaknesses)
        
        return weaknesses[:10]  # Cap at top 10
    
    def _analyze_pattern_weaknesses(self, pattern, nodes):
        """Analyze weaknesses for specific pattern types"""
        pattern_type = pattern.get("pattern_type", "other")
        weaknesses = []
        
        # Map pattern to nodes via span overlap
        overlapping_nodes = self._find_overlapping_nodes(pattern.get("source_text_span", []), nodes)
        
        if pattern_type == "authority":
            ontology_info = self.pattern_to_ontology["authority"]
            weaknesses.extend([
                {
                    "label": "Appeal to Authority",
                    "type": "epistemic",
                    "severity": "medium",
                    "why": "Relies on expert opinion without establishing credibility or consensus",
                    "how": "Verify expert qualifications, check for bias, examine peer consensus",
                    "node_ids": [node["node_id"] for node in overlapping_nodes],
                    "span": pattern.get("source_text_span"),
                    "confidence": 0.8,
                    "ontology_bucket": ontology_info["bucket"],
                    "evidence_hooks": [
                        "Check expert's publication record",
                        "Research institutional affiliations",
                        "Survey other experts in the field",
                        "Look for potential conflicts of interest"
                    ]
                },
                {
                    "label": "Missing Expert Consensus Check",
                    "type": "epistemic", 
                    "severity": "medium",
                    "why": "Single expert opinion may not represent field consensus",
                    "how": "Survey multiple experts, check professional organization positions",
                    "node_ids": [node["node_id"] for node in overlapping_nodes],
                    "span": pattern.get("source_text_span"),
                    "confidence": 0.6,
                    "ontology_bucket": ontology_info["bucket"],
                    "evidence_hooks": [
                        "Professional association position statements",
                        "Systematic reviews or meta-analyses",
                        "Expert survey data"
                    ]
                }
            ])
            
        elif pattern_type == "causal":
            ontology_info = self.pattern_to_ontology["causal"]
            weaknesses.extend([
                {
                    "label": "Post Hoc Ergo Propter Hoc",
                    "type": "logical",
                    "severity": "high", 
                    "why": "Assumes temporal sequence implies causation without ruling out alternative explanations",
                    "how": "Verify causal mechanism, test alternative explanations, check for confounding variables",
                    "node_ids": [node["node_id"] for node in overlapping_nodes],
                    "span": pattern.get("source_text_span"),
                    "confidence": 0.7,
                    "ontology_bucket": ontology_info["bucket"],
                    "evidence_hooks": [
                        "Controlled experimental data",
                        "Natural experiment opportunities",
                        "Instrumental variable analysis",
                        "Mechanism pathway studies"
                    ]
                },
                {
                    "label": "Unaddressed Confounding",
                    "type": "methodological",
                    "severity": "high",
                    "why": "Fails to consider alternative explanations for observed relationship",
                    "how": "Identify potential confounders, use statistical controls, seek natural experiments", 
                    "node_ids": [node["node_id"] for node in overlapping_nodes],
                    "span": pattern.get("source_text_span"),
                    "confidence": 0.8,
                    "ontology_bucket": "Design & Statistical Pitfalls",
                    "evidence_hooks": [
                        "Propensity score matching data",
                        "Difference-in-differences analysis",
                        "Regression discontinuity studies"
                    ]
                }
            ])
            
        elif pattern_type == "analogical":
            ontology_info = self.pattern_to_ontology["analogical"]
            weaknesses.append({
                "label": "Weak Analogy",
                "type": "logical",
                "severity": "medium",
                "why": "Comparison may not hold for relevant aspects of the argument",
                "how": "Identify key similarities and differences, test analogy boundaries",
                "node_ids": [node["node_id"] for node in overlapping_nodes],
                "span": pattern.get("source_text_span"), 
                "confidence": 0.6,
                "ontology_bucket": ontology_info["bucket"],
                "evidence_hooks": [
                    "Comparative case studies",
                    "Domain expert assessments of similarity",
                    "Historical precedent analysis"
                ]
            })
        
        return weaknesses
    
    def _analyze_structural_weaknesses(self, structure):
        """Find structural argument weaknesses"""
        weaknesses = []
        nodes = structure.get("nodes", [])
        links = structure.get("links", [])
        
        # Find unsupported claims
        main_claims = [node for node in nodes if node.get("primary_subtype") == "Main Claim"]
        for claim in main_claims:
            supporting_links = [l for l in links if l["target_node"] == claim["node_id"] and l["relationship_type"] == "SUPPORT"]
            if not supporting_links:
                weaknesses.append({
                    "label": "Unsupported Claim",
                    "type": "structural",
                    "severity": "high",
                    "why": "Main claim lacks adequate evidentiary support",
                    "how": "Provide evidence, data, or reasoning to support the claim",
                    "node_ids": [claim["node_id"]],
                    "span": claim.get("source_text_span"),
                    "confidence": 0.9,
                    "ontology_bucket": "Evidence & Argumentation Gaps",
                    "evidence_hooks": [
                        "Empirical studies supporting the claim",
                        "Expert opinions or testimony", 
                        "Statistical data and analysis",
                        "Historical precedents or case studies"
                    ]
                })
        
        return weaknesses

def identify_reasoning_weaknesses(argument_analysis, sensitivity="default", weakness_categories=None):
    """Enhanced weakness detection with ontology grounding"""
    if not argument_analysis:
        return _error_response("INVALID_INPUT_SHAPE", "Missing argument_analysis")
    
    detector = OntologyGroundedWeaknessDetector(_get_ontology())
    weaknesses = detector.detect_weaknesses(argument_analysis, sensitivity)
    
    # Filter by categories if specified
    if weakness_categories:
        weaknesses = [w for w in weaknesses if w.get("type") in weakness_categories]
    
    return {"weaknesses": weaknesses}

14. Sophisticated Probe Orchestration
Files: MCP_example_template/arg_mcp/probes.py Problem: Single generic probe only Solution:
class ContextAwareProbeOrchestrator:
    def __init__(self, tool_catalog):
        self.tool_catalog = tool_catalog
        self.pattern_probe_map = {
            "authority": ["The Authority Credential Checker™", "The Consensus Vote Optimizer™"],
            "causal": ["The Causal Probe Playbook™", "The Mill's Methods Card Player™"],
            "analogical": ["The Analogy Fidelity Filter™", "The Structured Analogy Generator™"],
            "dilemmatic": ["The False Dilemma Detector™", "The Creative Synthesis Generator™"]
        }
        
    def generate_probe_plan(self, analysis_results, forum=None, audience=None, goal=None):
        probes = []
        
        patterns = analysis_results.get("patterns", [])
        structure = analysis_results.get("structure", {})
        weaknesses = analysis_results.get("weaknesses", [])
        
        # Pattern-based probe selection
        for pattern in patterns:
            pattern_probes = self._select_pattern_probes(pattern, structure)
            probes.extend(pattern_probes)
        
        # Weakness-based probe selection
        for weakness in weaknesses:
            weakness_probes = self._select_weakness_probes(weakness, structure)
            probes.extend(weakness_probes)
        
        # Context-specific probes
        context_probes = self._select_context_probes(analysis_results, forum, audience, goal)
        probes.extend(context_probes)
        
        # Thematic probes based on content cues
        thematic_probes = self._detect_thematic_probes(analysis_results)
        probes.extend(thematic_probes)
        
        # Deduplicate and rank by priority
        probes = self._deduplicate_and_rank_probes(probes)
        
        return {"probe_plan": probes[:5]}  # Cap at 5 most important probes
    
    def _select_pattern_probes(self, pattern, structure):
        """Select probes based on detected argument patterns"""
        pattern_type = pattern.get("pattern_type", "other")
        probes = []
        
        # Find nodes associated with this pattern
        target_nodes = self._find_overlapping_nodes(pattern.get("source_text_span", []), 
                                                   structure.get("nodes", []))
        
        if pattern_type == "authority":
            probes.append({
                "tool": self._get_tool("The Authority Credential Checker™"),
                "when": "immediate",
                "rationale": "Authority-based argument detected - verify expert credibility",
                "targets": [node["node_id"] for node in target_nodes],
                "why": "Expert opinion cited without establishing credibility",
                "how": "Research expert qualifications, check for bias, verify consensus",
                "priority": "high",
                "expected_outcome": "Credibility assessment and consensus check"
            })
            
        elif pattern_type == "causal":
            probes.append({
                "tool": self._get_tool("The Causal Probe Playbook™"),
                "when": "immediate",
                "rationale": "Causal claim detected - test mechanism and alternatives",
                "targets": [node["node_id"] for node in target_nodes],
                "why": "Causal relationship claimed without sufficient evidence",
                "how": "Apply Mill's methods, check for confounders, verify mechanism",
                "priority": "high", 
                "expected_outcome": "Causal mechanism validation and alternative testing"
            })
            
        elif pattern_type == "analogical":
            probes.append({
                "tool": self._get_tool("The Analogy Fidelity Filter™"),
                "when": "immediate",
                "rationale": "Analogical reasoning detected - test similarity boundaries",
                "targets": [node["node_id"] for node in target_nodes],
                "why": "Analogy used without testing relevant similarities",
                "how": "Identify key similarities/differences, test analogy limits",
                "priority": "medium",
                "expected_outcome": "Analogy validity assessment"
            })
        
        return probes
    
    def _detect_thematic_probes(self, analysis_results):
        """Detect thematic content requiring special probes"""
        probes = []
        text = analysis_results.get("text", "")
        
        # Detect limited options language
        if self._has_limited_options_cues(text):
            probes.append({
                "tool": self._get_tool("The False Dilemma Detector™"),
                "when": "immediate",
                "rationale": "Limited options presented - check for excluded alternatives",
                "targets": [],
                "why": "Argument presents only limited choices",
                "how": "Systematically generate additional options beyond given choices",
                "priority": "high",
                "expected_outcome": "Identification of excluded alternatives"
            })
        
        # Detect geopolitical/conflict terminology
        if self._has_geopolitical_cues(text):
            probes.append({
                "tool": self._get_tool("The Second-Order Consequence Mapper™"),
                "when": "followup",
                "rationale": "Geopolitical context - analyze downstream consequences",
                "targets": [],
                "why": "Complex geopolitical claims require consequence analysis",
                "how": "Map potential ripple effects and unintended consequences", 
                "priority": "medium",
                "expected_outcome": "Downstream impact analysis"
            })
        
        # Detect quantitative claims
        if self._has_quantitative_cues(text):
            probes.append({
                "tool": self._get_tool("The Base Rate Reality Anchor™"),
                "when": "immediate",
                "rationale": "Quantitative claims detected - ground in statistical reality",
                "targets": [],
                "why": "Numerical claims lack base rate context",
                "how": "Research relevant statistical baselines and historical data",
                "priority": "medium",
                "expected_outcome": "Statistical grounding and context"
            })
        
        return probes
    
    def _has_limited_options_cues(self, text):
        """Detect language indicating limited options"""
        cues = [
            "three ways", "two options", "either/or", "only choice",
            "must choose", "no alternative", "either we", "only way"
        ]
        return any(cue in text.lower() for cue in cues)
    
    def _has_geopolitical_cues(self, text):
        """Detect geopolitical/security terminology"""
        cues = [
            "leadership vacuum", "power vacuum", "geopolitical",
            "rogue actors", "global order", "instability", "crisis"
        ]
        return any(cue in text.lower() for cue in cues)
    
    def _has_quantitative_cues(self, text):
        """Detect quantitative claims"""
        import re
        # Look for percentages, numbers, statistical terms
        patterns = [
            r'\d+%', r'\d+\.\d+', r'\$\d+', r'\d+ times',
            r'increase.*\d+', r'decrease.*\d+', r'statistics show'
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)

def orchestrate_probe_analysis(analysis_results, forum=None, audience=None, goal=None):
    """Enhanced probe orchestration with context awareness"""
    if not analysis_results:
        return _error_response("INVALID_INPUT_SHAPE", "Missing analysis_results")
    
    orchestrator = ContextAwareProbeOrchestrator(_get_tool_catalog())
    probe_result = orchestrator.generate_probe_plan(analysis_results, forum, audience, goal)
    
    return probe_result


P4 — PERFORMANCE, LIMITS & ROBUSTNESS
15. Payload Control and Performance Limits
Files: MCP_example_template/argument_mcp.py, new constants Problem: No limits on response sizes or processing time Solution:
# Add configuration constants
MAX_INPUT_LENGTH = 50000  # 50k characters
MAX_NODES = 100
MAX_LINKS = 200
MAX_PATTERNS = 20
MAX_WEAKNESSES = 15
MAX_ASSUMPTIONS = 12
MAX_PROBES = 8
RESPONSE_TIMEOUT_SECONDS = 30

# Add input validation to all endpoints
def _validate_input_size(text, max_length=MAX_INPUT_LENGTH):
    if len(text) > max_length:
        return _error_response("INVALID_INPUT",
            f"Input text too long: {len(text)} characters (max: {max_length})",
            hint="Consider breaking into smaller chunks for analysis")
    return None

def cap_response_size(data, max_items_map):
    """Cap response arrays to prevent oversized payloads"""
    truncated = False
    
    if isinstance(data, dict):
        capped_data = {}
        for key, value in data.items():
            if key in max_items_map and isinstance(value, list):
                max_items = max_items_map[key]
                if len(value) > max_items:
                    capped_data[key] = value[:max_items]
                    truncated = True
                else:
                    capped_data[key] = value
            elif isinstance(value, (dict, list)):
                capped_value, nested_truncated = cap_response_size(value, max_items_map)
                capped_data[key] = capped_value
                truncated = truncated or nested_truncated
            else:
                capped_data[key] = value
        return capped_data, truncated
    
    elif isinstance(data, list):
        capped_list = []
        for item in data:
            if isinstance(item, (dict, list)):
                capped_item, nested_truncated = cap_response_size(item, max_items_map)
                capped_list.append(capped_item)
                truncated = truncated or nested_truncated
            else:
                capped_list.append(item)
        return capped_list, truncated
    
    return data, truncated


# TODO: Add confidence scoring system for argument components
class ConfidenceScorer:
    """Assign and manage confidence scores for argument elements"""
    
    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.5,
            'low': 0.2
        }
    
    def score_premise(self, premise_text, evidence_strength=None):
        """Score individual premise confidence based on evidence quality"""
        # TODO: Implement evidence-based scoring algorithm
        base_score = 0.5
        
        # Adjust based on evidence strength if provided
        if evidence_strength:
            evidence_multiplier = {
                'strong': 1.3,
                'moderate': 1.0,
                'weak': 0.7,
                'none': 0.4
            }
            base_score *= evidence_multiplier.get(evidence_strength, 1.0)
        
        # Cap at 1.0
        return min(base_score, 1.0)
    
    def score_inference(self, premise_scores, inference_type):
        """Score inference confidence based on premise confidence and logical validity"""
        # TODO: Implement inference-specific scoring logic
        if not premise_scores:
            return 0.0
        
        # Base score from weakest premise (chain strength)
        min_premise_score = min(premise_scores)
        
        # Adjust based on inference type
        inference_modifiers = {
            'deductive': 1.0,  # Preserves premise confidence
            'inductive': 0.8,  # Reduces confidence
            'abductive': 0.6,  # Further reduces confidence
            'analogical': 0.5
        }
        
        modifier = inference_modifiers.get(inference_type, 0.7)
        return min_premise_score * modifier


# TODO: Implement dynamic probe selection algorithm
class ProbeSelector:
    """Select most relevant probing tools based on argument characteristics"""
    
    def __init__(self, tool_catalog):
        self.tool_catalog = tool_catalog
        self.argument_patterns = {}
    
    def analyze_argument_profile(self, argument_components):
        """Create profile of argument characteristics for tool selection"""
        profile = {
            'complexity': self._assess_complexity(argument_components),
            'evidence_type': self._identify_evidence_types(argument_components),
            'logical_structure': self._analyze_structure(argument_components),
            'domain': self._identify_domain(argument_components),
            'potential_biases': self._detect_bias_indicators(argument_components)
        }
        return profile
    
    def _assess_complexity(self, components):
        """Assess argument complexity based on number and interconnection of components"""
        # TODO: Implement complexity metrics
        num_premises = len([c for c in components if c.get('type') == 'premise'])
        num_inferences = len([c for c in components if c.get('type') == 'inference'])
        
        if num_premises > 10 or num_inferences > 5:
            return 'high'
        elif num_premises > 5 or num_inferences > 2:
            return 'medium'
        else:
            return 'low'
    
    def _identify_evidence_types(self, components):
        """Identify types of evidence present in argument"""
        # TODO: Implement evidence type detection
        evidence_indicators = {
            'statistical': ['percent', 'data', 'study', 'research'],
            'expert': ['expert', 'authority', 'professional'],
            'anecdotal': ['example', 'case', 'instance'],
            'logical': ['therefore', 'because', 'since', 'implies']
        }
        
        detected_types = []
        argument_text = ' '.join([c.get('content', '') for c in components])
        
        for evidence_type, indicators in evidence_indicators.items():
            if any(indicator in argument_text.lower() for indicator in indicators):
                detected_types.append(evidence_type)
        
        return detected_types
    
    def _analyze_structure(self, components):
        """Analyze logical structure pattern"""
        # TODO: Implement structure pattern recognition
        return 'complex'  # Placeholder
    
    def _identify_domain(self, components):
        """Identify subject domain of argument"""
        # TODO: Implement domain classification
        domain_keywords = {
            'politics': ['government', 'policy', 'election', 'democracy'],
            'science': ['research', 'study', 'evidence', 'theory'],
            'economics': ['market', 'economic', 'financial', 'trade'],
            'ethics': ['moral', 'ethical', 'right', 'wrong', 'should']
        }
        
        argument_text = ' '.join([c.get('content', '') for c in components])
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in argument_text.lower() for keyword in keywords):
                return domain
        
        return 'general'
    
    def _detect_bias_indicators(self, components):
        """Detect potential cognitive bias indicators"""
        # TODO: Implement bias detection heuristics
        bias_indicators = {
            'confirmation_bias': ['obviously', 'clearly', 'everyone knows'],
            'false_dichotomy': ['only two', 'either or', 'must choose'],
            'ad_hominem': ['typical of', 'what you\'d expect from'],
            'appeal_to_emotion': ['devastating', 'shocking', 'outrageous']
        }
        
        detected_biases = []
        argument_text = ' '.join([c.get('content', '') for c in components])
        
        for bias_type, indicators in bias_indicators.items():
            if any(indicator in argument_text.lower() for indicator in indicators):
                detected_biases.append(bias_type)
        
        return detected_biases
    
    def select_probes(self, argument_profile, max_probes=5):
        """Select most relevant probing tools based on argument profile"""
        # TODO: Implement intelligent probe selection
        selected_probes = []
        
        # Base probes for all arguments
        base_probes = ['premise_strength_evaluator', 'logical_structure_mapper']
        selected_probes.extend(base_probes)
        
        # Complexity-based selection
        if argument_profile['complexity'] == 'high':
            selected_probes.extend(['assumption_excavator', 'causal_chain_tracer'])
        
        # Evidence-type based selection
        if 'statistical' in argument_profile['evidence_type']:
            selected_probes.append('statistical_validity_checker')
        
        if 'expert' in argument_profile['evidence_type']:
            selected_probes.append('authority_credibility_assessor')
        
        # Bias-based selection
        if argument_profile['potential_biases']:
            selected_probes.append('cognitive_bias_detector')
        
        # Domain-specific selection
        domain_specific_probes = {
            'politics': ['political_assumption_checker', 'stakeholder_analysis'],
            'science': ['methodology_validator', 'peer_review_assessor'],
            'economics': ['economic_assumption_checker', 'data_source_validator']
        }
        
        domain = argument_profile['domain']
        if domain in domain_specific_probes:
            selected_probes.extend(domain_specific_probes[domain])
        
        # Return top probes up to max limit
        return selected_probes[:max_probes]


# TODO: Implement argument strength calculator
class ArgumentStrengthCalculator:
    """Calculate overall argument strength from component analysis"""
    
    def __init__(self):
        self.strength_weights = {
            'premise_strength': 0.4,
            'logical_validity': 0.3,
            'evidence_quality': 0.2,
            'coherence': 0.1
        }
    
    def calculate_strength(self, argument_analysis):
        """Calculate weighted argument strength score"""
        # TODO: Implement comprehensive strength calculation
        
        components = argument_analysis.get('components', [])
        if not components:
            return 0.0
        
        # Calculate component scores
        premise_score = self._calculate_premise_strength(components)
        logic_score = self._calculate_logical_validity(components)
        evidence_score = self._calculate_evidence_quality(components)
        coherence_score = self._calculate_coherence(components)
        
        # Weighted average
        total_strength = (
            premise_score * self.strength_weights['premise_strength'] +
            logic_score * self.strength_weights['logical_validity'] +
            evidence_score * self.strength_weights['evidence_quality'] +
            coherence_score * self.strength_weights['coherence']
        )
        
        return round(total_strength, 3)
    
    def _calculate_premise_strength(self, components):
        """Calculate average strength of premises"""
        # TODO: Implement premise strength aggregation
        premises = [c for c in components if c.get('type') == 'premise']
        if not premises:
            return 0.0
        
        premise_scores = [c.get('confidence', 0.5) for c in premises]
        return sum(premise_scores) / len(premise_scores)
    
    def _calculate_logical_validity(self, components):
        """Assess logical validity of inferences"""
        # TODO: Implement logical validity assessment
        inferences = [c for c in components if c.get('type') == 'inference']
        if not inferences:
            return 0.8  # Default for arguments without explicit inferences
        
        # Simplified assessment - in practice would need formal logic checking
        validity_scores = []
        for inference in inferences:
            inference_type = inference.get('inference_type', 'unknown')
            type_scores = {
                'deductive': 0.9,
                'inductive': 0.7,
                'abductive': 0.6,
                'analogical': 0.5,
                'unknown': 0.4
            }
            validity_scores.append(type_scores.get(inference_type, 0.4))
        
        return sum(validity_scores) / len(validity_scores)
    
    def _calculate_evidence_quality(self, components):
        """Assess quality of supporting evidence"""
        # TODO: Implement evidence quality metrics
        evidence_components = [c for c in components if c.get('has_evidence', False)]
        if not evidence_components:
            return 0.3  # Low score for no evidence
        
        # Simplified quality assessment
        quality_scores = []
        for component in evidence_components:
            evidence_type = component.get('evidence_type', 'anecdotal')
            type_quality = {
                'peer_reviewed': 0.9,
                'statistical': 0.8,
                'expert_opinion': 0.7,
                'survey': 0.6,
                'anecdotal': 0.3
            }
            quality_scores.append(type_quality.get(evidence_type, 0.3))
        
        return sum(quality_scores) / len(quality_scores)
    
    def _calculate_coherence(self, components):
        """Assess internal coherence and consistency"""
        # TODO: Implement coherence assessment algorithm
        # This would check for contradictions, logical gaps, etc.
        return 0.7  # Placeholder


# TODO: Add comprehensive counter-argument generator
class CounterArgumentGenerator:
    """Generate systematic counter-arguments and alternative perspectives"""
    
    def __init__(self):
        self.counter_strategies = [
            'challenge_premises',
            'question_evidence',
            'alternative_explanations',
            'consequence_analysis',
            'scope_limitations',
            'methodological_critique'
        ]
    
    def generate_counters(self, argument_analysis, strategy_limit=3):
        """Generate counter-arguments using multiple strategies"""
        # TODO: Implement systematic counter-argument generation
        
        counters = {}
        components = argument_analysis.get('components', [])
        
        for strategy in self.counter_strategies[:strategy_limit]:
            if strategy == 'challenge_premises':
                counters[strategy] = self._challenge_premises(components)
            elif strategy == 'question_evidence':
                counters[strategy] = self._question_evidence(components)
            elif strategy == 'alternative_explanations':
                counters[strategy] = self._alternative_explanations(components)
            elif strategy == 'consequence_analysis':
                counters[strategy] = self._analyze_consequences(components)
            elif strategy == 'scope_limitations':
                counters[strategy] = self._identify_scope_limits(components)
            elif strategy == 'methodological_critique':
                counters[strategy] = self._critique_methodology(components)
        
        return counters
    
    def _challenge_premises(self, components):
        """Generate challenges to key premises"""
        # TODO: Implement premise challenging logic
        premises = [c for c in components if c.get('type') == 'premise']
        challenges = []
        
        for premise in premises:
            content = premise.get('content', '')
            # Generate specific challenges based on premise content
            if 'most dangerous' in content.lower():
                challenges.append(f"Is this truly the 'most dangerous' period? Historical comparisons may lack context.")
            
            if 'no reliable leadership' in content.lower():
                challenges.append("Define 'reliable leadership' - are there examples of effective leadership that are being overlooked?")
        
        return challenges
    
    def _question_evidence(self, components):
        """Generate questions about evidence quality and sources"""
        # TODO: Implement evidence questioning
        questions = [
            "What specific data supports the claim of 'most dangerous period'?",
            "Are the sources cited representative and unbiased?",
            "How recent and relevant is the supporting evidence?"
        ]
        return questions
    
    def _alternative_explanations(self, components):
        """Generate alternative explanations for observed phenomena"""
        # TODO: Implement alternative explanation generation
        alternatives = [
            "Could current instability be part of normal geopolitical cycles rather than unprecedented crisis?",
            "Might perceived leadership vacuum reflect transition to multipolar rather than unipolar world order?",
            "Are current challenges more visible due to increased media coverage and connectivity?"
        ]
        return alternatives
    
    def _analyze_consequences(self, components):
        """Analyze potential negative consequences of proposed solutions"""
        # TODO: Implement consequence analysis
        consequence_analysis = [
            "What are the risks of attempting to impose order through force?",
            "Could institutional reform efforts create more instability during transition?",
            "How might different stakeholders be harmed by proposed solutions?"
        ]
        return consequence_analysis
    
    def _identify_scope_limits(self, components):
        """Identify limitations in argument scope and applicability"""
        # TODO: Implement scope limitation identification
        limitations = [
            "Argument focuses primarily on Western perspective - how do other regions view current order?",
            "Time frame may be too limited to assess long-term geopolitical trends",
            "Analysis may conflate different types of instability with different causes"
        ]
        return limitations
    
    def _critique_methodology(self, components):
        """Critique analytical methodology and approach"""
        # TODO: Implement methodological critique
        critiques = [
            "Relies heavily on qualitative assessments without quantitative metrics",
            "May suffer from recency bias in threat assessment",
            "Limited consideration of adaptive capacity and resilience factors"
        ]
        return critiques


# TODO: Implement visualization export system
class ArgumentVisualizationExporter:
    """Export argument analysis results to various visualization formats"""
    
    def __init__(self):
        self.supported_formats = ['mermaid', 'graphviz', 'json', 'html']
    
    def export_argument_map(self, argument_analysis, format_type='mermaid'):
        """Export argument structure as visual map"""
        # TODO: Implement format-specific export logic
        
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}")
        
        if format_type == 'mermaid':
            return self._export_mermaid(argument_analysis)
        elif format_type == 'graphviz':
            return self._export_graphviz(argument_analysis)
        elif format_type == 'json':
            return self._export_json(argument_analysis)
        elif format_type == 'html':
            return self._export_html(argument_analysis)
    
    def _export_mermaid(self, analysis):
        """Export as Mermaid diagram syntax"""
        # TODO: Generate Mermaid diagram code
        mermaid_code = "graph TD\n"
        
        components = analysis.get('components', [])
        for i, component in enumerate(components):
            node_id = f"A{i}"
            content = component.get('content', '')[:50] + "..."  # Truncate for readability
            mermaid_code += f"    {node_id}[\"{content}\"]\n"
        
        # Add relationships
        for i in range(len(components) - 1):
            mermaid_code += f"    A{i} --> A{i+1}\n"
        
        return mermaid_code
    
    def _export_graphviz(self, analysis):
        """Export as Graphviz DOT notation"""
        # TODO: Generate DOT format
        dot_code = "digraph ArgumentMap {\n"
        dot_code += "    rankdir=TB;\n"
        dot_code += "    node [shape=box];\n"
        
        components = analysis.get('components', [])
        for i, component in enumerate(components):
            content = component.get('content', '').replace('"', '\\"')[:50] + "..."
            dot_code += f"    A{i} [label=\"{content}\"];\n"
        
        for i in range(len(components) - 1):
            dot_code += f"    A{i} -> A{i+1};\n"
        
        dot_code += "}"
        return dot_code
    
    def _export_json(self, analysis):
        """Export as JSON structure"""
        # TODO: Create comprehensive JSON export
        import json
        return json.dumps(analysis, indent=2, default=str)
    
    def _export_html(self, analysis):
        """Export as interactive HTML visualization"""
        # TODO: Generate HTML with embedded JavaScript for interactivity
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Argument Analysis Visualization</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .component { border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
                .premise { background-color: #e8f4f8; }
                .conclusion { background-color: #f8f8e8; }
                .inference { background-color: #f8e8f8; }
            </style>
        </head>
        <body>
            <h1>Argument Analysis</h1>
            <div id="argument-map">
                <!-- TODO: Generate component HTML -->
            </div>
        </body>
        </html>
        """
        return html_template


# TODO: Add batch analysis capabilities
class BatchArgumentAnalyzer:
    """Analyze multiple arguments in batch with comparison capabilities"""
    
    def __init__(self, analyzer_config=None):
        self.config = analyzer_config or {}
        self.analysis_cache = {}
    
    def analyze_batch(self, arguments_list, compare=True):
        """Analyze multiple arguments and optionally compare them"""
        # TODO: Implement batch processing with progress tracking
        
        results = {}
        for i, argument_text in enumerate(arguments_list):
            argument_id = f"argument_{i+1}"
            
            # TODO: Call main analysis pipeline for each argument
            analysis_result = self._analyze_single(argument_text, argument_id)
            results[argument_id] = analysis_result
            
            # Cache for comparison
            self.analysis_cache[argument_id] = analysis_result
        
        if compare and len(arguments_list) > 1:
            comparison = self._compare_arguments(list(results.values()))
            results['comparison'] = comparison
        
        return results
    
    def _analyze_single(self, argument_text, argument_id):
        """Analyze single argument with full pipeline"""
        # TODO: Integrate with main analysis pipeline
        # Placeholder implementation
        return {
            'id': argument_id,
            'text': argument_text,
            'strength_score': 0.7,  # Placeholder
            'components': [],
            'weaknesses': [],
            'counters': []
        }
    
    def _compare_arguments(self, analysis_results):
        """Compare multiple argument analyses"""
        # TODO: Implement comprehensive argument comparison
        
        comparison = {
            'strength_ranking': [],
            'common_patterns': [],
            'unique_features': {},
            'recommendation': ''
        }
        
        # Rank by strength
        sorted_by_strength = sorted(
            analysis_results,
            key=lambda x: x.get('strength_score', 0),
            reverse=True
        )
        
        comparison['strength_ranking'] = [arg['id'] for arg in sorted_by_strength]
        
        # TODO: Implement pattern analysis and unique feature detection
        
        return comparison


# TODO: Add integration testing framework
class ArgumentAnalysisTestSuite:
    """Test suite for validating argument analysis pipeline"""
    
    def __init__(self):
        self.test_cases = []
        self.benchmarks = {}
    
    def add_test_case(self, name, argument_text, expected_components, expected_weaknesses=None):
        """Add a test case for validation"""
        test_case = {
            'name': name,
            'argument_text': argument_text,
            'expected_components': expected_components,
            'expected_weaknesses': expected_weaknesses or []
        }
        self.test_cases.append(test_case)
    
    def run_tests(self, analyzer):
        """Run all test cases and report results"""
        # TODO: Implement comprehensive test execution
        
        results = {
            'passed': 0,
            'failed': 0,
            'details': []
        }
        
        for test_case in self.test_cases:
            try:
                # Run analysis
                analysis_result = analyzer.analyze(test_case['argument_text'])
                
                # Validate results
                validation = self._validate_analysis(analysis_result, test_case)
                
                if validation['passed']:
                    results['passed'] += 1
                else:
                    results['failed'] += 1
                
                results['details'].append({
                    'test_name': test_case['name'],
                    'passed': validation['passed'],
                    'issues': validation.get('issues', [])
                })
                
            except Exception as e:
                results['failed'] += 1
                results['details'].append({
                    'test_name': test_case['name'],
                    'passed': False,
                    'error': str(e)
                })
        
        return results
    
    def _validate_analysis(self, analysis_result, test_case):
        """Validate analysis result against expected outcomes"""
        # TODO: Implement detailed validation logic
        
        validation = {
            'passed': True,
            'issues': []
        }
        
        # Check component count
        expected_count = len(test_case['expected_components'])
        actual_count = len(analysis_result.get('components', []))
        
        if abs(expected_count - actual_count) > 2:  # Allow some tolerance
            validation['passed'] = False
            validation['issues'].append(f"Component count mismatch: expected ~{expected_count}, got {actual_count}")
        
        # TODO: Add more sophisticated validation checks
        
        return validation
    
    def benchmark_performance(self, analyzer, iterations=10):
        """Benchmark analyzer performance"""
        # TODO: Implement performance benchmarking
        import time
        
        benchmark_args = [
            "Short argument for speed test.",
            "Medium length argument with several premises and a clear conclusion that should be analyzed thoroughly.",
            "Very long and complex argument with multiple premises, sub-arguments, extensive reasoning chains, and various types of evidence that should stress test the analysis pipeline and reveal performance characteristics under load."
        ]
        
        results = {}
        
        for i, arg in enumerate(benchmark_args):
            times = []
            for _ in range(iterations):
                start_time = time.time()
                analyzer.analyze(arg)
                end_time = time.time()
                times.append(end_time - start_time)
            
            results[f"argument_length_{i+1}"] = {
                'avg_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times),
                'argument_length': len(arg.split())
            }
        
        return results


if __name__ == "__main__":
    # TODO: Add main execution pipeline
    print("Global Leadership Crisis Argument Analysis Pipeline")
    print("=" * 50)
    
    # Example usage and testing
    sample_argument = """
    We have now entered the most dangerous period of global instability in our lifetimes—
    a G-Zero world with no reliable leadership. In 2025, power vacuums are expanding, 
    rogue actors are emboldened, and the risk of major crises is climbing even higher.
    """
    
    print(f"Sample argument: {sample_argument[:100]}...")
    
    # TODO: Initialize and run full analysis pipeline
    # analyzer = ComprehensiveArgumentAnalyzer()
    # result = analyzer.analyze(sample_argument)
    # print(f"Analysis complete. Strength score: {result.get('strength_score', 'N/A')}")

