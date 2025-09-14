import json, sys

from MCP_example_template.argument_mcp import (
    decompose_argument_structure,
    detect_argument_patterns,
    ontology_pattern_match,
    generate_missing_assumptions,
    construct_argument_graph,
    validate_argument_graph,
    identify_reasoning_weaknesses,
    orchestrate_probe_analysis,
    export_graph,
    analyze_and_probe,
)

def data(env):
    return env.get('data', env) if isinstance(env, dict) else env

with open('tmp_argument.txt', 'r', encoding='utf-8') as f:
    text = f.read()

steps = []

env1 = decompose_argument_structure(text)
d1 = data(env1)
structure = d1.get('structure', {})
patterns_d1 = d1.get('patterns', [])
steps.append({'step': 'decompose_argument_structure', 'nodes': len(structure.get('nodes', [])), 'links': len(structure.get('links', []))})

env2 = detect_argument_patterns(text)
d2 = data(env2)
patterns = d2.get('patterns', [])
top_patterns = [{'type': p.get('pattern_type'), 'conf': p.get('confidence'), 'span': p.get('source_text_span')} for p in patterns[:5]]
steps.append({'step': 'detect_argument_patterns', 'count': len(patterns), 'top': top_patterns})

env3 = ontology_pattern_match(patterns)
d3 = data(env3)
matches = d3.get('matches', [])
steps.append({'step': 'ontology_pattern_match', 'matches': len(matches)})

env4 = generate_missing_assumptions({'text': text, 'patterns': patterns})
d4 = data(env4)
assumptions = d4.get('assumptions', [])
steps.append({'step': 'generate_missing_assumptions', 'assumptions': len(assumptions)})

env5 = construct_argument_graph({'structure': structure})
d5 = data(env5)
graph = {'nodes': d5.get('nodes', []), 'links': d5.get('links', [])}
steps.append({'step': 'construct_argument_graph', 'nodes': len(graph['nodes']), 'links': len(graph['links'])})

env6 = validate_argument_graph(graph)
d6 = data(env6)
issues = d6.get('issues', {})
steps.append({'step': 'validate_argument_graph', 'issues': list(issues.keys())})

env7 = identify_reasoning_weaknesses({'patterns': patterns, 'text': text, 'structure': structure})
d7 = data(env7)
weaknesses = d7.get('weaknesses', [])
steps.append({'step': 'identify_reasoning_weaknesses', 'weaknesses': weaknesses})

env8 = orchestrate_probe_analysis({'patterns': patterns, 'structure': structure}, forum='policy', audience='general', goal='decision support')
d8 = data(env8)
probe_plan = d8.get('probe_plan', [])
steps.append({'step': 'orchestrate_probe_analysis', 'probes': probe_plan})

env9 = export_graph(graph, 'mermaid')
d9 = data(env9)
mermaid = d9.get('content', '')
steps.append({'step': 'export_graph', 'format': 'mermaid', 'content_len': len(mermaid)})

env10 = analyze_and_probe(text, analysis_depth='comprehensive', audience='general', goal='decision support')
d10 = data(env10)
steps.append({'step': 'analyze_and_probe', 'patterns': len(d10.get('patterns', [])), 'assumptions': len(d10.get('assumptions', [])), 'weaknesses': len(d10.get('weaknesses', [])), 'probes': len(d10.get('probe_plan', []))})

print(json.dumps({'steps': steps, 'sample_outputs': {
    'assumptions': assumptions[:3],
    'weaknesses': weaknesses[:3],
    'probes': probe_plan[:3],
    'mermaid_snippet': mermaid.splitlines()[:8]
}}, ensure_ascii=False))


