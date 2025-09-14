import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.argument_mcp import (
    analyze_and_probe,
    decompose_argument_structure,
    identify_reasoning_weaknesses,
    export_graph,
)


SAMPLE = (
    "If we withdraw troops now, the region will fall into a leadership vacuum, "
    "and either we accept instability or we return later at higher cost."
)


class TestIntegrationGeopolitics(unittest.TestCase):
    def _data(self, env):
        return env.get('data', env)

    def test_assumptions_weaknesses_probes_and_export(self):
        # Analyze end-to-end
        res = self._data(analyze_and_probe(SAMPLE))
        # Assumptions present
        self.assertTrue(len(res.get('assumptions', [])) >= 1)
        # Weaknesses with spans and optional node_ids
        weak_env = identify_reasoning_weaknesses({"patterns": res.get('patterns', []), "text": SAMPLE, "structure": res.get('structure', {})})
        weaknesses = self._data(weak_env).get('weaknesses', [])
        self.assertTrue(len(weaknesses) >= 1)
        # Probes present
        self.assertTrue(len(res.get('probe_plan', [])) >= 1)
        # Export graph contains main claim id in mermaid output indirectly (node id labels present)
        decomp = self._data(decompose_argument_structure(SAMPLE))
        mer = self._data(export_graph(decomp.get('structure', {}), 'mermaid'))
        self.assertIn('content', mer)
        self.assertIn('graph TD', mer.get('content', ''))


if __name__ == '__main__':
    unittest.main()


