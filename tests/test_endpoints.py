import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.argument_mcp import (
    decompose_argument_structure,
    map_assumptions_to_nodes,
    export_graph,
    analyze_and_probe,
)


TXT = "Experts say we should ban X because it causes harm. Therefore, we must act."


class TestEndpoints(unittest.TestCase):
    def _data(self, env):
        return env.get('data', env)

    def test_map_assumptions_to_nodes(self):
        env = decompose_argument_structure(TXT)
        structure = self._data(env).get('structure', {})
        env2 = map_assumptions_to_nodes({"structure": structure}, [{"text": "Because X causes harm"}])
        d = self._data(env2)
        self.assertIn('mappings', d)
        self.assertIsInstance(d['mappings'], list)

    def test_export_graph_formats(self):
        env = decompose_argument_structure(TXT)
        structure = self._data(env).get('structure', {})
        for fmt in ["mermaid", "graphviz", "jsonld"]:
            out = export_graph(structure, fmt)
            d = self._data(out)
            self.assertEqual(d.get('format'), fmt)
            self.assertTrue(isinstance(d.get('content'), str))
        bad = export_graph(structure, "pptx")
        self.assertIsNotNone(bad.get('error'))
        self.assertEqual(bad['error'].get('code'), 'UNSUPPORTED_FORMAT')

    def test_analyze_and_probe(self):
        env = analyze_and_probe(TXT)
        d = self._data(env)
        self.assertIn('structure', d)
        self.assertIn('probe_plan', d)


if __name__ == '__main__':
    unittest.main()


