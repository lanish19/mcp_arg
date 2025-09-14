import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.argument_mcp import (
    decompose_argument_structure,
    map_assumptions_to_nodes,
    export_graph,
    analyze_and_probe,
    health_status,
    ontology_list_dimensions,
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

    def test_health_status_ok_and_schema_compliant(self):
        res = health_status()
        self.assertIsInstance(res, dict)
        self.assertEqual(res["metadata"]["schema_url"], "schemas/v1/health_status.response.json")
        data = res["data"]
        self.assertEqual(data["status"], "healthy")
        self.assertGreater(data["tool_count"], 0)
        self.assertIn("version", data)
        self.assertIn("uptime_ms", data)
        self.assertIn("dimensions", data)
        self.assertIn("ontology_bucket_counts", data)
        self.assertIn("commit", data)

    def test_ontology_list_dimensions_compat_mode(self):
        res = ontology_list_dimensions(compat="raw")
        self.assertIsInstance(res, list)
        self.assertIn("Argument Scheme", res)
        self.assertIn("Fallacy", res)
        # Ensure it's just a list of strings
        self.assertTrue(all(isinstance(x, str) for x in res))


if __name__ == '__main__':
    unittest.main()


