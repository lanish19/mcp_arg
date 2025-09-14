import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.argument_mcp import (
    detect_argument_patterns,
    decompose_argument_structure,
    tools_get,
    export_graph,
)


class TestErrors(unittest.TestCase):
    def test_missing_argument_text(self):
        env = detect_argument_patterns(" ")
        self.assertIsNotNone(env.get('error'))
        self.assertEqual(env['error'].get('code'), 'MISSING_ARGUMENT_TEXT')
        env2 = decompose_argument_structure("")
        self.assertIsNotNone(env2.get('error'))
        self.assertEqual(env2['error'].get('code'), 'MISSING_ARGUMENT_TEXT')

    def test_tool_not_found(self):
        env = tools_get("Nonexistent Tool")
        self.assertIsNotNone(env.get('error'))
        self.assertEqual(env['error'].get('code'), 'TOOL_NOT_FOUND')

    def test_export_bad_format(self):
        env = export_graph({"nodes": [], "links": []}, "pptx")
        self.assertIsNotNone(env.get('error'))
        self.assertEqual(env['error'].get('code'), 'UNSUPPORTED_FORMAT')


if __name__ == '__main__':
    unittest.main()


