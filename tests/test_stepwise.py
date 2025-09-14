import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.argument_mcp import (
    decompose_argument_structure,
    detect_argument_patterns,
    generate_missing_assumptions,
    construct_argument_graph,
    validate_argument_graph,
    analyze_argument_stepwise,
)


FIXED_INPUT = "Experts say we should ban X because it causes harm. Therefore, we must act."


class TestStepwise(unittest.TestCase):
    def _data(self, env):
        return env.get('data', env)

    def test_next_tools_presence(self):
        # decompose_argument_structure
        r1 = self._data(decompose_argument_structure(FIXED_INPUT))
        self.assertIn('next_tools', r1)
        self.assertTrue(len(r1['next_tools']) > 0)

        # detect_argument_patterns
        r2 = self._data(detect_argument_patterns(FIXED_INPUT))
        self.assertIn('next_tools', r2)
        self.assertTrue(len(r2['next_tools']) > 0)

        # generate_missing_assumptions
        r3 = self._data(generate_missing_assumptions({'text': FIXED_INPUT}))
        self.assertIn('next_tools', r3)
        self.assertTrue(len(r3['next_tools']) > 0)

        # construct_argument_graph
        r4 = self._data(construct_argument_graph({'structure': r1.get('structure', {})}))
        self.assertIn('next_tools', r4)
        self.assertTrue(len(r4['next_tools']) > 0)

        # validate_argument_graph
        r5 = self._data(validate_argument_graph({'nodes': r4.get('nodes', []), 'links': r4.get('links', [])}))
        self.assertIn('next_tools', r5)
        self.assertTrue(len(r5['next_tools']) > 0)

    def test_default_stepwise(self):
        env = analyze_argument_stepwise(FIXED_INPUT)
        res = self._data(env)
        self.assertIn('stages', res)
        self.assertTrue(len(res['stages']) >= 6)
        for st in res['stages']:
            self.assertIn('name', st)
            self.assertIn('inputs_summary', st)
            self.assertIn('key_outputs', st)
            self.assertIn('next_tools', st)
        fa = res.get('final_artifacts', {})
        self.assertIn('structure', fa)
        self.assertIn('patterns', fa)
        self.assertIn('assumptions', fa)
        self.assertIn('graph', fa)
        self.assertIn('assessments', fa)
        self.assertIn('probes', fa)

    def test_custom_steps_subset(self):
        steps = ["decompose_argument_structure", "detect_argument_patterns"]
        res = self._data(analyze_argument_stepwise(FIXED_INPUT, steps=steps))
        self.assertEqual(len(res['stages']), 2)
        fa = res.get('final_artifacts', {})
        self.assertIn('structure', fa)
        self.assertIn('patterns', fa)
        # assumptions/graph may be empty at this point

    def test_invalid_step_name(self):
        steps = ["decompose_argument_structure", "not_a_real_step"]
        env = analyze_argument_stepwise(FIXED_INPUT, steps=steps)
        self.assertIsNotNone(env.get('error'))
        self.assertEqual(env['error'].get('code'), 'INVALID_INPUT_SHAPE')
        data = self._data(env)
        self.assertEqual(data.get('step'), 'not_a_real_step')
        self.assertIn('allowed', data)
        self.assertIn('stages_so_far', data)

    def test_max_steps_truncation(self):
        res = self._data(analyze_argument_stepwise(FIXED_INPUT, max_steps=3))
        self.assertTrue(res.get('truncated', False))
        self.assertEqual(len(res.get('stages', [])), 3)


if __name__ == '__main__':
    unittest.main()


