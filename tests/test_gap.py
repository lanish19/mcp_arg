import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from MCP_example_template.arg_mcp.gap import InferenceEngine
from MCP_example_template.arg_mcp.ontology import Ontology, load_ontology
from MCP_example_template.arg_mcp.domain_profiles import PROFILES

# load small ontology from CSV
onto = Ontology(load_ontology('new_argumentation_database_buckets_fixed.csv'))

class TestInferenceEngine(unittest.TestCase):
    def test_causal_temporal_requirement(self):
        engine = InferenceEngine(onto, PROFILES['general'])
        candidate = {
            'scheme': 'Argument from Cause to Effect',
            'text': 'Smoking causes cancer.',
            'roles': {'cause': 'Smoking', 'effect': 'cancer'}
        }
        res = engine.evaluate_scheme(candidate)
        names = [r.requirement.name for r in res.requirements if not r.satisfied]
        self.assertIn('temporal_order', names)

if __name__ == '__main__':
    unittest.main()
