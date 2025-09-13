import unittest
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from MCP_example_template.arg_mcp.domain_profiles import PROFILES
from MCP_example_template.arg_mcp.gap import InferenceEngine
from MCP_example_template.arg_mcp.ontology import Ontology, load_ontology

onto = Ontology(load_ontology('new_argumentation_database_buckets_fixed.csv'))

class TestDomainProfiles(unittest.TestCase):
    def test_scientific_extra_requirement(self):
        engine = InferenceEngine(onto, PROFILES['scientific'])
        candidate = {
            'scheme': 'Argument from Cause to Effect',
            'text': 'Study shows X leads to Y',
            'roles': {'cause': 'X', 'effect': 'Y'},
            'scheme_key': 'causal'
        }
        res = engine.evaluate_scheme(candidate)
        names = [r.requirement.name for r in res.requirements]
        self.assertIn('statistical_power', names)

if __name__ == '__main__':
    unittest.main()
