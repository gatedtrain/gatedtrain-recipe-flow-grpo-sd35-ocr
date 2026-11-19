"""The adapter `flow_grpo_sd35` conforms to gatedtrain's adapter interface (gatedtrain.ext.conformance)."""

import unittest

from gatedtrain.ext.conformance import AdapterConformance


class TestConformance(AdapterConformance, unittest.TestCase):
    ADAPTER = "flow_grpo_sd35"


if __name__ == "__main__":
    unittest.main()
