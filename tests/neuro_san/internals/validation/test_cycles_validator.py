
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT
from typing import Any
from typing import Dict
from typing import List

from unittest import TestCase

from neuro_san import REGISTRIES_DIR
from neuro_san.internals.graph.persistence.agent_network_restorer import AgentNetworkRestorer
from neuro_san.internals.graph.registry.agent_network import AgentNetwork
from neuro_san.internals.validation.cycles_network_validator import CyclesNetworkValidator


class TestCyclesNetworkValidator(TestCase):
    """
    Unit tests for CyclesNetworkValidator class.
    """

    @staticmethod
    def restore(file_reference: str):
        # Open a known good network file
        restorer = AgentNetworkRestorer()
        hocon_file: str = REGISTRIES_DIR.get_file_in_basis(file_reference)
        agent_network: AgentNetwork = restorer.restore(file_reference=hocon_file)
        config: Dict[str, Any] = agent_network.get_config()
        return config

    def test_assumptions(self):
        """
        Can we construct?
        """
        validator = CyclesNetworkValidator()
        self.assertIsNotNone(validator)

    def test_empty(self):
        """
        Tests empty network
        """
        validator = CyclesNetworkValidator()

        errors: List[str] = validator.validate(None)
        self.assertEqual(1, len(errors))

        errors: List[str] = validator.validate({})
        self.assertEqual(1, len(errors))

    def test_valid(self):
        """
        Tests a valid network
        """
        validator = CyclesNetworkValidator()

        # Open a known good network file
        config: Dict[str, Any] = self.restore("hello_world.hocon")

        errors: List[str] = validator.validate(config)

        failure_message: str = None
        if len(errors) > 0:
            failure_message = errors[0]
        self.assertEqual(0, len(errors), failure_message)

    def test_cycles(self):
        """
        Tests a network where there is a cycle.
        These can actually be ok, but we want to test that we can detect them.
        """
        validator = CyclesNetworkValidator()

        # Open a known good network file
        config: Dict[str, Any] = self.restore("esp_decision_assistant.hocon")

        # Invalidate per the test - add a link from the predictor to the prescriptor
        config["tools"][2]["tools"] = ["prescriptor"]

        errors: List[str] = validator.validate(config)

        self.assertEqual(1, len(errors), errors[-1])
