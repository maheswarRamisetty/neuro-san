
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san-studio SDK Software in commercial settings.
#
# END COPYRIGHT

from typing import Any
from typing import Dict
from typing import List

from logging import getLogger
from logging import Logger

from neuro_san.internals.interfaces.agent_network_validator import AgentNetworkValidator


class KeywordNetworkValidator(AgentNetworkValidator):
    """
    AgentNetworkValidator that looks for correct keywords in an agent network
    """

    def __init__(self):
        """
        Constructor
        """
        self.logger: Logger = getLogger(self.__class__.__name__)

    def validate(self, agent_network: Dict[str, Any]) -> List[str]:
        """
        Validation the agent network.

        :param agent_network: The agent network to validate
        :return: List of errors indicating agents and missing keywords
        """
        errors: List[str] = []

        self.logger.info("Validating agent network keywords...")

        if agent_network is None:
            errors.append("Agent network is empty.")
            return errors

        # We can validate either from a top-level agent network,
        # or from the list of tools from the agent spec.
        agent_network = agent_network.get("tools", agent_network)

        # Currently, only required "instructions" for non-function agents.
        for agent_name, agent in agent_network.items():
            if agent.get("instructions") == "":
                error_msg = f"{agent_name} 'instructions' cannot be empty."
                errors.append(error_msg)

        # Only warn if there is a problem
        if len(errors) > 0:
            self.logger.warning(str(errors))

        return errors
