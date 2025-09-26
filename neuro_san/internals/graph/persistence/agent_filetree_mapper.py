
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
from pathlib import Path

from neuro_san.interfaces.agent_name_mapper import AgentNameMapper


class AgentFileTreeMapper(AgentNameMapper):

    def agent_name_to_filepath(self, agent_name: str) -> str:
        """
        Converts an agent name to file path to this agent definition file.
        """
        return agent_name.replace("-", "/")

    def filepath_to_agent_network_name(self, filepath: str) -> str:
        """
        Converts a file path to agent definition file (relative to registry root directory)
        to agent network name identifying it to the service.
        """
        return str(Path(filepath.replace("/", "-")).with_suffix(""))
