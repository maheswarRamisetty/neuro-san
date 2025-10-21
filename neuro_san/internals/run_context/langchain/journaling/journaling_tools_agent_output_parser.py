
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
from typing import TypeVar

from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables.config import RunnableConfig

# Bizarre convention from the superclass to adhere to overridden method.
T = TypeVar("T")


class JournalingToolsAgentOutputParser(ToolsAgentOutputParser):
    """
    ToolsAgentOutputParser implementation that intercepts agent-level chatter

    We use this to intercept the "Invoking <agent> with <params>" kinds of messages
    to stream them back to the client as AgentMessages.
    """

    # pylint: disable=redefined-builtin
    async def ainvoke(self, input: str | BaseMessage,
                      config: RunnableConfig | None = None,
                      **kwargs: Any | None) -> T:
        """
        Comments and method signature from superclass method.
        """
        use_input = input
        if isinstance(input, Dict):
            messages: List[BaseMessage] = input.get("messages", [])
            use_input = None
            for message in messages:
                if isinstance(message, AIMessage):
                    use_input = message

            if use_input is None:
                return None

        return await super().ainvoke(use_input, config, **kwargs)
