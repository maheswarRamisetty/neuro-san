
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
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from neuro_san.internals.messages.traced_message import TracedMessage


class AgentFrameworkMessage(TracedMessage):
    """
    TracedMessage implementation of a message from the agent framework
    """

    structure: Optional[Dict[str, Any]] = None
    sly_data: Optional[Dict[str, Any]] = None
    chat_context: Optional[Dict[str, Any]] = None

    type: Literal["agent-framework"] = "agent-framework"

    def __init__(self, content: Union[str, List[Union[str, Dict]]] = "",
                 chat_context: Dict[str, Any] = None,
                 sly_data: Dict[str, Any] = None,
                 structure: Dict[str, Any] = None,
                 other: AgentFrameworkMessage = None,
                 **kwargs: Any) -> None:
        """
        Pass in content as positional arg.

        :param content: The string contents of the message.
        :param chat_context: A dictionary that fully desbribes the state of play
                    of the chat conversation such that when it is passed on to a
                    different server, the conversation can continue uninterrupted.
        :param sly_data: A dictionary of private data, separate from the chat stream.
        :param structure: A dictionary previously extracted from the content
                        that had been optionally detected by the system as JSON text.
                        The idea is to have the server do the hard parsing so the
                        multitude of clients do not have to rediscover how to best do it.
        :param kwargs: Additional fields to pass to the superclass
        """
        super().__init__(content=content, other=other, **kwargs)
        self.chat_context: Dict[str, Any] = chat_context
        self.sly_data: Dict[str, Any] = sly_data
        self.structure: Dict[str, Any] = structure

    @property
    def lc_kwargs(self) -> Dict[str, Any]:
        """
        :return: the keyword arguments for serialization.
        """
        return {
            "content": self.content,
            "structure": self.structure,
            "sly_data": self.sly_data,
            "chat_context": self.chat_context,
        }
