
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

from neuro_san.internals.messages.origination import Origination


class ToolArgumentReporting:
    """
    Utility class to assist in preparing arguments dictionaries when reporing starting a tool.
    """

    @staticmethod
    def prepare_tool_start_dict(inputs: Dict[str, Any],
                                origin: List[Dict[str, Any]] = None) -> Dict[str, Any]:

        # Combine the original tool inputs with origin metadata
        tool_args: Dict[str, Any] = inputs.copy()

        if origin is not None:
            tool_args["origin"] = origin

            full_name: str = Origination.get_full_name_from_origin(origin)
            tool_args["origin_str"] = full_name

        # Remove any reservationist from the args as that will not transfer over the wire
        if "reservationist" in tool_args:
            del tool_args["reservationist"]

        # Remove any progress_reporter from the args as that will not transfer over the wire
        if "progress_reporter" in tool_args:
            del tool_args["progress_reporter"]

        # Create a dictionary for a future journal entry for this invocation
        tool_args_dict: Dict[str, Any] = {
            "tool_start": True,
            "tool_args": tool_args
        }

        return tool_args_dict
