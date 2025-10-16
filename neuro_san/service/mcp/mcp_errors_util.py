
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
"""
See class comment for details
"""
from typing import Any
from typing import Dict

from neuro_san.service.mcp.mcp_errors import MCPError

class MCPErrorsUtil:

    @classmethod
    def get_protocol_error(cls, request_id, error: MCPError, extra_msg: str = None) -> Dict[str, Any]:
        msg: str = error.str_label
        if extra_msg is not None:
            msg = f"{msg}: {extra_msg}"
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": error.num_value,
                "message": msg
            }
        }

    @classmethod
    def get_tool_error(cls, request_id, error_msg: str) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": error_msg
                    }
                ],
                "isError": True
            }
        }
