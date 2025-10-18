
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
from typing import List

from neuro_san.internals.interfaces.dictionary_validator import DictionaryValidator


class MCPRequestValidator(DictionaryValidator):

    def __init__(self, validation_schema: Dict[str, Any]):
        self.validation_schema = validation_schema

    def validate(self, candidate: Dict[str, Any]) -> List[str]:
        # Validate incoming RPC structure against MCP schema:
        #jsonschema.validate(instance=data, schema=self.mcp_protocol_schema)
        #except jsonschema.exceptions.ValidationError as exc:
        # except jsonschema.exceptions.ValidationError as exc:
        #     error_msg: Dict[str, Any] =\
        #         MCPErrorsUtil.get_protocol_error(request_id, MCPError.InvalidRequest, str(exc))
        #     self.set_status(400)
        #     self.write(error_msg)
        #     self.logger.error(self.get_metadata(), "error: Invalid JSON/RPC request")

        return None