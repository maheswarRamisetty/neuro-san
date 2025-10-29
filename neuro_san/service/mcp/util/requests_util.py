
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

import html

from typing import Union


class RequestsUtil:
    """
    Utility helper class for MCP requests processing.
    """

    @staticmethod
    def safe_request_id(request_id: Union[int, str]) -> Union[int, str]:
        """
        Return HTML-safe representation of user request id to be sent back in MCP response.
        :param request_id: MCP request id (as received from user);
        :return: HTML-escaped request id
        """
        if isinstance(request_id, str):
            return html.escape(request_id)
        return request_id
