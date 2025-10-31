
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
from typing import Union


class MetadataUtil:
    """
    Utilities to deal with metadata dictionaries.
    """

    @staticmethod
    def minimize_metadata(request_metadata: Dict[str, Any],
                          keys: Union[str, List[str]]) -> Dict[str, Any]:
        """
        :param request_metadata: The raw request metadata dictionary that could easily contain
                    more keys than we want to send to the UsageLogger.
        :param keys: The keys we want to send to the UsageLogger
                This can either be a single string with space-delimited keys, or a list of keys
        :return: A minimized dictionary that only sends the keys we need to the UsageLogger.
                The idea is that this prevents the UsageLogger from getting potentially
                sensitive information it shouldn't really have.

                If the requested keys in the metadata are not there, they will also not appear
                in the returned minimized dictionary.
        """
        minimized: Dict[str, Any] = {}

        # If there are no keys, there is nothing to fill.
        if not keys:
            return minimized

        keys_list: List[str] = []

        # Check if keys is a string or a list
        if isinstance(keys, str):
            keys_list = keys.split(" ")
        elif isinstance(keys, List):
            keys_list = keys

        for key in keys_list:

            if not key:
                # Skip any empty key split from the list. Allows for multi-spaces.
                continue

            value: str = request_metadata.get(key)
            if value:
                minimized[key] = value

        return minimized
