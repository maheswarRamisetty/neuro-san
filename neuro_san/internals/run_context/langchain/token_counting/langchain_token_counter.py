
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
import copy
from typing import Any
from typing import Awaitable
from typing import Dict
from typing import List
from typing import Union

from asyncio import Task
from contextvars import Context
from contextvars import ContextVar
from contextvars import copy_context
from time import time

from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.language_models.base import BaseLanguageModel

from leaf_common.asyncio.asyncio_executor import AsyncioExecutor

from neuro_san.internals.interfaces.context_type_llm_factory import ContextTypeLlmFactory
from neuro_san.internals.interfaces.invocation_context import InvocationContext
from neuro_san.internals.journals.originating_journal import OriginatingJournal
from neuro_san.internals.messages.agent_message import AgentMessage
from neuro_san.internals.messages.origination import Origination
from neuro_san.internals.run_context.langchain.token_counting.get_llm_token_callback import get_llm_token_callback
from neuro_san.internals.run_context.langchain.token_counting.get_llm_token_callback import llm_token_callback_var

# Keep a ContextVar for the origin info.  We do this because the
# langchain callbacks this stuff is based on also uses ContextVars
# and we want to be sure these are in sync.
# See: https://docs.python.org/3/library/contextvars.html
ORIGIN_INFO: ContextVar[str] = ContextVar('origin_info', default=None)


class LangChainTokenCounter:
    """
    Helps with per-llm means of counting tokens.
    Main entrypoint is count_tokens().

    Notes as to how each BaseLanguageModel/BaseChatModel should be configured
    are in get_callback_for_llm()
    """

    # This is for calculating model token dict by subtracting it from cumulative models token dict
    previous_cumulative_models_token_dict: Dict[str, Any] = {}

    # This is for calculating cumulative time taken by the frontman (LLMs + overhead)
    time_per_iter: List[float] = []

    # This is for keeping track of frontman
    _instance_count = 0

    def __init__(self, llm: BaseLanguageModel,
                 invocation_context: InvocationContext,
                 journal: OriginatingJournal):
        """
        Constructor

        :param llm: The Llm to monitor for tokens
        :param invocation_context: The InvocationContext
        :param journal: The OriginatingJournal which through which this
                    will send token count AGENT messages
        """
        self.llm: BaseLanguageModel = llm
        self.invocation_context: InvocationContext = invocation_context
        self.journal: OriginatingJournal = journal
        self.debug: bool = False

        # Frontman has instance number = 0
        self.instance_number = LangChainTokenCounter._instance_count
        # Other agents have higher instance number because they were called later
        LangChainTokenCounter._instance_count += 1

    async def count_tokens(self, awaitable: Awaitable) -> Any:
        """
        Counts the tokens (if possible) from what happens inside the awaitable
        within a separate context.  If tokens are counted, they are added to
        the InvocationContext's request_reporting and sent over the message queue
        via the journal

        Recall awaitables are a full async method call with args.  That is, where you would expect to
                baz = await myinstance.foo(bar)
        you instead do
                baz = await token_counter.count_tokens(myinstance.foo(bar)).

        :param awaitable: The awaitable whose tokens we wish to count.
        :return: Whatever the awaitable would return
        """

        retval: Any = None
        llm_factory: ContextTypeLlmFactory = self.invocation_context.get_llm_factory()
        llm_infos: Dict[str, Any] = llm_factory.llm_infos
        # Take a time stamp so we measure another thing people care about - latency.
        start_time: float = time()

        # Attempt to count tokens/costs while invoking the agent.
        # The means by which this happens is on a per-LLM basis, so get the right hook
        # given the LLM we've got.
        callback: Union[AsyncCallbackHandler, BaseCallbackHandler] = None

        # Record origin information in our own context var so we can associate
        # with the langchain callback context vars more easily.
        origin: List[Dict[str, Any]] = self.journal.get_origin()
        origin_str: str = Origination.get_full_name_from_origin(origin)
        ORIGIN_INFO.set(origin_str)

        old_callback: Union[AsyncCallbackHandler, BaseCallbackHandler] = None
        callback_var: ContextVar = llm_token_callback_var
        old_callback = callback_var.get()

        # Use the context manager to count tokens as per
        #   https://python.langchain.com/docs/how_to/llm_token_usage_tracking/#using-callbacks
        #
        # Caveats:
        # * In using this context manager approach, any tool that is called
        #   also has its token counts contributing to its callers for better or worse.
        # * As of 2/21/25, it seems that tool-calling agents (branch nodes) are not
        #   registering their tokens correctly. Not sure if this is a bug in langchain
        #   or there is something we are not doing in that scenario that we should be.
        # * As of 8/21/25, placing the journaling callback in the invoke config
        #   appears to change the context manager’s behavior. The returned tokens from callback
        #   are now limited to the calling agent only, and no longer include those
        #   from downstream (chained) agents. However, `cumulative_models_token_dict` is added
        #   to the `LlmTokenCallbackHandler` to collect token stats of each model call.
        with get_llm_token_callback(llm_infos) as callback:
            # Create a new context for different ContextVar values
            # and use the create_task() to run within that context.
            new_context: Context = copy_context()
            task: Task = new_context.run(self.create_task, awaitable)
            retval = await task

        callback_var.set(old_callback)

        # Figure out how much time our agent took.
        end_time: float = time()
        time_taken_in_seconds: float = end_time - start_time

        await self.report(callback, time_taken_in_seconds)

        return retval

    def create_task(self, awaitable: Awaitable) -> Task:
        """
        Riffed from:
        https://stackoverflow.com/questions/78659844/async-version-of-context-run-for-context-vars-in-python-asyncio
        """
        executor: AsyncioExecutor = self.invocation_context.get_asyncio_executor()
        origin_str: str = ORIGIN_INFO.get()
        task: Task = executor.create_task(awaitable, origin_str)

        if self.debug:
            # Print to be sure we have a different callback object.
            oai_call = llm_token_callback_var.get()
            print(f"origin is {origin_str} callback var is {id(oai_call)}")

        return task

    async def report(self, callback: Union[AsyncCallbackHandler, BaseCallbackHandler], time_taken_in_seconds: float):
        """
        Report on the token accounting results of the callback

        :param callback: An AsyncCallbackHandler or BaseCallbackHandle instance that contains token counting information
        :param time_taken_in_seconds: The amount of time the awaitable took in count_tokens()
        """

        agent_name: str = ORIGIN_INFO.get().split(".")[-1]
        # Token counting results are collected in the callback.
        # Create a token counding dictionary for each agent
        agent_token_dict: Dict[str, Any] = self._generate_agent_token_dict(callback, time_taken_in_seconds, agent_name)
        if self.journal is not None:
            # We actually have a token dictionary to report, so go there.
            agent_message = AgentMessage(structure=agent_token_dict)
            await self.journal.write_message(agent_message)

        # Accumulate what we learned about tokens to request reporting.
        # For now we just overwrite the one key because we know
        # the last one out will be the front man, and as of 2/21/25 his stats
        # are cumulative.  At some point we might want a finer-grained breakdown
        # that perhaps contributes to a service/er-wide periodic token stats breakdown
        # of some kind.  For now, get something going.
        #
        # Update (8/21/25):
        # Placing the journaling callback in the invoke config changes the context
        # manager’s behavior. The returned tokens from the callback are now limited
        # to the calling agent only, not downstream (chained) agents.
        # Instead, `cumulative_models_token_dict` has been added to
        # `LlmTokenCallbackHandler` to collect per-model token stats.
        # Since the frontman is always the last to finish, by the time it exits,
        # `cumulative_models_token_dict` is complete and ready to report.

        # If instance number is zero then it is frontman
        if self.instance_number == 0:
            self.time_per_iter.append(time_taken_in_seconds)
            request_reporting: Dict[str, Any] = self.invocation_context.get_request_reporting()
            cumulative_models_token_dict: Dict[str, Any] = callback.cumulative_models_token_dict
            self._generate_token_reporting(cumulative_models_token_dict, request_reporting)

            # Copy the cumulative models token dict for subtraction in the next iteration
            LangChainTokenCounter.previous_cumulative_models_token_dict = \
                copy.deepcopy(callback.cumulative_models_token_dict)
            # Reset counter for the next iteration
            LangChainTokenCounter._instance_count = 0

    def _generate_agent_token_dict(
            self,
            callback: Union[AsyncCallbackHandler, BaseCallbackHandler],
            time_taken_in_seconds: float,
            agent_name: str
    ) -> Dict[str, Any]:
        """
        Generate the token counting dictionary for journals

        :param callback: An AsyncCallbackHandler or BaseCallbackHandler instance that contains
                            token counting information
        :param time_taken_in_seconds: The amount of time the awaitable took in count_tokens()
        :param agent_name: Name of the agent responsible for the token dictionary
        :return: Formatted token dictionary
        """

        # Organize the token dict for each agent to be the same format
        agent_token_dict = {
            f"{agent_name}_token_accounting": {
                "total_tokens": callback.total_tokens,
                "prompt_tokens": callback.prompt_tokens,
                "completion_tokens": callback.completion_tokens,
                "successful_requests": callback.successful_requests,
                "total_cost": callback.total_cost,
                "time_taken_in_seconds": time_taken_in_seconds,
                "caveats": [
                    "Token usage is tracked at the agent level.",
                    "Token counts are approximate and estimated using tiktoken.",
                    "time_taken_in_seconds includes overhead from Langchain and Neuro-SAN"
                ]
            }
        }

        return agent_token_dict

    def _generate_token_reporting(
            self,
            cumulative_models_token_dict: Dict[str, Any],
            request_reporting: Dict[str, Any]
    ):
        """
        Generate token accounting reports with cumulative and delta calculations.
        :param cumulative_models_token_dict: Cumulative values of stats for each model
        :param request_reporting: Dictionary to report after frontman finishes
        """

        # Calculate model token stats for this iteration
        models_token_dict = self._calculate_models_token_dict(cumulative_models_token_dict)

        # Network stats: time includes overhead from both LangChain and Neuro-SAN.
        # Model stats: time reflects only the LLM execution.

        # Report network and models token accounting

        # Time for this iteration is the last one in the "time_per_iter" list
        request_reporting["network_token_accounting"] = self._sum_all_tokens(
            models_token_dict, self.time_per_iter[-1]
        )
        request_reporting["models_token_accounting"] = models_token_dict

        # Report cumulative network and models token accounting

        # Sum "time_per_iter" list to get cumulative time
        request_reporting["cumulative_network_token_accounting"] = self._sum_all_tokens(
            cumulative_models_token_dict, sum(self.time_per_iter)
        )
        request_reporting["cumulative_models_token_accounting"] = cumulative_models_token_dict

    def _calculate_models_token_dict(self, cumulative_models_token_dict) -> Dict[str, Any]:
        """
        Calculate models token dict from the difference between cumulative and previous cumulative token stats.
        :param cumulative_models_token_dict: Cumulative values of token stats for each model
        :return: Values of token stats for each model invoked in this iteration
        """
        models_token_dict: Dict[str, Any] = {}
        for provider, models in cumulative_models_token_dict.items():
            models_token_dict[provider] = {}
            for model, cumulative_stats in models.items():
                one_iteration_stats = {}
                prev_cumulative_stats = self.previous_cumulative_models_token_dict.get(provider, {}).get(model, {})
                # Calculate value of each metric
                for metric, cumulative_value in cumulative_stats.items():
                    # If it is a model that doen not existing previously, set the metric to zero
                    prev_cumulative_value = prev_cumulative_stats.get(metric, 0)
                    one_iteration_stats[metric] = cumulative_value - prev_cumulative_value

                models_token_dict[provider][model] = one_iteration_stats

        return models_token_dict

    def _sum_all_tokens(self, token_dict: Dict[str, Any], time_value: float) -> Dict[str, Any]:
        """
        Sum all token metrics across providers and models, **excluding time**.
        :param token_dict: Models token dict to aggregate into network stats
        :param time_value: Time taken for frontman to finish
        :return: Token stats of the entire network, either cumulative or single iteration
        """
        aggregated: Dict[str, Any] = {}
        for models in token_dict.values():
            for model_stats in models.values():
                for metric, value in model_stats.items():
                    if metric != "time_taken_in_seconds":
                        aggregated[metric] = aggregated.get(metric, 0) + value

        aggregated["time_taken_in_seconds"] = time_value

        return aggregated
