# *************** CodeAct Agent (Unused for project, potential future work!) ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.manipulator import groupby_aggregate, pivot_table, transpose_df, \
    select_columns, drop_columns, discretize_column, rename_column #add, subtract, multiply, divide
import os
import openai
import asyncio
from llama_index.llms.openai import OpenAI
from typing import Any, Dict, Tuple
import io
import contextlib
import ast
import traceback
from llama_index.core.agent.workflow import CodeActAgent
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
    AgentStream,
)
import configparser
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
config = configparser.ConfigParser()
OPENAI_LLM = config.get('SETTINGS', 'OPENAI_LLM')

# Configure the LLM
llm = OpenAI(model=OPENAI_LLM, api_key=openai.api_key)


# =================================================
# ================== Definition ===================
# =================================================

class SimpleCodeExecutor:
    """
    A simple code executor that runs Python code with state persistence.

    This executor maintains a global and local state between executions,
    allowing for variables to persist across multiple code runs.

    NOTE: not safe for production use! Use with caution.
    """

    def __init__(self, locals: Dict[str, Any], globals: Dict[str, Any]):
        """
        Initialize the code executor.

        Args:
            locals: Local variables to use in the execution context
            globals: Global variables to use in the execution context
        """
        # State that persists between executions
        self.globals = globals
        self.locals = locals

    def execute(self, code: str) -> Tuple[bool, str, Any]:
        """
        Execute Python code and capture output and return values.

        Args:
            code: Python code to execute

        Returns:
            Dict with keys `success`, `output`, and `return_value`
        """
        # Capture stdout and stderr
        stdout = io.StringIO()
        stderr = io.StringIO()

        output = ""
        return_value = None
        try:
            # Execute with captured output
            with contextlib.redirect_stdout(
                stdout
            ), contextlib.redirect_stderr(stderr):
                # Try to detect if there's a return value (last expression)
                try:
                    tree = ast.parse(code)
                    last_node = tree.body[-1] if tree.body else None

                    # If the last statement is an expression, capture its value
                    if isinstance(last_node, ast.Expr):
                        # Split code to add a return value assignment
                        last_line = code.rstrip().split("\n")[-1]
                        exec_code = (
                            code[: -len(last_line)]
                            + "\n__result__ = "
                            + last_line
                        )

                        # Execute modified code
                        exec(exec_code, self.globals, self.locals)
                        return_value = self.locals.get("__result__")
                    else:
                        # Normal execution
                        exec(code, self.globals, self.locals)
                except:
                    # If parsing fails, just execute the code as is
                    exec(code, self.globals, self.locals)

            # Get output
            output = stdout.getvalue()
            if stderr.getvalue():
                output += "\n" + stderr.getvalue()

        except Exception as e:
            # Capture exception information
            output = f"Error: {type(e).__name__}: {str(e)}\n"
            output += traceback.format_exc()

        if return_value is not None:
            output += "\n\n" + str(return_value)

        return output


# =================================================
# ================= Initializing ==================
# =================================================
'''
code_executor = SimpleCodeExecutor(
    # give access to our functions defined above
    locals={
        "add": add,
        "subtract": subtract,
        "multiply": multiply,
        "divide": divide,
    },
    globals={
        # give access to all builtins
        "__builtins__": __builtins__,
        # give access to numpy
        "np": __import__("numpy"),
        "pd": __import__("pandas"),
        "sk": __import__("sklearn"),
        "sm": __import__("statsmodels"),
        "sns": __import__("seaborn"),
        "matplotlib": __import__("matplotlib"),
    },
)

agent = CodeActAgent(
    code_execute_fn=code_executor.execute,
    llm=llm,
    tools=[add, subtract, multiply, divide],
)

# context to hold the agent's session/state/chat history
ctx = Context(agent)

from llama_index.core.agent.workflow import (
    ToolCall,
    ToolCallResult,
    AgentStream,
)


async def run_agent_verbose(agent, ctx, query):
    handler = agent.run(query, ctx=ctx)
    print(f"User:  {query}")
    async for event in handler.stream_events():
        if isinstance(event, ToolCallResult):
            print(
                f"\n-----------\nCode execution result:\n{event.tool_output}"
            )
        elif isinstance(event, ToolCall):
            print(f"\n-----------\nParsed code:\n{event.tool_kwargs['code']}")
        elif isinstance(event, AgentStream):
            print(f"{event.delta}", end="", flush=True)

    return await handler
'''
# =================================================
# ==================== Calling ====================
# =================================================
'''
#response = await run_agent_verbose(
#    agent, ctx, "Calculate the sum of all numbers from 1 to 10"
#)

async def call_agent(user_input):
    response = await run_agent_verbose(
        agent, ctx, user_input
        )
    return response

response = asyncio.run(call_agent("Calculate the sum of all numbers from 1 to 10"))
print(response)
'''
# =================================================

class CodeactAgent(BaseAgent):
    """
    This is LlamaIndex's CodeActAgent which we borrow to conduct Data Manipulation and Transformation on the dataframe passed to it.
    It is called by the RouterAgent upon receiving a data manipulation task.
    """

    # Field(s) (Class)
    ctx = None
    agent = None

    # Public Method(s)
    async def run_agent_verbose(agent, ctx, query):
        handler = agent.run(query, ctx=ctx)
        print(f"User:  {query}")
        async for event in handler.stream_events():
            if isinstance(event, ToolCallResult):
                print(
                    f"\n-----------\nCode execution result:\n{event.tool_output}"
                )
            elif isinstance(event, ToolCall):
                print(f"\n-----------\nParsed code:\n{event.tool_kwargs['code']}")
            elif isinstance(event, AgentStream):
                print(f"{event.delta}", end="", flush=True)

        return await handler

    async def call_agent(user_input):
        response = await CodeactAgent.run_agent_verbose(
            CodeactAgent.agent, CodeactAgent.ctx, user_input
            )
        return response
    
    # Private Method(s)
    
    # Constructor(s)
    def __init__(self, dataframe):
        self.dataframe = dataframe
        code_executor = SimpleCodeExecutor(
            # give access to our functions defined above
            locals={
                #"dataframe" : self.dataframe,
                "groupby_aggregate": groupby_aggregate,
                "pivot_table": pivot_table,
                "transpose_df": transpose_df,
                "select_columns": select_columns,
                "drop_columns": drop_columns,
                "discretize_column": discretize_column,
                "rename_column": rename_column,
            },
            globals={
                # give access to all builtins
                "__builtins__": __builtins__,
                # give access to numpy
                "np": __import__("numpy"),
                "pd": __import__("pandas"),
                "sk": __import__("sklearn"),
                "sm": __import__("statsmodels"),
                "sns": __import__("seaborn"),
                "matplotlib": __import__("matplotlib"),
            },
        )

        CodeactAgent.agent = CodeActAgent(
            code_execute_fn=code_executor.execute,
            llm=llm,
            tools=[groupby_aggregate, pivot_table, transpose_df, \
                   select_columns, drop_columns, discretize_column, rename_column],#[add, subtract, multiply, divide],
        )

        # context to hold the agent's session/state/chat history
        CodeactAgent.ctx = Context(CodeactAgent.agent)
        # End of initialization

    # Call Override(s)
    def __call__(self, message):
        try:
            #result = self.agent.invoke(message)
            result = asyncio.run(CodeactAgent.call_agent(message))
        except Exception as exc:
            result = ''
            return exc
        except:
            return 'Base Exception Error...'
        self.chat_history += [result]
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

