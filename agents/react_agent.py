# *************** Basic Agent ***************

import os
import openai
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions
from langchain.memory import ConversationBufferMemory # TO BE DEPRECATED, BUT IS THE ONLY COMPATIBLE OPTION
import configparser
from dotenv import load_dotenv, find_dotenv



_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
config = configparser.ConfigParser()
config.read('config.ini')
OPENAI_LLM = config.get('SETTINGS', 'OPENAI_LLM')


class BasicAgent:

    def __init__(self, starter="You are helpful but sassy assistant", tool_dict={}):
        self.tool_dict = tool_dict.copy()
        tools = list(tool_dict.values())
        functions = [convert_to_openai_function(f) for f in tools]
        model = ChatOpenAI(model=OPENAI_LLM, temperature=0).bind(functions=functions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{starter}"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        chain = prompt | model | OpenAIFunctionsAgentOutputParser()
        agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | chain
        self.model = agent_chain
        agent_executor = AgentExecutor(agent=agent_chain,  # ERROR
                                       tools=list(self.tool_dict.values()),
                                       handle_parsing_errors=True,
                                       verbose=True)
        self.model2 = agent_executor

    def __call__(self, message):
        try:
            result = self.execute_manual(message)
        except:
            result = self.execute(message)
        finally:
            return result

    def execute_manual(self, user_input): # AKA run_agent(), manual function call and has no memory
        intermediate_steps = []
        while True:
            result = self.model.invoke({#self.agent_chain.invoke({
                "input": user_input,
                "intermediate_steps": intermediate_steps
            })
            if isinstance(result, AgentFinish):
                return result
            tool = self.tool_dict[result.tool]
            observation = tool.run(result.tool_input)
            intermediate_steps.append((result, observation))

    def execute(self, user_input): # AKA agent_executor(), has no memory
        #agent_executor = AgentExecutor(agent=self.model, # ERROR
        #                               tools=list(self.tool_dict.values()),
        #                               handle_parsing_errors=True,
        #                               verbose=True)
        #agent_executor.invoke({"input": f"{user_input}"})
        self.model2.invoke({"input": f"{user_input}"})



class ChatAgent:

    def __init__(self, starter="You are helpful but sassy assistant", tool_dict={}):
        self.tool_dict = tool_dict
        tools = list(tool_dict.values())
        functions = [convert_to_openai_function(f) for f in tools]
        model = ChatOpenAI(model=OPENAI_LLM, temperature=0).bind(functions=functions)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"{starter}"),
            MessagesPlaceholder(variable_name="chat_history"),  # Used with ConversationalBufferMemory
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        chain = prompt | model | OpenAIFunctionsAgentOutputParser()
        agent_chain = RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_to_openai_functions(x["intermediate_steps"])
        ) | chain
        self.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
        agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=False, memory=self.memory) # Memory integrated
        self.model = agent_executor
        print('initializaed')

    def __call__(self, message):
        self.execute(message)

    def execute(self, user_input): # AKA agent_executor(), remembers chat history
        self.model.invoke({"input": f"{user_input}"})
        print('executing')
