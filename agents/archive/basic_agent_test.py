# *************** Basic Agent ***************

import os
import openai
from pydantic import BaseModel, Field
from langchain.agents import tool
from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
#from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import MessagesPlaceholder
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#from langchain.tools.render import format_tool_to_openai_function # TO BE DEPRECATED
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.agents.format_scratchpad.openai_functions import format_to_openai_functions
from langchain.memory import ConversationBufferMemory # TO BE DEPRECATED, BUT IS THE ONLY COMPATIBLE OPTION
#from langchain_core.messages import BaseMessage, AIMessage
#from langchain_core.chat_history import BaseChatMessageHistory
#from typing import List
import requests, wikipedia, datetime



from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Define the input schema
class OpenMeteoInput(BaseModel):
    latitude: float = Field(..., description="Latitude of the location to fetch weather data for")
    longitude: float = Field(..., description="Longitude of the location to fetch weather data for")

@tool(args_schema=OpenMeteoInput)
def get_current_temperature(latitude: float, longitude: float) -> dict:
    """Fetch current temperature for given coordinates."""

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    # Parameters for the request
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'hourly': 'temperature_2m',
        'forecast_days': 1,
    }

    # Make the request
    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        results = response.json()
    else:
        raise Exception(f"API Request failed with status code: {response.status_code}")

    current_utc_time = datetime.datetime.utcnow()
    time_list = [datetime.datetime.fromisoformat(time_str.replace('Z', '+00:00')) for time_str in
                 results['hourly']['time']]
    temperature_list = results['hourly']['temperature_2m']

    closest_time_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - current_utc_time))
    current_temperature = temperature_list[closest_time_index]

    return f'The current temperature is {current_temperature}°C'

@tool
def search_wikipedia(query: str) -> str:
    """Run Wikipedia search and get page summaries."""
    page_titles = wikipedia.search(query)
    summaries = []
    for page_title in page_titles[: 3]:
        try:
            wiki_page =  wikipedia.page(title=page_title, auto_suggest=False)
            summaries.append(f"Page: {page_title}\nSummary: {wiki_page.summary}")
        except (
            self.wiki_client.exceptions.PageError,
            self.wiki_client.exceptions.DisambiguationError,
        ):
            pass
    if not summaries:
        return "No good Wikipedia Search Result was found"
    return "\n\n".join(summaries)

tool_dict = {
    "search_wikipedia": search_wikipedia,
    "get_current_temperature": get_current_temperature,
}
tools = list(tool_dict.values())

functions = [convert_to_openai_function(f) for f in tools]#format_tool_to_openai_function(f) for f in tools]
#model = ChatOpenAI(temperature=0).bind(functions=functions)
model = ChatOpenAI(model='gpt-4o', temperature=0).bind(functions=functions)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    #MessagesPlaceholder(variable_name="chat_history"), # Used with ConversationalBufferMemory
    #MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain

def run_agent(user_input, tool_dict):#): # Has no memory
    intermediate_steps = []
    while True:
        result = agent_chain.invoke({
            "input": user_input,
            "intermediate_steps": intermediate_steps
        })
        if isinstance(result, AgentFinish):
            return result
        tool = tool_dict[result.tool]

        observation = tool.run(result.tool_input)
        intermediate_steps.append((result, observation))

#run_agent("what is the weather is sf?", tool_dict)

#agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True)

#agent_executor.invoke({"input": "what is the weather in visakhapatnam?"})

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    MessagesPlaceholder(variable_name="chat_history"), # Used with ConversationalBufferMemory
    #MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain

memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory) # Memory integrated

#agent_executor.invoke({"input": "my name is bob"})

#agent_executor.invoke({"input": "whats my name"})

#agent_executor.invoke({"input": "whats the weather in sf?"})

'''
class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

# Here we use a global variable to store the chat message history.
# This will make it easier to inspect it to see the underlying results.
store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful but sassy assistant"),
    #MessagesPlaceholder(variable_name="chat_history"),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

chain = prompt | model | OpenAIFunctionsAgentOutputParser()

agent_chain = RunnablePassthrough.assign(
    agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
) | chain

chain_with_history = RunnableWithMessageHistory(
    agent_chain,
    get_session_history=get_by_session_id,
    input_messages_key="question",
    history_messages_key="history",
)

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
))

# Uses the store defined in the example above.
print(store)  # noqa: T201

print(chain_with_history.invoke(  # noqa: T201
    {"ability": "math", "question": "What's its inverse"},
    config={"configurable": {"session_id": "foo"}}
))

print(store)  # noqa: T201

agent_chain.invoke({"question": "What does cosine mean?",
                    "intermediate_steps": []
        }, config={"configurable": {"session_id": "foo"})
'''