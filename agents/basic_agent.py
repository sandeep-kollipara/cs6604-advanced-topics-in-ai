# *************** Basic Agent ***************

from langchain.schema.agent import AgentFinish
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.prompts import MessagesPlaceholder
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
#from langchain.tools.render import format_tool_to_openai_function # TO BE DEPRECATED
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain.agents import AgentExecutor
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents.format_scratchpad import format_to_openai_functions
#from langchain.memory import ConversationBufferMemory # TO BE DEPRECATED

#functions = [format_tool_to_openai_function(f) for f in tools]
functions = [convert_to_openai_function(f) for f in tools]
#model = ChatOpenAI(temperature=0).bind(functions=functions)
model = ChatOpenAI(model='gpt-4o', temperature=0).bind(functions=functions)

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


tool_dict = {
    #"search_wikipedia": search_wikipedia,
    #"get_current_temperature": get_current_temperature,
}
tools = list(tool_dict.values())

def run_agent(user_input, tool_dict):#):
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

run_agent("what is the weather is sf?")

#memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

#agent_executor = AgentExecutor(agent=agent_chain, tools=tools, verbose=True, memory=memory)

#agent_executor.invoke({"input": "my name is bob"})

#agent_executor.invoke({"input": "whats my name"})

#agent_executor.invoke({"input": "whats the weather in sf?"})


chain_with_history = RunnableWithMessageHistory(
    agent_chain,
    input_messages_key="question",
    history_messages_key="history",
)



