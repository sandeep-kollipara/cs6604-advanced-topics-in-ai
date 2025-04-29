# *************** Encoder Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
import os
import openai
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
import configparser
from dotenv import load_dotenv, find_dotenv


_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']
config = configparser.ConfigParser()
config.read('config.ini')
OPENAI_LLM = config.get('SETTINGS', 'OPENAI_LLM')


class ExplorerAgent(BaseAgent):
    """
    This is LangChain's PandasDataFrameAgent which we borrow to conduct Exploratory Data Analysis (EDA) on the dataframe passed to it.
    It is called by the RouterAgent upon receiving a data exploration task, however there are no receives since no updates are done on data.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    
    # Constructor(s)
    def __init__(self, dataframe):
        self.dataframe = dataframe
        llm = ChatOpenAI(model=OPENAI_LLM, temperature=0)
        self.agent = create_pandas_dataframe_agent(df=dataframe, 
                                                   llm=llm, 
                                                   allow_dangerous_code=True, 
                                                   verbose=True, 
                                                   agent_type=AgentType.OPENAI_FUNCTIONS)

    # Call Override(s)
    def __call__(self, message):
        try:
            result = self.agent.invoke(message)
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
