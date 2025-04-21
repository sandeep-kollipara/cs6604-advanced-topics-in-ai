import os
import openai
import pandas as pd
from langchain_experimental.agents import create_pandas_dataframe_agent, create_csv_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']


df = pd.read_csv(r'./datasets/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
llm = ChatOpenAI(model='gpt-4o', temperature=0)
agent = create_pandas_dataframe_agent(df=df,
                                      llm=llm,
                                      allow_dangerous_code=True,
                                      verbose=True,
                                      agent_type=AgentType.OPENAI_FUNCTIONS)

agent.invoke("What are columns in the dataframe?")

agent.invoke("drop the column SizeRank and return the df.")


agent2 = create_csv_agent(llm=llm,
                          path=r'./datasets/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv',
                          allow_dangerous_code=True,
                          #verbose=True,
                          agent_type=AgentType.OPENAI_FUNCTIONS)

agent2.invoke("What are columns in the data?")
