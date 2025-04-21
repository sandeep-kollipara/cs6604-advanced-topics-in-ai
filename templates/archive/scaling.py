# *************** Feature Scaling ***************
import os
import openai
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from typing import List

# Variable instantiation
df = pd.read_csv(r'./datasets/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv')
llm = ChatOpenAI(model='gpt-4o', temperature=0)
agent = create_pandas_dataframe_agent(df=df,
                                      llm=llm,
                                      allow_dangerous_code=True,
                                      verbose=True,
                                      agent_type=AgentType.OPENAI_FUNCTIONS)

# Tool definitions:
class scaler(BaseModel):
    scalable_features: List[str] = Field(..., description="Column names of the dataframe to be scaled")

@tool(args_schema=scaler)
def standardization(scalable_features: str) -> dict:
    """Check if .CSV file is present, load it into the dataframe and return it"""
    return f'Standardized the columns: {scalable_features}'

@tool(args_schema=scaler)
def normalization(scalable_features: str) -> dict:
    """Check if .CSV file is present, load it into the dataframe and return it"""
    return f'Normalized the columns: {scalable_features}'

tools = [standardization, normalization]

agent.invoke("Retrieve the categorical features in the dataframe")

# Prompt template

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

normalization:
e.g. normalization: 
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

standardization:
e.g. get_numerical_features: scalable_features
applies scaling to the numerical features

Example session:

Question: Standardize the columns feature1, feature2 and feature3 within the dataframe df
Thought: I should check the get
Action: get_numerical_features: Bulldog
PAUSE

You will be called again with this:

Observation: Standardized the columns: feature1, feature2 and feature3

You then output:

Answer: Successfully scaled the data. 
""".strip()

from agents.react_agent import *