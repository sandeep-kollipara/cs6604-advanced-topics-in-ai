# *************** Base Agent ***************

import pandas as pd
from agents.react_agent import BasicAgent
#from dataclasses import dataclass
#from pydantic import BaseModel, Field
#from typing import List, Any


#class DataFrameModel(BaseModel):
#   chat_history: List[str]
#   dataframe: pd.DataFrame
#   train: pd.DataFrame
#   test: pd.DataFrame
#   validation: pd.DataFrame
#   y_target: str
#   y_scaler: Any
#   y_encoder: Any
#   
#   class Config:
#        arbitrary_types_allowed = True

#@dataclass
class BaseAgent(BasicAgent):
    """
    This is the Base agent which interacts is just a template for storing class variables/fields.
    It isn't used directly in the workflow, only as a superclass with shared variables/fields.
    """

    # Field(s) (Class)
    chat_history = []
    dataframe = pd.DataFrame()
    train = pd.DataFrame()
    test = pd.DataFrame()
    validation = pd.DataFrame()
    y_target = ''
    y_scaler = object
    y_encoder = object

    # Public Method(s)

    # Private Method(s)

    # Constructor(s)
    def __init__(self, starter="You are a data analyst with tools to preprocess and manage dataframes", tool_dict={}):
        super().__init__(starter=starter, tool_dict=tool_dict) # Passthrough

    # Call Override(s)
    def __call__(self, message):
        result = self.model2.invoke({"input": f"{message}"})
        self.chat_history += result

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

