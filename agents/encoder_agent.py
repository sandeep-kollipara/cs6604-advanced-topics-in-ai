# *************** Encoder Agent ***************

from agents.base_agent import BaseAgent
from toolset.encoder import identify_categorical_features, translate_to_number, label_encoding, one_hot_encoding
from templates.scaling import prompt
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool


class Encoding(BaseModel):
    categorical_features: List[str] = Field(..., description="List of column names of the categorical features in the dataframe")


class EncoderAgent(BaseAgent):
    """
    This is the EncoderAgent which conducts translation into numbers, label encoding or one-hot encoding on numerical features of a dataframe.
    It is called by the RouterAgent upon receiving a data encoding task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=Encoding)
    def __identify_categorical_features():
        """
        Identify categorical features in the dataframe and returns their column names and categories or unique classes within them
        """
        identify_categorical_features(dataframe=EncoderAgent.dataframe)
    
    @staticmethod
    @tool(args_schema=Encoding)
    def __translate_to_number(categorical_features):
        """
        Converts the categorical features containing string representations of numbers to numeric datatypes of numbers
        """
        translate_to_number(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features)
    
    @staticmethod
    @tool(args_schema=Encoding)
    def __label_encoding(categorical_features):
        """
        Converts the categorical features of ordinal type to numbers based on their ordering or ranking
        """
        label_encoding(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features)

    @staticmethod
    @tool(args_schema=Encoding)
    def __one_hot_encoding(categorical_features):
        """
        Converts the categorical features of nominal type to multiple categories or classes in binary representation
        """
        one_hot_encoding(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features)
    
    # Constructor(s)
    def __init__(self):
        super().__init__(starter=prompt, tool_dict={'identify_categorical_features':self.__identify_categorical_features, 
                                                    'translate_to_number':self.__translate_to_number, 
                                                    'label_encoding':self.__label_encoding, 
                                                    'one_hot_encoding':self.__one_hot_encoding})

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

