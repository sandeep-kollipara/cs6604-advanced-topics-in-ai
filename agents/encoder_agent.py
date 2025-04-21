# *************** Encoder Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.encoder import identify_categorical_features, translate_to_number, label_encoding, one_hot_encoding
from templates.encoding import prompt
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langchain.tools import tool


class Void(BaseModel):
    void: Optional[Any] = Field(..., description="No value is assigned to this argument")

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
    @tool(args_schema=Void)
    def __identify_categorical_features(void=''):
        """
        Identify categorical features in the dataframe and returns their column names and categories or unique classes within them
        """
        categorical_features = identify_categorical_features(dataframe=EncoderAgent.dataframe)
        return str(categorical_features), {categorical_features[i]:[getattr(x, 'tolist', lambda: x)()  # Limit to max 15 classes/categories
                                                                    for x in list(EncoderAgent.dataframe.loc[:, categorical_features[i]].unique())[:15]] 
                                           for i in range(len(categorical_features))} # Unique classes/categories are not sorted
    
    #@staticmethod
    #@tool(args_schema=Encoding)
    #def __translate_to_number(categorical_features):
    #    """
    #    Converts the categorical features containing string representations of numbers to numeric datatypes of numbers
    #    """
    #    translate_to_number(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features)
    
    @staticmethod
    @tool(args_schema=Encoding)
    def __label_encoding(categorical_features): # converts to numbers based on ordinality
        """
        Converts the categorical features of ordinal type with ordering or ranking or heirarchy within classes or categories for modelling
        """
        EncoderAgent.dataframe, EncoderAgent.y_encoder = label_encoding(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features, y_target=EncoderAgent.y_target)

    @staticmethod
    @tool(args_schema=Encoding)
    def __one_hot_encoding(categorical_features): # converts to multiple categories/classes in binary representation     
        """
        Converts the categorical features of nominal type with no ordering or ranking or heirarchy within classes or categories for modelling
        """
        EncoderAgent.dataframe, EncoderAgent.y_encoder = one_hot_encoding(dataframe=EncoderAgent.dataframe, categorical_features=categorical_features)
    
    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'identify_categorical_features':self.__identify_categorical_features, 
                                                    #'translate_to_number':self.__translate_to_number, 
                                                    'label_encoding':self.__label_encoding, 
                                                    'one_hot_encoding':self.__one_hot_encoding})
        EncoderAgent.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        self.dataframe = EncoderAgent.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

