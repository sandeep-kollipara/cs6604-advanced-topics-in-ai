# *************** Scaler Agent ***************

from agents.base_agent import BaseAgent
from agents.router_agent import RouterAgent
from toolset.scaler import identify_numerical_features, standardization, normalization
from templates.scaling import prompt
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool


class Scaling(BaseModel):
    numerical_features: List[str] = Field(..., description="List of column names of the numerical features in the dataframe")


class ScalerAgent(BaseAgent):
    """
    This is the ScalerAgent which conducts standardization and normalization on numerical features of a dataframe.
    It is called by the RouterAgent upon receiving a data scaling task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=Scaling)
    def __identify_numerical_features():
        """
        Identify numerical features in the dataframe and returns their column names and few random data samples within them
        """
        numerical_features = identify_numerical_features(dataframe=ScalerAgent.dataframe)
        return numerical_features, ScalerAgent.dataframe.loc[:, numerical_features].head(10)
    
    @staticmethod
    @tool(args_schema=Scaling)
    def __standardization(numerical_features):
        """
        Applies standardization or scaling to the numerical features of the dataframe
        """
        ScalerAgent.dataframe, ScalerAgent.y_encoder = standardization(dataframe=ScalerAgent.dataframe, numerical_features=numerical_features)
    
    @staticmethod
    @tool(args_schema=Scaling)
    def __normalization(numerical_features):
        """
        Applies normalization to the numerical features of the dataframe
        """
        ScalerAgent.dataframe, ScalerAgent.y_encoder = normalization(dataframe=ScalerAgent.dataframe, numerical_features=numerical_features)

    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'identify_numerical_features':self.__identify_numerical_features, 
                                                    'standardization':self.__standardization, 
                                                    'normalization':self.__normalization})
        self.dataframe = dataframe

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        callAgent = RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

