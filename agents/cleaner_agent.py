# *************** Cleaner Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.cleaner import EliminateConstantFeatures, EliminateFeaturesWithHighNulls, EliminateIdenticalFeatures, \
    EliminateFeaturesWithHighZeroes, EliminateFeaturesWithNearZeroVariance, EliminateFeaturesWithHighNonnumericUniqueValues, \
    EliminateCorrelatedFeatures, MissingValueTreatment, OutlierTreatment
from templates.cleaning import prompt
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langchain.tools import tool


class Void(BaseModel):
    void: Optional[Any] = Field(..., description="No value is assigned to this argument")

class Cleaning1(BaseModel):
    percentage: float = Field(..., description="A numerical value with decimals indicating the percentage set between 0 and 100 including both")

class Cleaning2(BaseModel):
    threshold: float = Field(..., description="A numerical value with decimals indicating the threshold or limit set between 0 and 1 including both")

class Cleaning3(BaseModel):
    percentile: float = Field(..., description="A numerical value with decimals indicating the percentile set between 0 and 100 including both")

class MVT(BaseModel):
    features: List[str] = Field(..., description="List of column names of the selected features in the dataframe")


class CleanerAgent(BaseAgent):
    """
    This is the CleanerAgent conducts a range of data cleaning techniques on the features of a dataframe.
    It is called by the RouterAgent upon receiving a data cleaning task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=Void)
    def __EliminateConstantFeatures(void=''):
        """
        Drops the constant features that have the same value for all the rows of the dataframe
        """
        CleanerAgent.dataframe = EliminateConstantFeatures(dataframe=CleanerAgent.dataframe)
    
    @staticmethod
    @tool(args_schema=Cleaning1)
    def __EliminateFeaturesWithHighNulls(percentage):
        """
        Drops the features that have null values above a set percentage of all the rows of the dataframe, usually the percentage is set at 80
        """
        CleanerAgent.dataframe = EliminateFeaturesWithHighNulls(dataframe=CleanerAgent.dataframe, percentage=percentage)
    
    @staticmethod
    @tool(args_schema=Void)
    def __EliminateIdenticalFeatures(void=''):
        """
        Drops the identical features that are duplicate of another feature within the dataframe
        """
        CleanerAgent.dataframe = EliminateIdenticalFeatures(dataframe=CleanerAgent.dataframe)
    
    @staticmethod
    @tool(args_schema=Cleaning1)
    def __EliminateFeaturesWithHighZeroes(percentage):
        """
        Drops the features that have zero values above a set percentage of all the rows of the dataframe, usually the percentage is set at 80
        """
        CleanerAgent.dataframe = EliminateFeaturesWithHighZeroes(dataframe=CleanerAgent.dataframe, percentage=percentage)
    
    @staticmethod
    @tool(args_schema=Void)
    def __EliminateFeaturesWithNearZeroVariance(void=''):
        """
        Drops the features with close to zero variance across all the rows of the dataframe
        """
        CleanerAgent.dataframe = EliminateFeaturesWithNearZeroVariance(dataframe=CleanerAgent.dataframe)
    
    @staticmethod
    @tool(args_schema=Cleaning1)
    def __EliminateFeaturesWithHighNonnumericUniqueValues(percentage):
        """
        Drops the features that have high repetition of unique categorical or string values above a set percentage of all the rows of the dataframe, usually the percentage is set at 80
        """
        CleanerAgent.dataframe = EliminateFeaturesWithHighNonnumericUniqueValues(dataframe=CleanerAgent.dataframe, percentage=percentage)
    
    @staticmethod
    @tool(args_schema=Cleaning2)
    def __EliminateCorrelatedFeatures(threshold):
        """
        Drops the features that highly correlated with another feature within the dataframe above a set threshold between 0 and 1 including both, usually the threshold is set at 0.8
        """
        CleanerAgent.dataframe = EliminateCorrelatedFeatures(dataframe=CleanerAgent.dataframe, threshold=threshold)
    
    @staticmethod
    @tool(args_schema=MVT)
    def __MissingValueTreatment(features=[]):
        """
        Applies Missing Value Treatment (MVT) to the selected features of the dataframe if provided and applies to all features if no argument is provided
        """
        CleanerAgent.dataframe = MissingValueTreatment(dataframe=CleanerAgent.dataframe, features=features)
    
    @staticmethod
    @tool(args_schema=Cleaning3)
    def __OutlierTreatment(percentile):
        """
        Applies Outlier Treatment to numerical features with significant deviation for the exteme percentiles among observations of the dataframe, usually the percentile is set at 1
        """
        CleanerAgent.dataframe = OutlierTreatment(dataframe=CleanerAgent.dataframe, percentile=percentile) # arg is 'percentile' not 'percentage'
    
    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'EliminateConstantFeatures':self.__EliminateConstantFeatures, 
                                                    'EliminateFeaturesWithHighNulls':self.__EliminateFeaturesWithHighNulls, 
                                                    'EliminateIdenticalFeatures':self.__EliminateIdenticalFeatures, 
                                                    'EliminateFeaturesWithHighZeroes':self.__EliminateFeaturesWithHighZeroes, 
                                                    'EliminateFeaturesWithNearZeroVariance':self.__EliminateFeaturesWithNearZeroVariance, 
                                                    'EliminateFeaturesWithHighNonnumericUniqueValues':self.__EliminateFeaturesWithHighNonnumericUniqueValues, 
                                                    'EliminateCorrelatedFeatures':self.__EliminateCorrelatedFeatures, 
                                                    'MissingValueTreatment':self.__MissingValueTreatment, 
                                                    'OutlierTreatment':self.__OutlierTreatment})
        CleanerAgent.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        self.dataframe = CleanerAgent.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

