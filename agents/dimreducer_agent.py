# *************** Dimensionality Reduction Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.dimreducer import principal_component_analysis, linear_discriminant_analysis
from templates.scaling import prompt
from pydantic import BaseModel, Field
from langchain.tools import tool


class DimReducer(BaseModel):
    components: int = Field(..., description="A positive integer indicating the top relevant components from all the features in the dataframe")


class DimReducerAgent(BaseAgent):
    """
    This is the DimReducer which conducts Dimensionality Reduction (PCA or LDA) on the features of a dataframe.
    It is called by the RouterAgent upon receiving a data scaling task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=DimReducer)
    def __principal_component_analysis(components):
        """
        Applies Principal Component Analysis (PCA) to retrieve the top components of the features with variance in the dataframe
        """
        DimReducer.dataframe = principal_component_analysis(dataframe=DimReducer.dataframe, components=components)
    
    @staticmethod
    @tool(args_schema=DimReducer)
    def __linear_discriminant_analysis(components):
        """
        Applies Linear Discriminant Analysis (LDA) to retrieve the top components of the features with separation in the dataframe
        """
        DimReducer.dataframe = linear_discriminant_analysis(dataframe=DimReducer.dataframe, components=components)

    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'principal_component_analysis':self.__principal_component_analysis, 
                                                    'linear_discriminant_analysis':self.__linear_discriminant_analysis})
        DimReducer.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        self.dataframe = DimReducer.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

