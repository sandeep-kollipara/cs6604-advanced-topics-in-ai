# *************** Dimensionality Reduction Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.dimreducer import principal_component_analysis, linear_discriminant_analysis
from templates.dimreduction import prompt
from pydantic import BaseModel, Field
from langchain.tools import tool


class DimReducer1(BaseModel):
    components: int = Field(..., description="A positive integer indicating the top relevant components from all the features in the dataframe")

class DimReducer2(BaseModel):
    components: int = Field(..., description="A positive integer indicating the top relevant components from all the features in the dataframe")
    y_target: str = Field(..., description="A string indicating the target column for dimensionality reduction in the dataframe")


class DimReducerAgent(BaseAgent):
    """
    This is the DimReducer which conducts Dimensionality Reduction (PCA or LDA) on the features of a dataframe.
    It is called by the RouterAgent upon receiving a dimensionality reduction task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=DimReducer1)
    def __principal_component_analysis(components):
        """
        Applies Principal Component Analysis (PCA) to retrieve the top components of the features with variance in the dataframe
        """
        try:
            DimReducerAgent.dataframe, explaied_variance_dict = principal_component_analysis(dataframe=DimReducerAgent.dataframe, components=components)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return DimReducerAgent.dataframe, explaied_variance_dict
    
    @staticmethod
    @tool(args_schema=DimReducer2)
    def __linear_discriminant_analysis(components, y_target):
        """
        Applies Linear Discriminant Analysis (LDA) to retrieve the top components of the features with separation in the dataframe
        """
        try:
            DimReducerAgent.dataframe = linear_discriminant_analysis(dataframe=DimReducerAgent.dataframe, components=components, y_target=y_target)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return DimReducerAgent.dataframe

    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'principal_component_analysis':self.__principal_component_analysis, 
                                                    'linear_discriminant_analysis':self.__linear_discriminant_analysis})
        DimReducerAgent.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        message += '\nThe columns currently in the dataframe are: ' + str(list(self.dataframe.columns))
        super().__call__(message)
        self.dataframe = DimReducerAgent.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

