# *************** Feature Selection Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.featselector import random_forest_analysis, backward_stepwise_linear_regression, drop_features_from_dataframe, select_features_in_dataframe 
from templates.featselection import prompt
from pydantic import BaseModel, Field
from typing import List
from langchain.tools import tool


class SelectedFeatures(BaseModel):
    selected_features: List[str] = Field(..., description="List of column names of the features to be selected or kept in the dataframe without the remaining ")

class DroppedFeatures(BaseModel):
    dropped_features: List[str] = Field(..., description="List of column names of the features to be dropped or eliminated from the dataframe")

class FeatSelection(BaseModel):
    y_target: str = Field(..., description="A string indicating the target column for feature selection in the dataframe")

#class FeatSelection2(BaseModel):
#    y_target: str = Field(..., description="A string indicating the target column for dimensionality reduction in the dataframe")
#    test_size: Optional[float] = Field(..., description="A decimal indicating the ratio of the original data used as test set for fitting the linear model")
#    train_size: Optional[float] = Field(..., description="A decimal indicating the ratio of the original data used as train set for fitting the linear model")


class FeatSelectorAgent(BaseAgent):
    """
    This is the FeatSelectorAgent which conducts Feature Selection on the dataframe.
    It is called by the RouterAgent upon receiving a feature selection task.
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=SelectedFeatures)
    def __select_features_in_dataframe(selected_features):
        """
        Applies Principal Component Analysis (PCA) to retrieve the top components of the features with variance in the dataframe
        """
        try:
            FeatSelectorAgent.dataframe = select_features_in_dataframe(dataframe=FeatSelectorAgent.dataframe, selected_features=selected_features)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return FeatSelectorAgent.dataframe
    
    @staticmethod
    @tool(args_schema=DroppedFeatures)
    def __drop_features_from_dataframe(dropped_features):
        """
        Applies Principal Component Analysis (PCA) to retrieve the top components of the features with variance in the dataframe
        """
        try:
            FeatSelectorAgent.dataframe = drop_features_from_dataframe(dataframe=FeatSelectorAgent.dataframe, dropped_features=dropped_features)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return FeatSelectorAgent.dataframe
    
    @staticmethod
    @tool(args_schema=FeatSelection)
    def __random_forest_analysis(y_target):
        """
        Conducts Random Forest Analysis on the dataframe with respect to a target column which is also a numerical feature. \
        It returns a dictionary with features as keys and their importance as values.
        """
        try:
            return random_forest_analysis(dataframe=FeatSelectorAgent.dataframe, y_target=y_target)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
    
    @staticmethod
    @tool(args_schema=FeatSelection)
    def __backward_stepwise_linear_regression(y_target):
        """
        Conducts Backward Stepwise Linear Regression on the dataframe with respect to a target column which is also a numerical feature. \
        It returns a table with the dropped features ranked by the higher pValue of T-test first alongside Adjusted R-squared of the linear model.
        """
        try:
            return backward_stepwise_linear_regression(dataframe=FeatSelectorAgent.dataframe, y_target=y_target)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'

    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'select_features_in_dataframe':self.__select_features_in_dataframe, 
                                                    'drop_features_from_dataframe':self.__drop_features_from_dataframe, 
                                                    'random_forest_analysis':self.__random_forest_analysis, 
                                                    'backward_stepwise_linear_regression':self.__backward_stepwise_linear_regression})
        FeatSelectorAgent.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        self.dataframe = FeatSelectorAgent.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

