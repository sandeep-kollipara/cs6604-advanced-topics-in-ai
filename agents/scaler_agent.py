# *************** Scaler Agent ***************

from agents.base_agent import BaseAgent
import agents.router_agent
from toolset.scaler import identify_numerical_features, standardization, normalization
from templates.scaling import prompt
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langchain.tools import tool
from scipy.stats import normaltest
import random as rand

rand.seed(6604)


class Void(BaseModel):
    void: Optional[Any] = Field(..., description="No value is assigned to this argument")

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
    @tool(args_schema=Void)
    def __identify_numerical_features(void=''):
        """
        Identify numerical features in the dataframe and returns their column names and few random data samples within them
        """
        try:
            numerical_features = identify_numerical_features(dataframe=ScalerAgent.dataframe)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        random_sample_indices = list(ScalerAgent.dataframe.index[[rand.randint(0, len(ScalerAgent.dataframe)) for _ in range(5)]])
        return str(numerical_features), {numerical_features[i]: round(float(normaltest(ScalerAgent.dataframe[numerical_features[i]].to_numpy()).pvalue), 3) \
                                         for i in range(len(numerical_features))}
                                         #{numerical_features[i] : (list(ScalerAgent.dataframe[numerical_features[i]].sort_values(ascending=True))[:5]#[:10])
                                         #                         + list(ScalerAgent.dataframe[numerical_features[i]][random_sample_indices]))
                                         #for i in range(len(numerical_features))}
        #ScalerAgent.dataframe.loc[:, numerical_features].sort_values(by=numerical_features).head(10).to_string()
    
    @staticmethod
    @tool(args_schema=Scaling)
    def __standardization(numerical_features):
        """
        Applies standardization or scaling to the numerical features of the dataframe
        """
        try:
            ScalerAgent.dataframe, ScalerAgent.y_encoder = standardization(dataframe=ScalerAgent.dataframe, numerical_features=numerical_features)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return ScalerAgent.dataframe
    
    @staticmethod
    @tool(args_schema=Scaling)
    def __normalization(numerical_features):
        """
        Applies normalization to the numerical features of the dataframe
        """
        try:
            ScalerAgent.dataframe, ScalerAgent.y_encoder = normalization(dataframe=ScalerAgent.dataframe, numerical_features=numerical_features)
        except Exception as exc:
            return exc
        except:
            return 'Base Exception Error...'
        return ScalerAgent.dataframe

    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'identify_numerical_features':self.__identify_numerical_features, 
                                                    'standardization':self.__standardization, 
                                                    'normalization':self.__normalization})
        ScalerAgent.dataframe = self.dataframe = dataframe # Apparently both do not reference the same variable

    # Call Override(s)
    def __call__(self, message):
        message += '\nThe columns currently in the dataframe are: ' + str(list(self.dataframe.columns))
        super().__call__(message)
        self.dataframe = ScalerAgent.dataframe  # Apparently both do not reference the same variable
        callAgent = agents.router_agent.RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

