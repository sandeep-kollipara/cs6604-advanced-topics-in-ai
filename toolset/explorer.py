# *************** Data Exploration Tools ***************

from pydantic import BaseModel, Field
from langchain.tools import tool



class Scaling(BaseModel):
    numerical_features: str = Field(..., description="List of column names of the numerical features in the dataframe")



def univariate_analysis(dataframe) -> dict:
    """
    Checks the dataframe columns and returns the features containing categorical data
    """
    return True


def visualization(dataframe) -> dict:
    """
    Checks the dataframe columns and returns the features containing categorical data
    """
    return True
