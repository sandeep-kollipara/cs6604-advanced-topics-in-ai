# *************** Dimensionality Reduction Tools ***************

import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import tool



class Scaling(BaseModel):
    numerical_features: str = Field(..., description="List of column names of the numerical features in the dataframe")



def principal_component_analysis(dataframe) -> dict:
    """
    Applies PCA or Principal Component Analysis to the dataframe and returns the dataframe after removing the features with the least variance.
    """
    return True



def linear_discriminant_analysis(dataframe) -> dict:
    """
    Applies LDA or Linear Discriminant Analysis to the dataframe and returns the dataframe after reducing the dimensions by one and transforming them for maximum class separation.
    """
    return True