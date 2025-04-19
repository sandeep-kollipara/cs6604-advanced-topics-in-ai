# *************** Loader Agent ***************

from agents.base_agent import BaseAgent
from agents.router_agent import RouterAgent
from toolset.loader import identify_and_load_file, save_data_to_file
from templates.scaling import prompt
from pydantic import BaseModel, Field
from typing import Optional
from langchain.tools import tool


class Loading(BaseModel):
    approx_filename: str = Field(..., description="A string containing filename and extension of the file to be loaded")
    memo: Optional[str] = Field(..., description="A string containing information on the features of the dataset within the file")

class Saving(BaseModel):
    filename: Optional[str] = Field(..., description="A string containing filename and extension of the file to be saved")


class LoaderAgent(BaseAgent):
    """
    This is the LoaderAgent which is responsible for identifying the filename and loading it to the dataframe in the black-box (bb) directory.
    It is also responsible for saving the latest dataframe (with changes) to a CSV file in the output directory.
    The user first interacts with this agent before handing off controls to RouterAgent (which may call this agent again).
    """

    # Field(s) (Class)

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=Loading)
    def __identify_and_load_file(approx_filename):
        """
        Identifies the file with given filename and returns the closest match for filename if not found. If found, it loads and returns the dataframe
        """
        LoaderAgent.dataframe = identify_and_load_file(approx_filename)
        return LoaderAgent.dataframe
    
    @staticmethod
    @tool(args_schema=Saving)
    def __save_data_to_file(filename='dataframe.csv'):
        """
        Saves the dataframe as a CSV file with the provided filename
        """
        save_data_to_file(LoaderAgent.dataframe)


    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'identify_and_load_file':self.__identify_and_load_file, 
                                                    'save_data_to_file':self.__save_data_to_file})
        self.dataframe = dataframe

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        callAgent = RouterAgent(self.dataframe)
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

