# *************** Router Agent ***************

from agents.base_agent import BaseAgent
import agents.dimreducer_agent
import agents.featselector_agent
import agents.loader_agent
import agents.scaler_agent
import agents.encoder_agent
import agents.explorer_agent
import agents.cleaner_agent
import agents.dimreducer_agent
import agents.featselector_agent
from templates.routing import prompt
import pandas as pd
from pydantic import BaseModel, Field
from langchain.tools import tool


class Routing(BaseModel): # Tagging
    end: bool = Field(..., description="A boolean indicating whether the task mentions something like quit or exit or close the program")
    fast_forward: bool = Field(..., description="A boolean indicating whether the task mentions something like fast forward or redo or go next")
    rewind: bool = Field(..., description="A boolean indicating whether the task mentions something like rewind or undo or go back")
    load_save: bool = Field(..., description="A boolean indicating whether the task is related to loading data from a file or saving data to a file")
    exploration: bool = Field(..., description="A boolean indicating whether the task is related to data visualization or exploratory data analysis")
    cleaning: bool = Field(..., description="A boolean indicating whether the task is related to data cleaning like " \
                           "null value treatment, outlier treatment, variance filter, constant and identical feature elimination") # CAN BE IMPROVED
    scaling: bool = Field(..., description="A boolean indicating whether the task is related to standardization or normalization of numerical features")
    encoding: bool = Field(..., description="A boolean indicating whether the task is related to encoding of categorical features like " \
                           "string to number translation, label encoding or one-hot encoding")
    dimension_reduction: bool = Field(..., description="A boolean indicating whether the task is related to dimensionality reduction techniques like " \
                                      "principal components analysis (PCA) or linear discriminant analysis (LDA)")
    feature_selection: bool = Field(..., description="A boolean indicating whether the task is related to feature selection techniques like " \
                                    "Recursive Feature Elimination (RFE) or Random Forest Analysis or Boruta")
    manipulation: bool = Field(..., description="A boolean indicating whether the task is related to manipulating or transforming the dataframe like " \
                               "pivot, group by, aggregate, rename, drop columns or rows.")


class RouterAgent(BaseAgent):
    """
    This is the RouterAgent which is the center piece of the framework that interacts with the user.
    It handles assignment of tasks to all other agents as well as exiting the application.
    It calls the requisite agent per user command and passes the contents to it for processing.
    It ('s new instance) receives the dataframe and relevant objects from all other agents after their task is done.
    It can also rewind (undo) and fast-forward (redo) actions by accessing its past/future instances.
    """

    # Field(s) (Class)
    tag = None # Placeholder for assigned task description
    latest = None # Placeholder for the latest RouterAgent instance

    # Public Method(s)

    # Private Method(s)
    @staticmethod
    @tool(args_schema=Routing)
    def __routing(end = False, fast_forward=False, rewind=False, load_save=False, exploration=False, cleaning=False, scaling=False, 
                  encoding=False, dimension_reduction=False, feature_selection=False, manipulation=False):
        """
        Tags the task as True for just one of the arguments that describe how the task is related and as False for the remaining
        """
        route_dict = {end:'end',
                      fast_forward:'fast_forward',
                      rewind:'rewind',
                      load_save:'load_save',
                      exploration:'exploration',
                      cleaning:'cleaning',
                      scaling:'scaling',
                      encoding:'encoding',
                      dimension_reduction:'dimension_reduction',
                      feature_selection:'feature_selection',
                      manipulation:'manipulation'}
        try:
            RouterAgent.tag = route_dict[1]
        except KeyError:
            return 'stall' # This is the exception value for tag when none of the tools are selected for the user command.
        return route_dict[1] # This can be removed in production, waste of API call
    
    # Constructor(s)
    def __init__(self, dataframe):
        super().__init__(starter=prompt, tool_dict={'routing':self.__routing})
        if type(dataframe) is not pd.DataFrame or type(dataframe) is str: # Exception handling in case of failed tool call
            pass #self = RouterAgent.latest # Does not work like that
        else:
            self.dataframe = dataframe
        self.before = RouterAgent.latest
        RouterAgent.latest = self
        self.after = None

    # Call Override(s)
    def __call__(self, message):
        super().__call__(message)
        if self.tag == 'end':
            return None # Exit in the __main__
        elif self.tag == 'fast_forward':
            if self.after == None: return self
            else: return self.after
        elif self.tag == 'rewind':
            if self.before == None: return self
            else: return self.before
        elif self.tag == 'load_save':
            callAgent = agents.loader_agent.LoaderAgent(self.dataframe)
        elif self.tag == 'exploration':
            callAgent = agents.explorer_agent.ExplorerAgent(self.dataframe)
        elif self.tag == 'cleaning':
            callAgent = agents.cleaner_agent.CleanerAgent(self.dataframe)
        elif self.tag == 'scaling':
            callAgent = agents.scaler_agent.ScalerAgent(self.dataframe)
        elif self.tag == 'encoding':
            callAgent = agents.encoder_agent.EncoderAgent(self.dataframe)
        elif self.tag == 'dimension_reduction':
            callAgent = agents.dimreducer_agent.DimReducerAgent(self.dataframe)
        elif self.tag == 'feature_selection':
            callAgent = agents.featselector_agent.FeatSelectorAgent(self.dataframe)
        #elif self.tag == 'manipulation':
        #    callAgent = EvalAgent()
        elif self.tag == 'stall': # Exception case handling: Do nothing!
            callAgent = self
        else: return 'Error: RouterAgent malfunctioned'
        self.after = callAgent
        return callAgent

    # String Override(s)
    def __str__(self):
        print("Text temporary.")

