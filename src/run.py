# *************** Main Workflow ***************

from agents.react_agent import BasicAgent
from toolset.loader import identify_and_load_file


if '__main__'.__eq__(__name__):

    loaderAgent = BasicAgent(tool_dict={'identify_and_load_file': identify_and_load_file})
    usr_msg = input('Enter the filename')
    loaderAgent()