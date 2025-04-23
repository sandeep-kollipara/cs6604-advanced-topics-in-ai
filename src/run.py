# *************** Main Workflow ***************

from agents.loader_agent import LoaderAgent
import agents.router_agent


if '__main__'.__eq__(__name__):

    user_input = input('\nTalk to the AI agent: ')
    nextAgent = LoaderAgent(None)
    prevAgent = nextAgent(user_input) # RouterAgent
    while nextAgent is not None:
        user_input = input('\nContinue talking to AI agent: ')
        nextAgent = prevAgent(user_input) # WorkerAgent unless user fast-forwards or rewinds
        if type(nextAgent) == agents.router_agent.RouterAgent or nextAgent is None:
            prevAgent = nextAgent
            continue
        else:
            prevAgent = nextAgent(user_input)
    print('\n*** Program End ***')