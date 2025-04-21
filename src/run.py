# *************** Main Workflow ***************

from agents.router_agent import RouterAgent


if '__main__'.__eq__(__name__):

    router = RouterAgent()
    user_input = input('User command:')
    router(user_input) # should find the appropriate agent and pass the command to it