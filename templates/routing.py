# *************** Data Routing ***************
# Prompt template

prompt = """
Your job is to tag the given task or user command to the appropriate descriptions using tools.
If the task contains multiple actions that can be tagged to multiple descriptions, only consider the first action for tagging.
If the task satisfies multiple descriptions or arguments, choose the one that is most appropriate. 
If the task or command mentions about just selecting or dropping the columns names without any other modifications, choose feature selection over manipulation.
""".strip()