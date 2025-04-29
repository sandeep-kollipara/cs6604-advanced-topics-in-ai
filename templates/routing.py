# *************** Data Routing ***************
# Prompt template

prompt = """
Your job is to tag the given task or user command to the appropriate descriptions using tools.
If the task contains multiple actions that can be tagged to multiple descriptions, only consider the first action for tagging.
If the task satisfies multiple descriptions or arguments, choose the one that is most appropriate.
If the task or command is a query that asks information regarding the data or dataframe such as columns or data types, choose exploration. 
If the task or command mentions about just selecting or dropping the columns names without any other modifications, choose feature selection over manipulation.
If none of the tags are appropriate to the given task or user command, choose the stall option.
""".strip()