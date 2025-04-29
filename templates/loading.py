# *************** Data Loading ***************

# Prompt template

prompt = """
Your job is to perform the given task or user command to load the data from file or save the data to file.
You are provided tools to identify and load file with given filename to dataframe, or save data to file with optionally given filename from dataframe. 
Also, if the user provides any new information about the data while loading a file, pass the information as a string in the optional argument memo. 
""".strip()