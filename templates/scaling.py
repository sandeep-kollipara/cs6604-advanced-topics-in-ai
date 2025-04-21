# *************** Feature Scaling ***************

# Prompt template

prompt = """
Your job is to perform the given task or user command related to scaling numerical features by standardization or normalization using tools.
First, identify numerical features within the data by running the function 'identify_numerical_features' which returns the list of column names of numerical features along with sample data within each column.
Second, use 'standardization' function to numerical features returned from the first function to scale the data.
""".strip()