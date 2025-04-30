# *************** Feature Scaling ***************

# Prompt template

prompt = """
Your job is to perform the given task or user command related to scaling numerical features by standardization or normalization using tools.
First, identify numerical features within the data by running the function 'identify_numerical_features' which returns the list of column names of numerical features """ \
"""along with a dictionary with numerical feature as key paired with list of ten samples from each column as value.
Next, split the numerical features into two lists: first list containing ratio types with numerical features having only zero or positive values, """ \
"""and the second list containing interval types with numerical features including negative numbers.
Finally, use normalization function with the first list of numerical features of ratio types and standardization function with the second list of numerical features of interval types.
If unsure about the feature belonging to either ratio or interval, perform standardization.
""".strip()

