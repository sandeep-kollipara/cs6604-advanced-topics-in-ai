# *************** Feature Scaling ***************

# Prompt template

prompt = """
Your job is to perform the given task or user command related to scaling numerical features by standardization or normalization using tools.
First, identify numerical features within the data by running the function 'identify_numerical_features' which returns the list of column names of numerical features """ \
"""along with a dictionary with numerical feature as key paired with p-value of its normal test that checks whether the distribution of numerical feature is normal or not.
Next, split the numerical features into two lists: first list containing numerical features with p-value greater than 0.05, """ \
"""and the second list containing numerical features with p-value less than or equal to 0.05.
If a p-value of nan appears, skip that feature and suggest to check for null values for that numerical feature. 
Finally, use normalization function with the first list of numerical features of ratio types and standardization function with the second list of numerical features of interval types.
""".strip()