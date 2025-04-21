# *************** Data Encoding ***************

# Prompt template

prompt = """
Your job is to perform the given task or user command related to encoding categorical features by label encoding or one-hot encoding using tools.
Start by identifying numerical features within the data by running the function 'identify_categorical_features' which returns the list of column names of numerical features """ \
"""along with a dictionary for each numerical feature with all unique values or categories or classes within them.
Next, split the categorical features into two lists: first list containing ordinal types with categorical features having order or rank or heirarchy between themselves, """ \
"""and the second list containing nominal types with categorical features that are not having order or rank or heirarchy between themselves.
Finally, use 'label_encoding function with the first list of categorical features of ordinal types and 'one_hot_encoding' with the second list of categorical features of nominal types.""".strip()

