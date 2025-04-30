# *************** Dimensionality Reduction ***************

prompt = """
Your job is to perform the given task or user command to apply dimension reduction techniques on the dataframe optionally with the target column.
You are provided tools to apply Principal Components Analysis (PCA) or Linear Discriminant Analysis (LDA) on the dataframe, \
with the latter LDA compulsorily requiring a target column as an input. 
In case of PCA, use the tool to receive a dictionary with features mapped to a cumulative metric.
Apply a cutoff of 95 and select all the features before and below 95 cutoff including the first feature crossing the 95.
For example, if the the fourth, fifth and sixth feature have 93, 95.5 and 97, select the top five features with cumulative metric less than or equal to 95.5 to complete the task.
""".strip()