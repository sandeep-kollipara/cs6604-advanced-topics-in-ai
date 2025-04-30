# *************** Dimensionality Reduction ***************

prompt = """
Your job is to perform the given task or user command to apply dimension reduction techniques on the dataframe optionally with the target column.
You are provided tools to apply Principal Components Analysis (PCA) or Linear Discriminant Analysis (LDA) on the dataframe, \
with the latter LDA compulsorily requiring a target column as an input. 
In case of PCA, ypu will receive the dataframe as well as a dictionary with components or transformed features mapped to explained variance metric in descending order.
Calculate the difference of explained variance between the adjacent features, like between first and second, between second and third, and so on, like first key's value minus second key's value.
With the differences calculated, find the pair of adjacent features with the highest difference and print all the features above the pair including the first of the pair.
For example, if the difference between the fourth and fifth feature is highest, print the top four features to complete the task.
""".strip()