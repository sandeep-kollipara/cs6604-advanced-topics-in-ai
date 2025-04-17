# *************** Feature Scaling ***************

# Prompt template

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:

normalization:
e.g. normalization: 
Runs a calculation and returns the number - uses Python so be sure to use floating point syntax if necessary

standardization:
e.g. get_numerical_features: scalable_features
applies scaling to the numerical features

Example session:

Question: Standardize the columns feature1, feature2 and feature3 within the dataframe df
Thought: I should check the get
Action: get_numerical_features: Bulldog
PAUSE

You will be called again with this:

Observation: Standardized the columns: feature1, feature2 and feature3

You then output:

Answer: Successfully scaled the data. 
""".strip()