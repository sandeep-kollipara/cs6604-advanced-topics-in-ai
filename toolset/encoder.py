# *************** Data Encoding Tools ***************

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from copy import deepcopy


def identify_categorical_features(dataframe) -> list[str]:
    featurelist = dataframe.columns
    numerical_features = []
    categorical_features = []
    integertypes = ['int16', 'int32', 'int64']
    floattypes = ['float16', 'float32', 'float64']
    for feature in featurelist:
        if dataframe[feature].dtype in integertypes:
            ints = sorted(list(dataframe[feature].unique()))
            ints_subtracted = {x - 1 for x in ints[1:]}
            if set(ints[:-1]) == ints_subtracted:
                categorical_features += [feature]
            else:
                numerical_features += [feature]
        elif dataframe[feature].dtype in floattypes:
            numerical_features += [feature]
        else:
            categorical_features += [feature]
    return categorical_features


def translate_to_number(dataframe, categorical_features) -> dict: # Using Label Encoding for the same indirectly
    y_encoder = object
    return dataframe, y_encoder


def label_encoding(dataframe, y_target, categorical_features) -> dict:
    labelEnc = LabelEncoder()
    y_encoder = None
    for feature in categorical_features:
        dataframe[feature] = labelEnc.fit_transform(dataframe[feature])
        if y_target == feature: y_encoder = labelEnc.deepcopy()
    return dataframe, y_encoder


def one_hot_encoding(dataframe, categorical_features) -> dict:
    oneHotEnc = OneHotEncoder(drop='first', 
                              feature_name_combiner=custom_combiner)
    arr_1HE = oneHotEnc.fit_transform(dataframe[categorical_features]).toarray()
    dataframe_1HE = pd.concat([dataframe.drop(categorical_features, axis=1), 
                               pd.DataFrame(arr_1HE, columns=oneHotEnc.get_feature_names_out(), index=dataframe.index)], 
                              axis=1)
    return dataframe_1HE, oneHotEnc


def custom_combiner(feature, category):
    return str(feature) + "_" + type(category).__name__ + "_" + str(category)
