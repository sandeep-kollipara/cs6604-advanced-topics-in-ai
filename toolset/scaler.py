# *************** Data Standardization Tools ***************

import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer


def identify_numerical_features(dataframe) -> list[str]:
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
    return numerical_features


def standardization(dataframe, numerical_features) -> dict:
    scaler = StandardScaler()
    scaler.fit(dataframe.loc[:, numerical_features])
    array_scaled = scaler.transform(dataframe.loc[:,numerical_features])
    dataframe_total = pd.concat([pd.DataFrame(array_scaled,
                                              columns=numerical_features,
                                              index=dataframe.index),
                                 dataframe.loc[:, [col for col in dataframe.columns if col not in numerical_features]]], 
                                 axis=1)
    return dataframe_total, scaler


def normalization(dataframe, numerical_features) -> dict:
    scaler = Normalizer()
    scaler.fit(dataframe.loc[:, numerical_features])
    array_scaled = scaler.transform(dataframe.loc[:,numerical_features])
    dataframe_total = pd.concat([pd.DataFrame(array_scaled,
                                              columns=numerical_features,
                                              index=dataframe.index),
                                 dataframe.loc[:, [col for col in dataframe.columns if col not in numerical_features]]], 
                                 axis=1)
    return dataframe_total, scaler
