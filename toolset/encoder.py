# *************** Data Encoding Tools ***************


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


def translate_to_number(dataframe, categorical_features) -> dict:
    y_encoder = object
    return y_encoder


def label_encoding(dataframe, categorical_features) -> dict:
    y_encoder = object
    return y_encoder


def one_hot_encoding(dataframe, categorical_features) -> dict:
    y_encoder = object
    return y_encoder