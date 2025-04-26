# *************** Dimensionality Reduction Tools ***************

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def principal_component_analysis(dataframe, components) -> dict:
    """
    Applies PCA or Principal Component Analysis to the dataframe and returns the dataframe after removing the features with the least variance.
    """
    pca = PCA(n_components=components, svd_solver='full') #svd_solver='covariance_eigh')
    X_columns = list(dataframe.columns)
    X_transformed = pca.fit_transform(dataframe)
    dataframe_transformed = pd.DataFrame(data=X_transformed, 
                                         columns=['feature'+str(i+1) for i in range(X_transformed.shape[1])], 
                                         index = dataframe.index)
    pca_expVarCumSum =  np.hstack([np.reshape(pca.get_feature_names_out(), (-1,1)), 
                                   np.reshape(pca.explained_variance_ratio_, (-1,1))])#.cumsum(), (-1,1))])
    df_pca_expVarCumSum = pd.DataFrame(data=pca_expVarCumSum, columns=['PrincipalCom', 'DepVarCumSum'])
    return dataframe_transformed, {df_pca_expVarCumSum['PrincipalCom'].iloc[i] : 100*float(df_pca_expVarCumSum['DepVarCumSum'].iloc[i]) 
                                   for i in range(len(df_pca_expVarCumSum))}


def linear_discriminant_analysis(dataframe, y_target, components) -> dict: # Work-In-Progress
    """
    Applies LDA or Linear Discriminant Analysis to the dataframe for the selected target column and returns the dataframe after reducing the dimensions by one and transforming them for maximum class separation.
    """
    lda = LinearDiscriminantAnalysis(n_components=components)
    if y_target == None or y_target not in dataframe.columns: return 'Invalid input for y_target, cannot peform LDA'
    X_columns = list(dataframe.columns)
    X_columns.remove(y_target)
    X, y = dataframe[X_columns], dataframe[[y_target]]
    X_transformed = lda.fit_transform(X, y)
    dataframe_transformed = pd.concat([pd.DataFrame(data=X_transformed, 
                                                    columns=['feature'+str(i+1) for i in range(X_transformed.shape[1])], 
                                                    index = dataframe.index), 
                                       y], axis=1)
    return dataframe_transformed
