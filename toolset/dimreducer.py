# *************** Dimensionality Reduction Tools ***************

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def principal_component_analysis(dataframe, components) -> dict:
    """
    Applies PCA or Principal Component Analysis to the dataframe and returns the dataframe after removing the features with the least variance.
    """
    pca = PCA(n_components=components, svd_solver='full') #svd_solver='covariance_eigh')
    dataframe = pca.fit_transform(dataframe)
    return dataframe


def linear_discriminant_analysis(dataframe, y_target, components) -> dict: # Work-In-Progress
    """
    Applies LDA or Linear Discriminant Analysis to the dataframe and returns the dataframe after reducing the dimensions by one and transforming them for maximum class separation.
    """
    lda = LinearDiscriminantAnalysis(n_components=components)

    return True