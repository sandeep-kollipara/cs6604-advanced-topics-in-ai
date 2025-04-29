# *************** Feature Selection Tools ***************

from agents.react_agent import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
#from prettytable import PrettyTable
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import train_test_split


def select_features_in_dataframe(dataframe, selected_features):
    return dataframe[selected_features]


def drop_features_from_dataframe(dataframe, dropped_features):
    dataframe.drop(dropped_features, axis=1, inplace=True)
    return dataframe


def random_forest_analysis(dataframe, y_target):
    remaining_columns = list(dataframe.columns)
    remaining_columns.remove(y_target)
    X = dataframe[remaining_columns]
    y = dataframe[y_target]
    forest = RandomForestRegressor(random_state=6604)
    forest.fit(X, y)
    importances = forest.feature_importances_
    df_featImportance = pd.DataFrame(index=X.columns, data=np.reshape(importances, (-1,1)), columns=['FeatImport'])\
        .reset_index(drop=False)\
        .rename({'index':'FeatName'}, axis=1)\
        .sort_values(by='FeatImport', ascending=False)
    df_featImportance['FeatImportPerc'] = (df_featImportance['FeatImport']/df_featImportance['FeatImport'].sum()*100).apply(round)
    sns.barplot(df_featImportance, x="FeatImport", y="FeatName", color='g')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.grid(axis='both')
    plt.title(label='Feature vs. Importance')
    plt.tight_layout()
    plt.show()
    return {df_featImportance['FeatName'].iloc[i]:float(df_featImportance['FeatImportPerc'].iloc[i]) 
            for i in range(len(df_featImportance))}


def backward_stepwise_linear_regression(dataframe, y_target):#, test_size=0.2, train_size=0.8):
    remaining_columns = list(dataframe.columns)
    remaining_columns.remove(y_target)
    X = dataframe[remaining_columns]
    y = dataframe[y_target]
    #X_train, X_test, y_train, y_test = train_test_split(X, 
    #                                                    y, 
    #                                                    train_size=0.8, 
    #                                                    shuffle=True, 
    #                                                    random_state=6604)
    r_squared_dict = {}
    #bslr_table = PrettyTable()
    #table.header = False
    #bslr_table.field_names = ["Iteration", #"AIC", "BIC", 
    #                          "Adj-R^2", "DroppedFeature", "PValue from T-test"]
    #droppedFeatureList = []
    #modelSummaryList = []
    for i in range(len(X.columns)):
        #lr_model_1 = sm.OLS(y_train, X_train).fit()
        lr_model_1 = sm.OLS(y, X).fit()
        #y_pred = lr_model_1.predict(X_test)
        #print(predSales_CarSeat_scaled)
        # AIC, BIC and Adjusted R2 and t-test
        #print(f'AIC is {lr_model_1.aic}')
        #print(f'BIC is {lr_model_1.bic}')
        #print(f'Adj-R^2 is {lr_model_1.rsquared_adj}')
        #print(f'T-values are: {lr_model_1.t_test}')
        r_matrix_list = []
        for j in range(len(lr_model_1.params)):
            r_matrix = np.zeros_like(lr_model_1.params)
            r_matrix[j] = 1
            r_matrix_list.append(r_matrix)
        tTest_pValue_param_pairs = sorted([(lr_model_1.t_test(r_matrix_list[k]).pvalue, lr_model_1.params.index[k]) for k in range(len(lr_model_1.params))], 
                                          reverse=True)
        pValueOfDroppedFeature, droppedFeature = tTest_pValue_param_pairs[0]
        #droppedFeatureList.append(droppedFeature)
        #modelSummaryList.append(lr_model_1.summary())
        #X_test.drop(droppedFeature, axis=1, inplace=True)
        #X_train.drop(droppedFeature, axis=1, inplace=True)
        X.drop(droppedFeature, axis=1, inplace=True)
        #bslr_table.add_rows([[i+1, #np.round(lr_model_1.aic, 3), np.round(lr_model_1.bic, 3), 
        #                      np.round(lr_model_1.rsquared_adj, 3), 
        #                      droppedFeature, 
        #                      np.round(pValueOfDroppedFeature, 3)]])
        r_squared_dict.update({droppedFeature : float(100*np.round(lr_model_1.rsquared_adj, 5))})
    #return bslr_table
    return r_squared_dict

