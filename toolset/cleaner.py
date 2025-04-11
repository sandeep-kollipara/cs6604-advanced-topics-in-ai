# *************** Data Cleaner ***************

from pydantic import BaseModel, Field
from langchain.tools import tool
import pandas
import collections
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler, LabelEncoder


def EliminateConstantFeatures(dataframe):
    featurelist=dataframe.columns;
    integertypes=['int16', 'int32', 'int64'];
    floattypes=['float16', 'float32', 'float64'];
    for feature in featurelist:
        if True in dataframe[feature].isnull().unique():
            continue;
        elif dataframe[feature].dtype in integertypes and len(dataframe[feature].unique())==1:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of int class: "+str(feature)+" dropped.");
        elif dataframe[feature].dtype in floattypes and dataframe[feature].std()==0:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of float class: "+str(feature)+" dropped.");
        elif dataframe[feature].dtype==object and len(dataframe[feature].unique())==1:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature of object class: "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Constant Feature Elimination.");
    return dataframe;

def EliminateFeaturesWithHighNulls(dataframe, percentage):
    featurelist=dataframe.columns;
    dflen=len(dataframe);
    for feature in featurelist:
        null_dict=collections.Counter(dataframe[feature].isnull());
        if null_dict[True] > percentage*dflen/100:
            dataframe.drop([feature], axis=1, inplace=True);
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Features with High Nulls Elimination.");
    return dataframe;

def EliminateIdenticalFeatures(dataframe):#returns a different dataframe
    dataframe2=dataframe.copy();
    featurelist=dataframe.columns;
    newfeaturelist=dataframe.columns;
    for feature_1 in featurelist:
        for feature_2 in featurelist:
            if feature_1==feature_2:
                continue;
            else:
                if dataframe[feature_1].dtype != dataframe[feature_2].dtype:
                    continue;
                else:
                    temp_1=pandas.DataFrame(dataframe[feature_1].fillna(method='pad'));
                    temp_1=temp_1.rename(columns={feature_1:'A'});
                    temp_2=pandas.DataFrame(dataframe[feature_2].fillna(method='pad'));
                    temp_2=temp_2.rename(columns={feature_2:'A'});
                    if len(temp_1) != len(temp_2) or len(temp_1)==0 or len(temp_2)==0:
                        continue;
                    else:
                        if temp_1.equals(temp_2) and feature_1 in newfeaturelist and feature_2 in newfeaturelist:
                            dataframe2.drop([feature_2], axis=1, inplace=True);
                            print("Feature "+str(feature_1)+" and "+str(feature_2)+" are identical.");
                            newfeaturelist=newfeaturelist.drop(feature_2);
    dataframe=dataframe2;
    return dataframe2;

def EliminateFeaturesWithHighZeroes(dataframe, percentage):
    featurelist=dataframe.columns;
    dflen=len(dataframe);
    for feature in featurelist:
        if dataframe[feature].dtype != object:
            zero_dict=collections.Counter(dataframe[feature]);
            if zero_dict[0] > percentage*dflen/100:
                dataframe.drop([feature], axis=1, inplace=True);
                print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(featurelist))+\
          " from "+str(len(dataframe.columns))+" after Features with High Zeroes Elimination.");
    return dataframe;

def EliminateFeaturesWithNearZeroVariance(dataframe):
    selector=VarianceThreshold();
    labelenc=LabelEncoder();
    dfcol=len(dataframe.columns);
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
    dataframe.copy();
    for feature in dataframe.columns:
        temp=dataframe[feature].copy();
        temp=temp.dropna(axis=0, how='any').reset_index();
        if temp[feature].dtype not in numerics:
            temp[feature]=labelenc.fit_transform(temp[feature].astype(str));
        selector.fit(temp);
        if feature not in temp.columns[selector.get_support(indices=True)]:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+\
          " after Features with Near Zero Variance Elimination.");
    return dataframe;

def EliminateFeaturesWithHighNonnumericUniqueValues(dataframe, percentage, absolute):
    featurelist=dataframe.columns;
    numerics=['int16', 'int32', 'int64', 'float16', 'float32', 'float64'];
    nonnumfeaturelist=featurelist.drop(dataframe.select_dtypes(include=numerics).columns);
    dfcol=len(dataframe.columns);
    for feature in nonnumfeaturelist:
        if len(dataframe[feature].unique())/len(dataframe) > percentage/100:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
        elif len(dataframe[feature].unique()) > absolute:
            dataframe.drop([feature], axis=1, inplace=True);
            print("Feature "+str(feature)+" dropped.");
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+\
          " after Features with High Non-numeric Unique Values Elimination.");
    return dataframe;

def EliminateCorrelatedFeatures(dataframe, threshold):
    correlation_matrix=dataframe.corr().reset_index();
    dfcol=len(dataframe.columns);
    side=len(correlation_matrix);
    featurelist=correlation_matrix.columns;
    newfeaturelist=correlation_matrix.columns;
    featurelist=featurelist.drop('index');
    j=side;
    for feature in featurelist:
        i=side-j+1;
        while i < side:
            if correlation_matrix[feature][i] > threshold \
            and feature in newfeaturelist \
            and correlation_matrix['index'][i] in newfeaturelist:
                dataframe.drop([correlation_matrix['index'][i]], axis=1, inplace=True);
                print("Feature "+str(correlation_matrix['index'][i])+" dropped.");
                newfeaturelist=newfeaturelist.drop(correlation_matrix['index'][i]);
            i+=1;
        j-=1;
    print("Num. of features reduced to "+str(len(dataframe.columns))+\
          " from "+str(dfcol)+" after Correlated Features Elimination.");
    return dataframe;

def MissingValueTreatment(dataframe):
    featurelist=dataframe.columns;
    integertypes=['int16', 'int32', 'int64'];
    floattypes=['float16', 'float32', 'float64'];
    numerics=list(set(integertypes+floattypes));
    intfeaturelist=dataframe.select_dtypes(include=integertypes).columns;
    floatfeaturelist=dataframe.select_dtypes(include=floattypes).columns;
    nonnumfeaturelist=featurelist.drop(dataframe.select_dtypes(include=numerics).columns);
    for feature in intfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(round(dataframe[feature].mean()));
    for feature in floatfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(dataframe[feature].mean());
    for feature in nonnumfeaturelist:
        dataframe[feature]=dataframe[feature].fillna(dataframe[feature].mode().item());
    return dataframe;

def OutlierTreatment(dataframe):
    floattypes=['float16', 'float32', 'float64'];
    floatfeaturelist=dataframe.select_dtypes(include=floattypes).columns;
    for feature in floatfeaturelist:
        dataframe[feature].loc[dataframe[feature] < dataframe[feature].quantile(0.01)]=dataframe[feature].quantile(0.01);
        dataframe[feature].loc[dataframe[feature] > dataframe[feature].quantile(0.99)]=dataframe[feature].quantile(0.99);
    return dataframe;

