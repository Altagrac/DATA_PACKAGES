# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Mon Feb  13 
@author: ADE

"""
# ******
    ## -- Packages --
    
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.datasets import make_classification, load_iris
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer,make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder,RobustScaler

    ## -- Encoding numerics variables --

def nums_values(dtrain, dtest_or_dtrain, oper_model):
    colmns_name = list(dtest_or_dtrain.select_dtypes(include = np.number))
    model = oper_model
    model.fit(dtrain[colmns_name]) 
    new_dtaset = pd.DataFrame(oper_model.transform(dtest_or_dtrain[colmns_name]))
    return new_dtaset

    ## -- Encoding categorics variables --

def encode_cat_values(dtrain,dtest_or_dtrain, oper_model):
    colmns_name = list(dtest_or_dtrain.select_dtypes(exclude= np.number))
    encod= oper_model
    encod.fit_transform(dtrain[colmns_name])
    df = pd.DataFrame.sparse.from_spmatrix(encod.transform(dtest_or_dtrain[colmns_name]))
    return df

    ## -- New data set with both numerics & categorics values encod --

# def df_encod(dtrain, dtest_or_dtrain, scaler): 
#     return pd.concat([encode_cat_values,nums_values], axis=1)


#**************

# # ==> Il faut scaler le X_test_2_prep_df. Avec la même methodologie que le X_train_2, sans faire de dataLeakeage. 
# # ==> Créer les methodes appropriées pour réutiliser la logique.

# def scaled(X_train, X_test, scaler):
#     columns = X_test._get_numeric_data().columns.values.tolist()
#     scaler.fit(X_train[columns])
#     df = pd.DataFrame(scaler.transform(X_test[columns]))
#     return df.rename(columns={i:f'{columns[i]}_scaled' for i in range(len(columns))})

# def oheed(X_train, X_test):
#     columns = list(set(X.columns)-set(X_test._get_numeric_data().columns))
#     ohe.fit(X_train[columns])
#     return pd.DataFrame.sparse.from_spmatrix(ohe.transform(X_test[columns]))

# def encod_df(X_train, X_test, scaler): 
#     return pd.concat([oheed(X_train, X_test), scaled(X_train, X_test, scaler)], axis=1)


#*********************

    ## -- Preprocessor workflow -- 

def data_pipeline(prep_num, prep_cat):
    
    #Trie entre les variables : catégorielles & numériques 
    var_numeriques = make_column_selector(dtype_include=np.number)
    var_categorielles = make_column_selector(dtype_exclude=np.number)
                                         
     #generation du pipeline : une transformation en serie sur les groupes de variables en fonction de leur nature
    numerique_pipe = make_pipeline(SimpleImputer(strategy = "mean"),prep_num)
    categorielle_pipe = make_pipeline(prep_cat)                                        
     #Application des transformations sur l'ensemble des variables (features)
    preprocessor = make_column_transformer((numerique_pipe,var_numeriques),(categorielle_pipe,var_categorielles))
                               
    return preprocessor