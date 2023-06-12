# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Fri Feb  3 14:50:12 2023

@author: asnga
"""
# ******
    ## -- Packages --

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

    ## -- function to print dataset shape: --

def df_shape(df):
    print("This dataset contains:")
    # Nombers of rows :
    print(f"--> {df.shape[0]} lignes")
    # Nombers of columns
    print(f"--> {df.shape[1]} colones")
    # Data index
    print(f"--> {df.index}")


    ## -- function which return values counts by data types ---

def data_types(df):
    list_num = []
    list_cat = []
    for i in df.columns.tolist():
        if 'float64' in str(df[i].dtype) or 'int64' in str(df[i].dtype):
            list_num.append(i)
        else:
            list_cat.append(i)
    df1 = pd.DataFrame(list(list_num), columns=["Numerics_cols"])
    df2 = pd.DataFrame(list(list_cat), columns=["Categorials_cols"])
    df_typs = pd.concat([df1,df2],axis=1)
    return df_typs, df.dtypes.value_counts().plot.pie()

    ## -- function which compute percentage of missing values --

def prcentag_misval(df):
    print("Percentage of missing values by columns")
    df_perc = (df.isna().sum()/df.shape[0]).sort_values(ascending=True)
    return df_perc

    ## -- function wich give some statistiques informations on numerics features --

def stats(df):
    count=  df.describe().loc['count']
    avrg  = df.describe().loc['mean']
    Q1 = df.describe().loc['25%']
    median = df.describe().loc['50%']
    Q3 = df.describe().loc['75%']
    df_stat = pd.concat([avrg,Q1,median,Q3,count],axis=1)
    return df_stat

    ## -- function which give more details on target : --

def info_target(df,target):
    print("Target values & percentage by categorie\n")

    if df[target].dtypes == "int64":
        print(df[target].value_counts().to_frame())
    else :
        print(df[target].dtypes.value_counts())
    perc = df[target].value_counts(normalize=True).to_frame()
    return perc

## ========= VISUALIZATION ================

def set_title():
    fontdict =  {'fontsize': '16',
        'fontweight': 'bold',
        'color': "black"}
    return fontdict

    # -- function to plot all the missing values in the dataset --

def map_nan(df):
    textstr = "Total black means no missing values"
    sns.heatmap(df.isna(), cbar=False)
    plt.text(12, 12, textstr, fontsize=14)
    plt.show()

    ## -- fucntion to plot numerics features --

def plot_num(df):
    for col in df.select_dtypes(exclude="object"):
        plt.figure()
        sns.histplot(df[col],kde=True)
        plt.title(f"Distribution of {col} Data")
        plt.tight_layout()

     ## -- fucntion to plot categorics features --

def plot_cat(df):
    green_diamond = dict(markerfacecolor='g', marker='D')
    for col in df.select_dtypes(include="object"):
       # print(f"{col :-<30} {df[col].unique()}") : print with clean style
        plt.figure()
        df[col].value_counts().plot.box(flierprops=green_diamond)
        plt.title(f"Distribution of {col} Data")
        #plt.tight_layout()

    ## -- function to plot all dataset --

def df_distribution(data):
   #Histrogramme de distribution par variable
    plt.figure(figsize=(15,10))
    for i,col in enumerate(data.columns,1):
        plt.subplot(4,3,i)
        plt.title(f"Distribution of {col} Data")
        sns.histplot(data[col],kde=True)
        plt.tight_layout()

    filename = "Ensemble de distributions par colones"
    plt.savefig(filename+'.png')
    plt.show();
