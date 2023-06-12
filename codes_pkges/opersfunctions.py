# -*- coding: utf-8 -*-
"""
Operators functions for Data Analyst
@author: asngar
"""
# ----
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
    return df1, df2, df.dtypes.value_counts().plot.pie()

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



    # -- function which delete all variables with more than 90% of missing value --

def remov_mv90(df):
    print("New data with only less than 90% of missing values ")
    df_new = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]
    return df_new

    # -- function which delete all ouliers values --

def outliers(df,target):
    Q1 = df["target"].quantile(0.25)
    Q3 = df["target"].quantile(0.75)
    IQR = Q3 - Q1
    No_outliers = df[~((df["target"] < (Q1-1.5*IQR)) | (df["target"] > (Q3 + 1.5*IQR)))]
    return No_outliers

    # -- function which split dataset according the target categori --
def sub_target_df(df,target):
    target_yes = df[df[target] == "positive"]
    target_no = df[df[target] == "negative"]
    return   target_yes, target_n

    # --- function which write dataset_seaborn list and read it afterward ---

def names_dtaset_sns(filename):
    dataset_names = sns.get_dataset_names()

    with open(filename, 'w') as f:
        for dataset_name in dataset_names:
            f.write(dataset_name + '\n')

    with open(filename, 'r') as f:
        print(f.read())

    # -- Missing values in a dataset --

def check_missing_values(dataset):
    """
    Cette fonction prend en entrée un dataset (pandas dataframe) et vérifie si
    le dataset contient des valeurs manquantes. Elle retourne le nombre de valeurs
    manquantes pour chaque colonne.
    """
    missing_values_count = dataset.isnull().sum()
    if missing_values_count.sum() == 0:
        return f"Il n'y a pas de valeurs manquantes dans le dataset."
    else:
        print("Le dataset contient des valeurs manquantes :")
        return missing_values_count

    # -- Removing columns with missing values at 75%  --

def rm_misvalues_cols(dataset):
    """
    Cette fonction prend en entrée un dataset (pandas dataframe) et supprime
    les colonnes qui ont un nombre de valeurs manquantes supérieur au 3ème quartile.
    """
    # Calculer le 3ème quartile des valeurs manquantes pour chaque colonne
    missing_values_count = dataset.isnull().sum()
    quartile_3 = np.quantile(missing_values_count, 0.75)

    # Sélectionner les colonnes à conserver (celles qui ont un nombre de valeurs manquantes
    # inférieur ou égal au 3ème quartile)
    columns_to_keep = missing_values_count[missing_values_count <= quartile_3].index

    # Supprimer les colonnes à partir du dataset
    dataset = dataset[columns_to_keep]

    # Afficher un message pour indiquer les colonnes supprimées
    removed_columns = set(missing_values_count.index) - set(columns_to_keep)
    if len(removed_columns) > 0:
        print("Les colonnes suivantes ont été supprimées en raison d'un grand nombre de valeurs manquantes :")
        print(removed_columns)

    # Retourner le dataset sans les colonnes supprimées
    return dataset
