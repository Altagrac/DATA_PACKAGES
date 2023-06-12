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
