# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Mon Feb  13
@author: ADE
Abouts : Classifications models with & without data preprocessing
"""

# ******
    ## -- Packages --

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


#class Classification_models():
    ## -- Classifiers models --

names = [ "Nearest Neighbors", "Linear SVM", "RBF SVM",
           "Gaussian Process", "Decision Tree", "Random Forest",
            "AdaBoost", "Naive Bayes"]

classifiers = [ KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),
                SVC(gamma=2, C=1), GaussianProcessClassifier(1.0 * RBF(1.0)),
               DecisionTreeClassifier(max_depth=5),  RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               AdaBoostClassifier(), GaussianNB()]

#************

    ## -- Method without preprocessing data --

def clas_ml(X_train,y_train,X_test,y_test):
    results = dict()
    global  predict_rslts
    predict_rslts = dict()
    for name, clasml in zip(names,classifiers):
        model = clasml                            # calling  model for each classifier
        model.fit(X_train,y_train)                # fitting the model
        score_ml = model.score(X_test,y_test)     # scoring the model
        results[name] = score_ml                  #
        predict_rslts[name] = model.predict(X_test)
    return "DONE!!!"

    ## -- Method with preprocessing data --

def pipe_mlclas(preproc, X_tr1, y_tr1, X_ts, y_ts):
    results = dict()
    for name, clf in zip(names,classifiers):
        model = make_pipeline(preproc, clf)
        model.fit(X_tr1, y_tr1)
        scr_ml = model.score(X_ts, y_ts)
        results[name] = scr_ml
    return results

def conf_matrix(y_test):
    print(" CONFUSION MATRIX BY MODEL")
    print("........................")
    for pred in predict_rslts:
        cm= metrics.confusion_matrix(y_test,predict_rslts[pred])
        cm_display = metrics.ConfusionMatrixDisplay(cm).plot()
        #cm_display.plot()
        plt.title(f"{pred} Confusion matrix")
        filename = f"{pred} Confusion matrix"
        plt.savefig(filename+'.png')



def eval_report(y_test):
    items = list(predict_rslts.items())
    Accuracy = dict()
    Sensitivity_recall = dict()
    Precision = dict()
    Specificity = dict()
    F1_score = dict()
    print(" MODELS EVALUATION REPORT")
    print("........................")
    for  key, pred in items:
        #print(key,'-->', pred)
        Accuracy[key] = metrics.accuracy_score(y_test,pred)
        Precision[key] = metrics.precision_score(y_test, pred)
        Sensitivity_recall[key] = metrics.recall_score(y_test, pred)
        Specificity[key] = metrics.recall_score(y_test, pred, pos_label=0)
        F1_score[key] = metrics.f1_score(y_test, pred)

    df_acc = pd.DataFrame(Accuracy.values(), index = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                                                   "Gaussian Process", "Decision Tree", "Random Forest",
                                                    "AdaBoost", "Naive Bayes"], columns =["Accuracy"])
    df_prec = pd.DataFrame(Precision.values(), index = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                                                   "Gaussian Process", "Decision Tree", "Random Forest",
                                                    "AdaBoost", "Naive Bayes"], columns =["Precision"])
    df_recall = pd.DataFrame(Sensitivity_recall.values(), index = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                                                   "Gaussian Process", "Decision Tree", "Random Forest",
                                                    "AdaBoost", "Naive Bayes"], columns =["Sensitivity_recall"])
    df_Spec = pd.DataFrame(Specificity.values(), index = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                                                   "Gaussian Process", "Decision Tree", "Random Forest",
                                                    "AdaBoost", "Naive Bayes"], columns =["Specificity"])
    df_F1 = pd.DataFrame(F1_score.values(), index = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
                                                   "Gaussian Process", "Decision Tree", "Random Forest",
                                                    "AdaBoost", "Naive Bayes"], columns =["F1_score"])

    reports = pd.concat([df_acc, df_prec, df_recall,df_Spec, df_F1],axis=1)
    return reports
