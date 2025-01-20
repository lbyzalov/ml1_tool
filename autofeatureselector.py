# python module to auto-select features for an ML algorithm

import numpy as np
import pandas as pd
import sklearn.feature_selection as skfs
import sklearn.preprocessing as skpp
import sklearn.linear_model as sklm
import sklearn.ensemble as ske
from lightgbm import LGBMClassifier

# uses pearson correlation to select features
def cor_selector(X, y, num_feats):
    # list of features
    feats = X.keys().tolist()

    # measures correlation between every feature and y
    corr_vals = []
    for f in X:
        corr_vals.append(np.corrcoef(x=X[f], y=y)[0,1])

    # selects to feat_num features
    best_feat = X.iloc[:,np.argsort(np.abs(corr_vals))[-num_feats:]].columns.tolist()

    # list of bools for which features in X are selected
    corr_support = [True if i in best_feat else False for i in feats]
    return corr_support, best_feat

# uses chi-squared correlateion to select features
def chi_squared_selector(X, y, num_feats):
    # creates chi-sq. selector function and trains it on normalized data
    chi_sel = skfs.SelectKBest(skfs.chi2, k=num_feats)
    x_norm = skpp.MinMaxScaler().fit_transform(X)
    chi_sel.fit(x_norm, y)

    # gets num_feats best columns from chi_sel
    chi_support = chi_sel.get_support()
    chi_feat = X.loc[:,chi_support].columns.to_list()
    
    return chi_support, chi_feat

# uses recursive feature elimination wrapper to select features
def rfe_selector(X, y, num_feats):
    # creates and trains rfe on normalized data
    rfe_sel = skfs.RFE(estimator=sklm.LogisticRegression(max_iter=100), n_features_to_select=num_feats, step=0.05)
    x_norm = skpp.MinMaxScaler().fit_transform(X)
    rfe_sel.fit(x_norm, y=y)

    # gets support and feature lists
    rfe_support = rfe_sel.get_support()
    rfe_feat = X.loc[:,rfe_support].columns.to_list()
    
    return rfe_support, rfe_feat

# uses an embedded L1 algorithm to select features
def embedded_log_reg_selector(X, y, num_feats):
    # creates and trains embedded logistic regression on normalized data
    elr_sel = skfs.SelectFromModel(estimator=sklm.LogisticRegression(max_iter=100), max_features=num_feats)
    x_norm = skpp.MinMaxScaler().fit_transform(X)
    elr_sel.fit(x_norm, y=y)

    # gets support and feature lists
    embedded_lr_support = elr_sel.get_support()
    embedded_lr_feat = X.loc[:,embedded_lr_support].columns.to_list()
    
    return embedded_lr_support, embedded_lr_feat

# uses an embedded random forest classifier to select features
def embedded_rf_selector(X, y, num_feats):
    # creates and trains embedded rf classifier on normalized data
    erf_sel = skfs.SelectFromModel(estimator=ske.RandomForestClassifier(n_estimators=100, min_samples_split=5), max_features=num_feats)
    x_norm = skpp.MinMaxScaler().fit_transform(X)
    erf_sel.fit(x_norm, y=y)

    # gets support and feature lists
    embedded_rf_support = erf_sel.get_support()
    embedded_rf_feat = X.loc[:,embedded_rf_support].columns.to_list()
    
    return embedded_rf_support, embedded_rf_feat

# uses embedded lightgbm to select features
def embedded_lgbm_selector(X, y, num_feats):
    # creates and trains embedded lgbm classifier on normalized data
    elgbm_sel = skfs.SelectFromModel(estimator=LGBMClassifier, max_features=num_feats)
    x_norm = skpp.MinMaxScaler().fit_transform(X)
    elgbm_sel.fit(x_norm, y=y)

    # gets support and feature lists
    embedded_lgbm_support = elgbm_sel.get_support()
    embedded_lgbm_feat = X.loc[:,embedded_lgbm_support].columns.to_list()
    
    return embedded_lgbm_support, embedded_lgbm_feat