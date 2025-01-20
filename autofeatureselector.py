# python module to auto-select features for an ML algorithm

import sys
import numpy as np
import pandas as pd
import sklearn.feature_selection as skfs
import sklearn.preprocessing as skpp
import sklearn.linear_model as sklm
import sklearn.ensemble as ske
from lightgbm import LGBMClassifier

# all selector functions accept pandas dataframes X and y (y must be 1-d) 
# as well as a int for max num. of features to select

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


# default pre-processing function
# for pre-processing a specific dataset
# will change this for something generic later
def preprocess_dataset(dataset_path):
    # selects X and y
    data = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    data = data[numcols+catcols]
    traindf = pd.concat([data[numcols], pd.get_dummies(data[catcols])],axis=1)
    features = traindf.columns

    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf,columns=features)
    
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']

    # number of features to select
    num_feats = 30
    return X, y, num_feats

# auto-selector accepts the dataset path as a string (url or file path)
# as well as a list of strings to select methods
# accepted methods: pearson, chi-square, rfe, log-reg, rf, lgbm
# optionally accepts a function to pre-process the dataset and choose number of features

def autoFeatureSelector(dataset_path, methods=[], preprocess_funct=preprocess_dataset()):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_funct(dataset_path)
    features = list(X.columns)
    fn = len(features)
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        for i in range(fn):
            if not cor_support[i]:
                features[i] = None
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
        for i in range(fn):
            if not chi_support[i]:
                features[i] = None
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
        for i in range(fn):
            if not rfe_support[i]:
                features[i] = None
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        for i in range(fn):
            if not embedded_lr_support[i]:
                features[i] = None
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        for i in range(fn):
            if not embedded_rf_support[i]:
                features[i] = None
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        for i in range(fn):
            if not embedded_lgbm_support[i]:
                features[i] = None
    
    
    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    best_features = list(filter(None, features))
    
    return best_features

# main function
def main():
    path = input('Enter path of dataset:')
    methods = ['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm']
    print(autoFeatureSelector(path, methods))
    return 0

if __name__ == '__main__':
    sys.exit(main())