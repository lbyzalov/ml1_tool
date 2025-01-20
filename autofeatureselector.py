# python module to auto-select features for an ML algorithm

import numpy as np
import pandas as pd
import sklearn as sk

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
    chi_sel = sk.feature_selection.SelectKBest(sk.feature_selectionchi2, k=num_feats)
    x_norm = sk.preprocessing.MinMaxScaler().fit_transform(X)
    chi_sel.fit(x_norm, y)

    # gets num_feats best columns from chi_sel
    chi_support = chi_sel.get_support()
    chi_feat = X.loc[:,chi_support].columns.to_list()
    
    return chi_support, chi_feat