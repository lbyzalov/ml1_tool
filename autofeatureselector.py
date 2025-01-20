# python module to auto-select features for an ML algorithm

import numpy as np
import pandas as pd
import sklearn as sk

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

