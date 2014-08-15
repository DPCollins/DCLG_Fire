from __future__ import division
import pandas as pd
import numpy as np
from numpy import *
import scipy
import scipy.fftpack
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import scatter_matrix
from pandas.tools.rplot import *
from pandas.tools.plotting import bootstrap_plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model
import scipy.sparse as sps
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from sklearn.svm import SVR
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import train_test_split

from sklearn.grid_search import GridSearchCV

# /////////////////// Load Data and Preprocess ////////////////////////

plt.close('all')
# df = pd.read_csv('ModelData_TRIM.csv')
df = pd.read_csv('ModelData_OutliersRemoved.csv')
df2 = df.dropna(axis=0)
cols = df.columns[2:]

colx = ['Fire_Incidents2012', '%Owned_mortgageOrLoan', 'All Household Spaces2011', 'population_estimates_2011', 'Cars_Per_Household_(2011)', '%Flat_maisonette_apartment', '%_Domestic_Buildings_(2005)', 'All Dwellings2011', 'Average_PTAL_Score2011_(pub transport accessibility)', 'Total Crime Rate 2011/12']



colsY = df.columns[1:]

# Dat = df2[cols]

Dat = df2[colx]
Targ = df2['Fire_Incidents2012']

# df3 = df2[df2.columns[1:]]

df3 = df2[colx]

x = shuffle(df3.index)
SET = df3.ix[x]

split = 0.8
X_train = SET[SET.columns[1:]].ix[SET.index[0:split*len(SET)]]
y_train = SET[SET.columns[0]].ix[SET.index[0:split*len(SET)]]

X_test = SET[SET.columns[1:]].ix[SET.index[split*len(SET):]]
y_test = SET[SET.columns[0]].ix[SET.index[split*len(SET):]]

# X, y = shuffle(Dat, Targ, random_state=1)
# X, y = shuffle(Dat, Targ, random_state=1)
# 
# X = X.astype(np.float32)
# offset = int(X.shape[0] * 0.9)
# X_train, y_train = X[:offset], y[:offset]
# X_test, y_test = X[offset:], y[offset:]

##########################################################

# X_train, X_test, y_train, y_test = train_test_split(Dat,
#                                                     Targ,
#                                                     test_size=0.1,
#                                                     random_state=1)


#R = preprocessing.scale(df2[cols])

##########################################################
#Fit Regression Model

param_grid = {'learning_rate': [0.02, 0.01],
			  'max_depth': [9,12,17],
			  'min_samples_leaf': [7, 12, 17]
	
			  }
est = ensemble.GradientBoostingRegressor(n_estimators=5000)
gs_cv = GridSearchCV(est, param_grid).fit(X_train, y_train)
gs_cv.best_params_


params = {'n_estimators': 5000, 'max_depth': 17,
          'learning_rate': 0.02, 'loss': 'ls', 'min_samples_leaf': 7}




clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))

meanMSE = np.sum(np.square(y_test-y_train.mean()))/(len(y_test))
print("MSE: %.4f" % mse)
print("MeanMSE: %.4f" %meanMSE)

###############################################################################
# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
        label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
        label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

###############################################################################
# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
# sorted_idx = sorted_idx[-20:]

pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, SET.columns[1:][sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()

