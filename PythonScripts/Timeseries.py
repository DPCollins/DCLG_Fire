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
from datetime import *

# /////////////////// Load Data and Preprocess ////////////////////////

plt.close('all')
df = pd.read_csv('LFB_Samp1000.csv')
# df2 = df.dropna(axis=0)

# lst = ['West End', 'St James'+"'"+'s', 'West Putney', 'Hounslow West']

LW = list(df['WardName'].unique())


Q = df[df['WardName']==LW[30]]  # select ward you're interested in


Q = df

Q2 = Q['DateOfCall']
Q2.index = range(0,len(Q2))
Q3 = pd.DataFrame(Q2)
Q3['Fires'] = 1

T = Q3.groupby(['DateOfCall']).sum()
T['date'] = T.index

T['dateDT'] = T['date'].apply(lambda x: datetime.strptime(x, "%d-%b-%y"))
T['DOW'] =  T['dateDT'].apply(lambda x: datetime.weekday(x))   # Mon - 0 Sun -6

T['date'] = T['dateDT'].apply(lambda x: str(x)[0:10])

T2 = T.sort(['date'], ascending=True)
T2.index = T2['date']
del T2['date']
del T2['dateDT']

#########################################################################
#Weather Data

W = pd.read_csv('Weather.txt')
y = ['GMT','MaxTempC','MeanTempC','MinTempC','DewPointC','MeanDewPointC','MinDewPointC','MaxHumidity','MeanHumidity','MinHumidity','MaxSeaLevelPressurePa','MeanSeaLevelPressurePa','MinSeaLevelPressurePa','MaxVisibilityKm','MeanVisibilityKm','MinVisibilityKm','MaxWindSpeedKm/h','MeanWindSpeedKm/h', 'MaxGustSpeedKm/h','Precipitation_mm','CloudCover','Events','WindDirDegrees']

W.columns = y
del W['MaxGustSpeedKm/h']
QQ = W.ix[:len(W)-2]
QQ['GMT'] = QQ['GMT'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
QQ['GMT'] = QQ['GMT'].apply(lambda x: datetime.strftime(x, "%Y-%m-%d"))
QQ.index = QQ['GMT']

del QQ['GMT']

OV = T2.join(QQ)

##########################################################################################
#Lag Days

Fir = OV['Fires']
YY = pd.DataFrame(Fir)
YY['Lag1'] = 0
YY['Lag2'] = 0
YY['Lag3'] = 0
YY['Lag4'] = 0
YY['Lag5'] = 0
YY['Lag6'] = 0
YY['Lag7'] = 0
YY['Lag8'] = 0

for j in range(8, len(YY)):
	YY['Lag1'][j] = YY.ix[j-1,0]
	YY['Lag2'][j] = YY.ix[j-2,0]
	YY['Lag3'][j] = YY.ix[j-3,0]
	YY['Lag4'][j] = YY.ix[j-4,0]
	YY['Lag5'][j] = YY.ix[j-5,0]
	YY['Lag6'][j] = YY.ix[j-6,0]
	YY['Lag7'][j] = YY.ix[j-7,0]
	YY['Lag8'][j] = YY.ix[j-8,0]

del YY['Fires']
OV2 = OV.join(YY)

# WW = pd.rolling_sum(OV2['Fires'], 7)

OV3 =  OV2.ix[8:,:]

# KK = OV3.join(pd.DataFrame(WW))

#########################################################################

idx = range(0,len(OV3))
IX = shuffle(idx)

split = 0.6

XT =  OV3.ix[IX]

# XT = KK.ix[IX]

del XT['Events']
del XT['CloudCover']

cols = ['Fires','DOW','Lag1','Lag2','Lag3','Lag4','Lag5','Lag6','Lag7','Lag8']

# cols = [0,'DOW','Lag1','Lag2','Lag3','Lag4','Lag5','Lag6','Lag7','Lag8']


XT = XT[cols]

X_Train = XT.ix[0:np.round(split*len(IX)),1:]
Y_Train =  XT.ix[0:np.round(split*len(IX)),0]

X_Test =  XT.ix[np.round(split*len(IX)):,1:]
Y_Test =  XT.ix[np.round(split*len(IX)):,0]


RF = RandomForestRegressor(n_estimators=100)

print "Fitting RF Model"			
RF.fit(X_Train, Y_Train)

Pred = np.round(RF.predict(X_Test))

r2 = r2_score(Pred, Y_Test)

print "R2 Score: ", r2

MSE_mean = np.sum(np.square(Y_Test - Y_Test.mean()))
MSE_pred = np.sum(np.square(Y_Test - Pred))

print "MSE Mean: ", MSE_mean
print "MSE Pred: ", MSE_pred

########################################################################################
#Poissson Distribution

Y = pd.DataFrame(OV3['Fires'])

j = np.random.poisson(Y.mean(),len(Y))

Y['Poisson'] = j


