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


# /////////////////// Load Data and Preprocess ////////////////////////

plt.close('all')
df = pd.read_csv('ModelData_TRIM.csv')
df2 = df.dropna(axis=0)
cols = df.columns[2:]

colsY = df.columns[1:]


des = df2['Fire_Incidents2012']
C = pd.qcut(des,10)
df2['Quartile'] = C.labels

R = preprocessing.scale(df2[cols])

# /////////////////////////////////////////////////////////////////////

supScore = []
FeatScore = []

LRsupScore = []
LRFeatScore = []

RFsupScore = []
RFFeatScore = []

features = shuffle(range(0,83))

# test from xcorrelation plot

# colx = ['%Chinese', 'Total Crime Rate 2011/12', '%Flat_maisonette_apartment', 'All Dwellings2011', 'Average_PTAL_Score2011_(pub transport accessibility)']

#/////////////

colx = ['%Owned_mortgageOrLoan', '%_Domestic_Buildings_(2005)', 'All Dwellings2011', 'Average_PTAL_Score2011_(pub transport accessibility)', 'Total Crime Rate 2011/12']


# P = shuffle(df2[cols].columns[:-1])
# colx = P[0:5]
# colx[4] = 'Total Crime Rate 2011/12'  


# colx = colx[3:]

TrainSzeSVR = []
TrainSzeLR = []
TrainSzeRF = []

for it in range(7,8):

	print "Training Size %i%%" %(it*10)
# 	for j in range(1,len(colx)+1):
	for j in range(len(colx),len(colx)+1):
	# for j in range(1,84):
		print "%i Features" %j
		supScore = []
		LRsupScore = []

		for i in arange(0,500):
			if i%50 == 0:
				print i
# 			Frac = 0.70
			Frac = it/10
			x = df2.index.values
			idx = shuffle(x)
			TT = pd.DataFrame(R)
			TT.columns = df2[cols].columns
			TT.index = df2.index
	# 		features = shuffle(range(0,TT.shape[1]))
		
	# ////To test for important Features
	# 		TT = TT[features[0:j]]    

	
	# /////////// To Test cols after PCS /////////////////////////
			TT = TT[colx[0:j]]
			
			TT2 = pd.DataFrame(R)
			TT2.index = TT.index
			TT2['Fires'] = df2['Fire_Incidents2012'].values
			# print "Saving File"
# 			TT2.to_csv('RTestRF.csv')
			
			FiresTrain = df2['Fire_Incidents2012'][idx[0:Frac*len(idx)]].astype('float')
			FiresTest = df2['Fire_Incidents2012'][idx[Frac*len(idx):]].astype('float')
		
			
			
			# FiresTrain = preprocessing.scale(FiresTrain)
# 			FiresTest = preprocessing.scale(FiresTest)
			
			SV = SVR(kernel='linear')
			LR = linear_model.LinearRegression()
			RF = RandomForestRegressor(n_estimators=5, max_features=50)
			
			
			SV.fit(TT.ix[idx[0:Frac*len(idx)],:], FiresTrain)
			LR.fit(TT.ix[idx[0:Frac*len(idx)],:], FiresTrain)
			RF.fit(TT2.ix[idx[0:Frac*len(idx)],:-1], TT2['Fires'][idx[0:Frac*len(idx)]] )
		
			Pred = SV.predict(TT.ix[idx[Frac*len(idx):],:]) 
			LRpred = LR.predict(TT.ix[idx[Frac*len(idx):],:])
			RFpred = RF.predict(TT2.ix[idx[Frac*len(idx):],:-1])
		
			# Estimate Mean
			
			mseMean = (np.sum(np.square((FiresTest - np.mean(FiresTrain)))))/len(FiresTest)
			
			# Support Vector Regression
		
			mse = np.mean((FiresTest - Pred)**2)
			mae = np.median(np.abs(FiresTest - Pred))
			r2 = r2_score(FiresTest,Pred)
			adjr2 = 1-(((len(Pred)-1)*(1-r2))/(len(Pred)-j-1))
			supScore.append([r2, adjr2, mse, mae, mseMean])
			Q = pd.DataFrame(supScore)
		
			# Linear Regression
			
			mseLR = np.mean((FiresTest - LRpred)**2)
			maeLR = np.median(np.abs(FiresTest - LRpred))
			r2LR = r2_score(FiresTest,LRpred)
			adjr2LR = 1-(((len(LRpred)-1)*(1-r2LR))/(len(LRpred)-j-1))
			LRsupScore.append([r2LR, adjr2LR, mseLR, maeLR, mseMean])
			QLR = pd.DataFrame(LRsupScore)
			
			# Random Forest Regression
			
			mseRF = np.mean((FiresTest - RFpred)**2)
			maeRF = np.median(np.abs(FiresTest - RFpred))
			r2RF = r2_score(FiresTest,RFpred)
			adjr2RF = 1-(((len(RFpred)-1)*(1-r2RF))/(len(RFpred)-j-1))
			RFsupScore.append([r2RF, adjr2RF, mseRF, maeRF, mseMean])
			QRF = pd.DataFrame(RFsupScore)
		

		FeatScore.append([features[j-1],Q[0].mean(),Q[0].std(), Q[1].mean(), Q[1].std(), Q[2].mean(), Q[2].std(), Q[3].mean(), Q[3].std(), Q[4].mean(), Q[4].std()])
		LRFeatScore.append([features[j-1],QLR[0].mean(),QLR[0].std(), QLR[1].mean(), QLR[1].std(), QLR[2].mean(), QLR[2].std(), QLR[3].mean(), QLR[3].std(), QLR[4].mean(), QLR[4].std()])
		RFFeatScore.append([features[j-1],QRF[0].mean(),QRF[0].std(), QRF[1].mean(), QRF[1].std(), QRF[2].mean(), QRF[2].std(), QRF[3].mean(), QRF[3].std(), QRF[4].mean(), QRF[4].std()])
	
	# DataFrame Business SVR

	Feature = pd.DataFrame(FeatScore)
	Feature.columns = ['Feature', 'R2', 'R2_(SD)', 'R_adj', 'R_adj(SD)', 'MSE', 'MSE_(SD)', 'MAE', 'MAE_(SD)', 'MeanMSE', 'MeanMSE(SD)']
	FeatureC = Feature

	# DataFrame Business LR

	FeatureLR = pd.DataFrame(LRFeatScore)
	FeatureLR.columns = ['Feature', 'R2', 'R2_(SD)', 'R_adj', 'R_adj(SD)', 'MSE', 'MSE_(SD)', 'MAE', 'MAE_(SD)', 'MeanMSE', 'MeanMSE(SD)']
	FeatureCLR = FeatureLR
	
	# DataFrame Business RF
	
	FeatureRF = pd.DataFrame(RFFeatScore)
	FeatureRF.columns = ['Feature', 'R2', 'R2_(SD)', 'R_adj', 'R_adj(SD)', 'MSE', 'MSE_(SD)', 'MAE', 'MAE_(SD)', 'MeanMSE', 'MeanMSE(SD)']
	FeatureCRF = FeatureRF
	
	
	FeatureC['Feature'] =   colx
	FeatureCLR['Feature'] = colx
	FeatureCRF['Feature'] = colx
	
# 	FeatureC['Feature'] = df2[cols][features].columns
# 	FeatureCLR['Feature'] = df2[cols][features].columns

TrainSzeSVR.append([FeatureC['R2'], FeatureC['R2_(SD)'], FeatureC['R_adj'], FeatureC['R_adj(SD)'], FeatureC['MSE']])
TrainSzeLR.append([FeatureCLR['R2'], FeatureCLR['R2_(SD)'], FeatureCLR['R_adj'], FeatureCLR['R_adj(SD)'], FeatureCLR['MSE']]) 
TrainSzeRF.append([FeatureCRF['R2'], FeatureCRF['R2_(SD)'], FeatureCRF['R_adj'], FeatureCRF['R_adj(SD)'], FeatureCRF['MSE']]) 


# Plotting

plt.scatter(FeatureC.index,FeatureC['R_adj'], c='r', s=40)
plt.scatter(FeatureC.index,FeatureC['R2'], s=40)

plt.figure()
plt.scatter(LRpred,FiresTest)
plt.plot(FiresTest,FiresTest)
plt.title('Linear Regression')

plt.figure()
plt.scatter(Pred,FiresTest)
plt.plot(FiresTest,FiresTest)
plt.title('SVM Regression')

plt.figure()
plt.scatter(RFpred,FiresTest)
plt.plot(FiresTest,FiresTest)
plt.title('Random Forest Regression')
