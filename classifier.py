# Import the neccessary modules for data manipulation and visual representation
import pandas as pd
import numpy as np
import heapq
#import matplotlib.pyplot as plt
#import seaborn as sns
import time

from sklearn.linear_model import LinearRegression,LogisticRegression,Lasso,Ridge
from sklearn import tree,preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import f1_score


def printwt(*args):
	print(datetime.now(), args)
if __name__ == '__main__':
	#Read the analytics csv file and store our dataset into a dataframe called "df"
	df = pd.read_csv('HR_comma_sep.csv', index_col=None)

	'''
	#Investigate features
	for index in df.columns:
		print(df[index].unique())
	'''

	# Renaming certain columns for better readability
	df = df.rename(columns={'satisfaction_level': 'satisfaction', 
	                        'last_evaluation': 'evaluation',
	                        'number_project': 'projectCount',
	                        'average_montly_hours': 'averageMonthlyHours',
	                        'time_spend_company': 'yearsAtCompany',
	                        'Work_accident': 'workAccident',
	                        'promotion_last_5years': 'promotion',
	                        'sales' : 'department',
	                        'left' : 'turnover'
	                        })
	'''
	cmat=df.corr()
	plt.subplots(figsize=(6,6))
	sns.heatmap(cmat,vmax=0.9,square=True)
	plt.show()
	'''
	
	# Convert these variables into categorical variables
	df["department"] = df["department"].astype('category').cat.codes
	tmp=pd.get_dummies(df['department'])
	df.drop(labels=['department'],axis=1,inplace=True)
	df=pd.concat([tmp,df],axis=1) 
	df["salary"]=df["salary"].astype('category').cat.set_categories(['low','medium','high'],ordered=True)
	df["salary"]=df["salary"].cat.codes


	target_label = df['turnover']
	df.drop(labels=['turnover'],axis=1,inplace=True)
	df.insert(0,'turnover',target_label)
	df.insert(1,'augmentation',1)

	X=df.drop('turnover',axis=1)
	y=df['turnover']
	X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y) #stratify keep the ratio of classes
	
	#preprocessing
	scaler=preprocessing.StandardScaler().fit(X_train) #Standardization
	X_train_scaled=scaler.transform(X_train)
	X_test_scaled=scaler.transform(X_test)


	#Feature Selection
	clf=tree.DecisionTreeClassifier()
	clf.fit(X_train_scaled,y_train)
	tmp=clf.feature_importances_
	index=heapq.nlargest(4,range(len(tmp)),tmp.take)
	index=sorted(index)
	print("Selected Features: "+str(X.columns[index]))

	X_new=X.ix[:,X.columns[index]]
	'''
	#only availabel when 2 features
	fig,ax=plt.subplots()
	ax.scatter(X.ix[:,X.columns[index[0]]],X.ix[:,X.columns[index[1]]],c=y,maker=)
	plt.ylabel(X.columns[index[1]],fontsize=10)
	plt.xlabel(X.columns[index[0]],fontsize=10)
	plt.show()
	'''
	

	#polynomial model
	#poly=PolynomialFeatures(2)
	#pX=poly.fit_transform(X_new)

	X_train,X_test,y_train,y_test=train_test_split(X_new,y,test_size=0.4,stratify=y) #stratify keep the ratio of classes
	
	#preprocessing
	scaler=preprocessing.StandardScaler().fit(X_train) #Standardization
	X_train_scaled=scaler.transform(X_train)
	X_test_scaled=scaler.transform(X_test)
	
	
	printwt("Logistic Regression Model:")
	model=LogisticRegression()
	penalty_c=['l1','l2']
	C_range=np.logspace(-3,3,50)
	param_grid=dict(penalty=penalty_c,C=C_range)
	grid=GridSearchCV(model,param_grid,cv=5,n_jobs=4,scoring='accuracy')
	grid.fit(X_train_scaled,y_train)
	printwt("The best parameters are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_))
	ACU_test=grid.score(X_test_scaled,y_test)
	print("Training error:%.4f"%(1-grid.cv_results_['mean_train_score'][grid.best_index_]))
	print("Validation error:%.4f"%(1-grid.cv_results_['mean_test_score'][grid.best_index_]))
	print("Test error:%.4f"%(1-ACU_test))
	y_pred=grid.predict(X_test_scaled)
	fsc=f1_score(y_test,y_pred)
	print("F_measure on testing set:%.4f"%fsc)
	
	
	
	printwt("Random Forest Classification:")
	model=RandomForestClassifier()
	#estimatros_r=np.logspace(1,4,20).astype(int) #93
	estimatros_r=[93]
	#features_r=['auto','sqrt','log2'] #sqrt
	features_r=['sqrt']
	#depth_r=np.linspace(1,300,20).astype(int) #48
	depth_r=[48]
	param_grid=dict(n_estimators=estimatros_r,max_features=features_r,max_depth=depth_r)
	grid=GridSearchCV(model,param_grid=param_grid,cv=5,n_jobs=4,scoring='accuracy')
	grid.fit(X_train_scaled,y_train)
	printwt("The best parameters are %s with a score of %0.4f" % (grid.best_params_, grid.best_score_))
	ACU_test=grid.score(X_test_scaled,y_test)
	print("Training error:%.4f"%(1-grid.cv_results_['mean_train_score'][grid.best_index_]))
	print("Validation error:%.4f"%(1-grid.cv_results_['mean_test_score'][grid.best_index_]))
	print("Test error:%.4f"%(1-ACU_test))
	y_pred=grid.predict(X_test_scaled)
	fsc=f1_score(y_test,y_pred)
	print("F_measure on testing set:%.4f"%fsc)

	'''
	x_min,x_max=X['satisfaction'].min()-1,X['satisfaction'].max()+1
	y_min,y_max=X['evaluation'].min()-1,X['evaluation'].max()+1
	xx,yy=np.meshgrid(np.arange(x_min,x_max,0.01),np.arange(y_min,y_max,0.01))
	Z=grid.predict(np.c_[xx.ravel(),yy.ravel()])
	Z=Z.reshape(xx.shape)
	plt.pcolormesh(xx,yy,Z,cmap=plt.cm.Paired)
	plt.xlim(xx.min(),xx.max())
	plt.ylim(yy.min(),yy.max())
	plt.show()
	'''

	
