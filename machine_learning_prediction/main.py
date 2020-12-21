from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import numpy as np
from sklearn.pipeline import make_pipeline
import time
from scipy.stats import pearsonr,spearmanr
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.pyplot import clabel
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import pandas as pd

data_array = np.loadtxt('shuffled_fatigue21.csv',encoding='utf-8-sig',delimiter=',')
# data_array = pd.read_csv('fatigue21.csv')
# data_array = data_array.sample(frac=1).reset_index(drop=True)
# data_array.to_csv('shuffer_fatigue21.csv', index=False)
# data_array = data_array.values

time_start = time.time()
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),GradientBoostingRegressor(n_estimators=300, learning_rate=0.01, max_depth=4,random_state=33)),
#                         param_grid=dict(),cv=10)
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),GradientBoostingRegressor()),
#                         param_grid={'gradientboostingregressor__n_estimators':[100,200,300,400,500],'gradientboostingregressor__learning_rate':[0.001,0.01,0.1,1],'gradientboostingregressor__max_depth':[1,2,3,4,5]},cv=10,scoring='r2',return_train_score=True)
#      param_grid={'gradientboostingregressor__n_estimators':[300],'gradientboostingregressor__learning_rate':[0.1],'gradientboostingregressor__max_depth':[2]},scoring='r2',cv=10)
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor( n_estimators=300,ccp_alpha=1.0)),param_grid=dict(),cv=10)
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),DecisionTreeRegressor(max_depth=10,min_samples_split=2,min_samples_leaf=1)),param_grid=dict(),cv=10)
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),DecisionTreeRegressor()),param_grid={'decisiontreeregressor__max_depth':[10]},cv=10)
# model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),DecisionTreeRegressor(max_depth=10,min_samples_split=2,min_samples_leaf=1)),param_grid={'decisiontreeregressor__ccp_alpha':[0.01,0.1,1.0]},cv=10)
# param_grid=dict(svr__gamma=[0.1],svr__C=[200])
model_pz = GridSearchCV(make_pipeline(preprocessing.StandardScaler(),SVR()), param_grid={'svr__gamma':[0.05],'svr__C':[200]},cv=10,refit=True)
# model_pz = GridSearchCV(make_pipeline(StandardScaler(),Lasso()), param_grid ={'lasso__alpha':[0.01,1,0.1]},cv=10,verbose=10)
X = data_array[:,0:-1]
Y = data_array[:,-1]
# X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=33,test_size=0.1)
# model_pz.fit(X_train,Y_train)
model_pz.fit(X,Y)
# model_pz = model_pz.sample(frac=1).reset_index(drop=True)
# print(model_pz.scoring)
# print(model_pz.return_train_score)
print(model_pz.best_params_)
print(model_pz.best_score_)
print(model_pz.cv_results_['mean_test_score'])
# print(model_pz.score(X,Y))
# svm_pre = model_pz.predict(X)
Y_pre=model_pz.predict(X)
print(Y_pre)
time_end = time.time()
# print(svm_pred)
# print(Y)
a=[0,1400]
# plt.figure(1)
# plt.figure(figsize=(500,500))
fig = plt.gcf()
fig.set_size_inches(3.5, 2.5)
plt.xlabel('Real Value', fontdict={'family': 'Times New Roman', 'size': 10})
plt.ylabel('Predicted Value', fontdict={'family': 'Times New Roman', 'size': 10})
plt.xticks(fontproperties = 'Times New Roman', size = 10)
plt.yticks(fontproperties = 'Times New Roman', size = 10)
x_major_locator=MultipleLocator(200)
y_major_locator=MultipleLocator(200)
plt.xlim(0,1500)
plt.ylim(0,1500)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.yaxis.set_major_locator(y_major_locator)
# plt.hlines(1200, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
# plt.hlines(1000, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
# plt.hlines(800, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
# plt.hlines(600, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
# plt.hlines(400, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
# plt.hlines(200, 0, 1399, colors='black', label='',linestyles='solid',linewidth=0.7)
plt.scatter(Y, Y_pre, c='black',s = 8, alpha=0.6)
plt.plot(a,a,c='black')
# lt.annotate(r'$2x+1=%s$'%y0,xy=(x0,y0),xytext=(+30,-30),textcoords='offset points',fontsize=10)
# plt.text(20,1400,r'$feture\ number\ =\ 21$',fontdict={'size':'10','color':'black','family': 'Times New Roman'})
# plt.text(20,1300,r'$model\ =\ Decision\ Tree$',fontdict={'size':'10','color':'black','family': 'Times New Roman'})
# plt.text(20,1300,r'$model\ =GBDT$',fontdict={'size':'10','color':'black'})
# plt.text(100,1200,r'$R^2\ =\ 99.50\%$',fontdict={'size':'10','color':'black','family': 'Times New Roman'})#svr
plt.text(100,1200,r'$R^2\ =\ 99.67\%$',fontdict={'size':'10','color':'black','family': 'Times New Roman'})#GBDT
# plt.text(20,1100,r'$RMSE\ =\ 10.73\%$',fontdict={'size':'10','color':'black'})
plt.show()
plt.savefig('GBDT_021.png')
plt.cla()

# print(mean_squared_error(Y, svm_pred, sample_weight=None, multioutput='uniform_average'))
# time = 1000000 * ( end.tv_sec - start.tv_sec ) + end.tv_usec - start.tv_usec;
# 	cout << "time：" << time << "s" <<endl;

# print(time_end-time_start)
print(r2_score(Y,Y_pre))
# print(mean_squared_error(Y,Y_pre))
print(np.sqrt(mean_squared_error(Y,Y_pre)))
print(mean_absolute_error(Y,Y_pre))
# print(mean_absolute_error(Y_test,Y_pre))
from numpy.random import randn
from numpy.random import seed
# seed(1)
# # data1=20*randn(1000)+100
# # data2=data1+(10*randn(1000)+50)
# for i in range(15):
#     # for j in range(15):
#     corr,_=spearmanr(data_array[:,i],data_array[:,-1])
#     print(corr)
