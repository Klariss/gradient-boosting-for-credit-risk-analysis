
# coding: utf-8

# In[1]:




# In[2]:

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.datasets import make_classification

from sklearn.model_selection import GridSearchCV   #Grid search

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.datasets import make_classification

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from numpy import corrcoef, sum, log, arange

from pylab import pcolor, show, colorbar, xticks, yticks


# In[3]:
def ParamTuning(X,y,tun_range,est,verb=False):
    gsearch1 = GridSearchCV(estimator = est, param_grid = tun_range, scoring='roc_auc',n_jobs=4,iid=False, cv=5)
    print("Wait Pls!")
    print("========================================")
    if verb:
        print(gsearch1.fit(X,y))
        print("========================================")
        print(gsearch1.cv_results_)
        print("========================================")
    else:
        gsearch1.fit(X,y)
    print(gsearch1.best_params_)
    print(gsearch1.best_score_)
    return


# In[4]:

data = pd.read_csv('./data/uci_credit_card/UCI_Credit_Card.csv')

df = data.copy()
target = 'default'


# In[5]:

predictors = df.columns.drop(['ID', target])
raw_X = np.asarray(df[predictors])
X = raw_X
y = np.asarray(df[target])
raw_data, raw_target = X, y
random_state=10


# In[6]:

first_est=GradientBoostingClassifier(learning_rate=0.1, n_estimators=50,
                                  max_depth=6, min_samples_split=450,
                                  min_samples_leaf=14,max_features='sqrt',
                                  subsample=0.78,random_state=random_state)


# In[7]:

first_est.fit(raw_data, raw_target )


# In[8]:

tr_X=X[:300,:].transpose()
R = corrcoef(tr_X)
pcolor(R)
colorbar()
yticks(arange(0,21),range(0,22))
xticks(arange(0,21),range(0,22))
show()


# In[9]:

from sklearn.feature_selection import RFE
model = LogisticRegression()
rfe = RFE(first_est, 3)
X_rfe, y_rfe=X,y
fit = rfe.fit(X_rfe, y_rfe)
print("Num Features:",fit.n_features_)
print("Selected Features:",fit.support_)
print("Feature Ranking: ",fit.ranking_)


# In[10]:

#pd.DataFrame(np.hstack([raw_data, raw_target[:,None]])).head(3)


# In[11]:

plt.figure(figsize=(18,10))
plt.title(u'Változók fontossága')
plt.bar(range(raw_data.shape[1]), first_est.feature_importances_,
       color="r", align="center")
plt.xticks(range(raw_data.shape[1]),('Li BAL', 'SEX', 'EDU', 'MAR', 'AGE',
       'PAY0', 'PAY2', 'PAY3', 'PAY4', 'PAY5', 'PAY6',
       'BILL1', 'BILL2', 'BILL3', 'BILL4', 'BILL5',
       'BILL6', 'P AMT1', 'P AMT2', 'P AMT3', 'P AMT4',
       'P AMT5', 'P AMT6'))
plt.xlim([-1, raw_data.shape[1]])
plt.show()


# In[12]:

X=pd.DataFrame(X)
X2=X[[0,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21]]
print(X.shape)
print(X2.shape)
raw_data, raw_target = X2, y
print(raw_data.shape)


# In[13]:


train, test, train_t, test_t = train_test_split(X2, y, test_size=0.3, random_state=random_state, stratify=y)

train=preprocessing.robust_scale(train, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
#
test=preprocessing.robust_scale(test, axis=0, with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)



# In[14]:

learning_rate=0.1


#tun_range1 = {'n_estimators':range(20,81,10)}
#{'n_estimators': 40} - 0.782280755933


n_estimators=40

#tun_range2 = {'max_depth':range(1,7,1), 'min_samples_split':range(350,451,30)}
#{'min_samples_split': 380, 'max_depth': 6} - 0.782239139011

max_depth=5


#tun_range3 = {'min_samples_split':range(400,501,10), 'min_samples_leaf':range(13,18,1)}
#{'min_samples_split': 430, 'min_samples_leaf': 16}  - 0.782614790876

min_samples_split=430
min_samples_leaf=16

#tun_range4 = {'max_features':range(1,raw_data.shape[1]-1,1)}
#{'max_features': 4} - 0.782614790876

max_features=4

#tun_range5 = {'subsample':[0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79,0.80,0.81,0.82,0.83,0.84,0.85,0.86,0.87,0.88,0.89]}
#{'subsample': 0.8} - 0.782614790876

subsample=0.8


# In[15]:

tuned_est=GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                  max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_features=max_features,
                                  subsample=subsample,random_state=random_state)


# In[16]:

from sklearn import model_selection, metrics


# In[17]:

#Fit the algorithm on the data
tuned_est.fit(raw_data, raw_target )
#Predict training set:
tuned_predictions = tuned_est.predict(raw_data)
tuned_predprob = tuned_est.predict_proba(raw_data)[:,1]
#Perform cross-validation:
cv_score=model_selection.cross_val_score(tuned_est,raw_data,raw_target, cv=5, scoring='roc_auc')
#Print model report:
print "\nModel Report"
print "Accuracy : %.4g" % metrics.accuracy_score(raw_target, tuned_predictions)
print "AUC Score (Train): %f" % metrics.roc_auc_score(raw_target, tuned_predprob)    
print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score))


# In[18]:

def Learning(parX,pary):
    est=GradientBoostingClassifier(learning_rate=learning_rate,n_estimators=n_estimators, 
                                  max_depth=max_depth, min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,max_features=max_features,
                                  subsample=subsample,random_state=random_state)
    est.fit(parX,pary)
    return est
    


# In[19]:

def ROCPair(est1, lab1, est2, lab2, ylow_lim):
    predscur = est1.predict_proba(test)[:,1]

    fpr1, tpr1, thresh = metrics.roc_curve(test_t, predscur)
    auc = metrics.roc_auc_score(test_t, predscur)
    plt.figure(0).clf()
    plt.plot(fpr1,tpr1,label=lab1+str(auc))

    predscur= est2.predict_proba(test)[:,1]

    fpr2, tpr2, thresh = metrics.roc_curve(test_t, predscur)
    auc = metrics.roc_auc_score(test_t, predscur)
    plt.plot(fpr2,tpr2,label=lab2+str(auc))
    plt.ylim( ylow_lim, 1)
    plt.legend(loc=0)
    plt.show()


# In[ ]:




# In[37]:

learning_rate=0.05
est_0_05=Learning(train,train_t)


# In[ ]:




# In[45]:

learning_rate=0.5
n_estimators=5 #0.782837586989
est_0_5=Learning(train,train_t)


# In[46]:

learning_rate=0.05
n_estimators=150 #0.782837586989
est_0_05=Learning(train,train_t)


# In[47]:


n_estimators=330
learning_rate=0.01
est_0_01=Learning(train,train_t)


# In[48]:

learning_rate=0.005
n_estimators=601 #
est_0_005=Learning(train,train_t)


# In[50]:

ROCPair(est_0_5, "ROC 0.5, auc=", est_0_005, "ROC 0.005, auc=",0.0)


# In[51]:

ROCPair(est_0_5, "ROC 0.5, auc=", est_0_005, "ROC 0.005, auc=",0.6)


# In[36]:

n_estimators=150
learning_rate=0.5
est_0_5=Learning(train,train_t)
learning_rate=0.1
est_0_1=Learning(train,train_t)
learning_rate=0.05
est_0_05=Learning(train,train_t)


# In[28]:

models = [('Kezdeti', est_0_5),
          ('Hangolt', est_0_1),
          ('Shrinking:0.05', est_0_05),
         ]
stage_preds = {}
for mname, m in models:
    m.fit(train, train_t)
    stage_preds[mname] = {'train': list(m.staged_predict_proba(train)),  'test': list(m.staged_predict_proba(test))}
    #final_preds[mname] = {'train': m.predict_proba(train),  'test': m.predict_proba(test)}
 
plt.figure(figsize=(12,6))
for marker, (mname, preds) in zip(["-", "--", ":"], stage_preds.iteritems()):
    for c, (tt_set, target) in zip(['#ff4444', '#4444ff'], [('train', train_t), ('test', test_t)]):
        aucs = map(lambda x: roc_auc_score(target, x[:,1]), preds[tt_set])
        label = "%s: %s" % (mname, tt_set) + (" (legjobb: %.3f @ fa: %d)" % (max(aucs), np.array(aucs).argmax()+1) if tt_set == 'test' else "")
        plt.plot(aucs, marker, c=c, label=label)
plt.ylim(0.7, 0.9)
plt.title(u'A ROC görbe alatti terület(AUC) a gyenge tanulók számának függvényében')
plt.xlabel("Fa #")
plt.ylabel("AUC")
plt.legend(loc="lower center")
plt.show()


# In[ ]:




# In[29]:

n_estimators=1900
learning_rate=0.01
est_0_01=Learning(raw_data,raw_target)
learning_rate=0.005
est_0_005=Learning(raw_data,raw_target)
max_depth=6
min_samples_split=450
min_samples_leaf=14
max_features='sqrt'
subsample=0.78
learning_rate=0.5
est_0_5=Learning(raw_data,raw_target)


# In[30]:

models = [('Kezdeti hangolatlan', est_0_5),
          ('Shrinking 0.01', est_0_01),
          ('Shrinking 0.005', est_0_005),
         ]
stage_preds = {}
for mname, m in models:
    m.fit(train, train_t)
    stage_preds[mname] = {'train': list(m.staged_predict_proba(train)),  'test': list(m.staged_predict_proba(test))}
    #final_preds[mname] = {'train': m.predict_proba(train),  'test': m.predict_proba(test)}
 
plt.figure(figsize=(12,6))
for marker, (mname, preds) in zip(["-", "--", ":"], stage_preds.iteritems()):
    for c, (tt_set, target) in zip(['#ff4444', '#4444ff'], [('train', train_t), ('test', test_t)]):
        aucs = map(lambda x: roc_auc_score(target, x[:,1]), preds[tt_set])
        label = "%s: %s" % (mname, tt_set) + (" (Legjobb: %.3f @ fa %d)" % (max(aucs), np.array(aucs).argmax()+1) if tt_set == 'test' else "")
        plt.plot(aucs, marker, c=c, label=label)
plt.ylim(0.6, 1.0)
plt.title(u'A ROC görbe alatti terület(AUC) a gyenge tanulók számának függvényében')
plt.xlabel("Fa #")
plt.ylabel("AUC")
plt.legend(loc="lower center")
plt.show()


# In[35]:

print(metrics.roc_curve(test_t,est_0_5.predict_proba(test)[:,1]))


# In[ ]:



