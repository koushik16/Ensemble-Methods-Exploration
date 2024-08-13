# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
# %matplotlib inline

"""**Function to calculate error**"""

def err(y,yp):
    # error function to compute the testing and training errors
    N = len(y)
    tem = np.round(yp)
    add = np.sum(y!=tem)
    er = add/N
    return er

"""**Bagging Method**"""

class bagging_tree_classif:
    #Class for the Bagging ensember with fit and predict methords
    def bag_fit(self,x,y,n,m):
        self.m_list=list()
        for iedx in range(n):
            l=len(x)
            samp=np.random.choice(np.arange(l),size=l,replace=True)
            t_set,Test_set=x[samp],y[samp]
            DT=DecisionTreeClassifier(max_depth=m)
            DT.fit(t_set,Test_set)
            self.m_list.append(DT)

    def predic(self,X,y,n):
        ll = len(X)
        scores_ba,te=np.zeros(ll),list()
        for dt in self.m_list:
            p=dt.predict(X)
            temp = err(y,p)
            scores_ba=scores_ba+dt.predict(X)
            te.append(temp)
            ret = scores_ba/n
            re = np.round(ret)
        return re,te

"""**Boosting: Using AdaBoost**"""

class adaboost_ensemble:
    def fit(self, X, y,N,m):
    #Class for the Ada Boost ensember with fit and predict methords
        we_coms,self.mod,tr_e = list(),list(),list()
        ll = len(y)
        var_we = np.ones(ll)/ ll
        for t in range(0, N):
            tree = DecisionTreeClassifier(max_depth=m)
            tree.fit(X,y,sample_weight = var_we)
            sc = tree.predict(X)
            self.mod.append(tree)
            error = err(y,sc)
            tr_e.append(error)
            ada_wec = 1/2*np.log((1 - error) / (error+1e-10))
            var_we = var_we * np.exp(-ada_wec * (y!=sc).astype(int))
            we_coms.append(ada_wec)
        return we_coms
    
    def predic(self,X,y,N,we_coms):
        scores_ada,te_e,re = list(),list(),list()
        for t in range(N):
            sc = self.mod[t].predict(X)
            y_pred_t = sc*we_coms[t]
            sc = np.sign(y_pred_t)
            error_test = np.sum(y!=sc)/len(y)
            te_e.append(error_test)
            scores_ada.append(y_pred_t)
        scores_ada = np.array(scores_ada).T
        tem2 = np.sum(scores_ada,axis=1)
        re = np.sign(tem2)
        re = np.array(re)
        
        return re,te_e

"""**Function for model fit, predict and plot results**"""

def graphs(X_train, X_test, y_train, y_test):
    #this function is the part where we do the bagging and ada-boost on the decition tree classifier along with finding errors for test and train 
    #process and plots for the error rate.

    lis11=[5,10]#list for max_depth
    for md in lis11:
        list1=[200]#list for number of rounds
        for pos in list1:
            g_val = pos+10
            g_list =[range(g_val)]
            plt.xlabel('Number of Rounds')
            plt.ylabel('Error in Testing')
            ada = adaboost_ensemble()#defining the ada_boost class
            aa = ada.fit(X_train,y_train,pos+10,md)#fitting the model
            t1,t2 = ada.predic(X_train,y_train,pos+10,aa)#prediction
            ada_trainerr = err(y_train,t1)#finding the training error
            ada_pred,ada_te = ada.predic(X_test,y_test,pos+10,aa)
            ada_finalerror = err(y_test,ada_pred)#finding the test error
            print("Final test error for Ada boosting with depth is",ada_finalerror,"with a depth of",md)
            #plotting graph for error rate while testing and number of rounds for ada boost for all the max_depth values
            plt.plot(list(range(pos+10)), ada_te, label='Ada Boosting error plot',color='black')

            bag = bagging_tree_classif()#defining the bagging class
            bag.bag_fit(X_train,y_train,pos+10,md)#fitting the model
            be,te = bag.predic(X_train,y_train,pos+10)#prediction
            bad_trainerr = err(y_train,be)#finding the training error
            bag_ypred,bag_te = bag.predic(X_test,y_test,pos+10)
            bag_finalerror = err(y_test,bag_ypred)#finding the test error
            print("Final test error for Bagging with depth:",bag_finalerror,"with a depth of",md)
            #plotting graph for error rate while testing and number of rounds for Bagging for all the max_depth values
            plt.plot(list(range(pos+10)), bag_te, label='Bagging error plot',color='green')
            
            plt.legend()
            plt.show()

"""**Ada-Boost and Bagging on Letter Recognition data**"""

df = pd.read_csv('letter-recognition.data',header=None)
c_vals = (df.iloc[:,0]=='C')
g_vals = (df.iloc[:,0]=='G')
df=df[c_vals|g_vals]
df[0]=df[0].astype('category').cat.codes
data = df.iloc[:,1:].values
lab = df.iloc[:,0].values
X_train, X_test, y_train, y_test = train_test_split(data,lab,test_size=0.3, random_state = 17)
graphs(X_train, X_test, y_train, y_test)

"""**Ada-Boost and Bagging on spambase data**"""

df = pd.read_csv('spambase.data', header = None)
cns = pd.read_csv('spambase.names', sep = ':', skiprows=range(0, 33), header = None)
var_heads = list(cns[0])
var_heads.append('Spam')
df.columns = var_heads
w = 2-1
df['Spam'] = df['Spam'] *w
v1 = df.drop(columns = 'Spam').values
v2 = df['Spam'].values
X_train, X_test, y_train, y_test = train_test_split(v1,v2,train_size = 0.3, random_state = 17) 
graphs(X_train, X_test, y_train, y_test)

"""**Ada-Boost and Bagging on German credit data**"""

df = pd.read_csv('german.data',sep=' ',header = None)
df = df.apply(lambda indx: LabelEncoder().fit_transform(indx))
v1,v2= df.iloc[:,:20].values,LabelEncoder().fit_transform(df[20])
X_train, X_test, y_train, y_test = train_test_split(v1,v2,test_size=0.3,random_state=17)
graphs(X_train, X_test, y_train, y_test)

