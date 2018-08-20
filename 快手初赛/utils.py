# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:59:41 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
from sklearn import metrics
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt

def getscore():
    N = 23727
    M = 24872
    F1 =  0.821334 
    x = F1*(M+N)/2
    p = x/M
    r = x/N
    print (x,p,r)
    
def getscore1():
    #N = 1133+1169   #23727#23727 #2303 #23727 #21424 #23727
    #M = 1294+1490   #24676#22088+2784 #2784 #24872 #22088 #48100
    #x = 986+1086    #19846#17889+2068 #2068 #19957 #17889
    N = 21425#+2302
    M = 22088#+2784
    x = 17886#+2072   
    p = x/M
    r = x/N
    F1 = 2*p*r/(p+r)
    print (F1,p,r)

def docount(df,tf,name,group):
    name = name+'$'+'_'.join(group)+'#'
    gp = tf[group].groupby(group).size().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left')
    df[name] = df[name].fillna(0)
    del gp
    return df
    
def domean(df,tf,name,group,count):
    name = name+'$'+'_'.join(group)+'_by_'+count+'_mean'
    gp = tf[group+[count]].groupby(group)[count].mean().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left') 
    #df[name] = df[name].fillna(0)
    del gp
    return df
    
def dovar(df,tf,name,group,count):
    name = name+'$'+'_'.join(group)+'_by_'+count+'_var'
    gp = tf[group+[count]].groupby(group)[count].var().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left') 
    #df[name] = df[name].fillna(0)
    del gp
    return df

def doiq(df,tf,name,group,count):
    name = name+'$'+'_'.join(group)+'_by_'+count+'_iq'
    gp = tf[group+[count]].groupby(group)[count].nunique().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left') 
    #df[name] = df[name].fillna(0)
    del gp
    return df
    
def domin(df,tf,name,group,count):
    name = name+'$'+'_'.join(group)+'_by_'+count+'_min'
    gp = tf[group+[count]].groupby(group)[count].min().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left') 
    #df[name] = df[name].fillna(0)
    del gp
    return df
    
def domax(df,tf,name,group,count):
    name = name+'$'+'_'.join(group)+'_by_'+count+'_max'
    gp = tf[group+[count]].groupby(group)[count].max().rename(name).reset_index()
    df = pd.merge(df,gp,on=group,how='left') 
    #df[name] = df[name].fillna(0)
    del gp
    return df
    
def is_active(df,d1,d2,app,video,act):
    c1 = list(app[(app.day>=d1) & (app.day<=d2)]['user_id'].unique())
    c2 = list(act[(act.day>=d1) & (act.day<=d2)]['user_id'].unique())
    c3 = list(video[(video.day>=d1) & (video.day<=d2)]['user_id'].unique())
    c1 = c1+c2+c3
    def isinc(s):
        if s in c1:
            return 1
        else:
            return 0
    y = df['user_id'].apply(isinc)  
    y1 = pd.DataFrame(list(y),columns=['Y'])
    return y1    

def getF1(val_y,test_y,th):
    test_y1 = list(test_y)
    for i in range(len(test_y1)):
        if test_y1[i]>=th:
            test_y1[i]=1
        else:
            test_y1[i]=0
    F1score = metrics.f1_score(val_y, test_y1)
    num = sum(test_y1)
    return F1score,num,metrics.precision_score(val_y, test_y1),metrics.recall_score(val_y, test_y1)
    
    
def showtop(val_y,test_y,nums=100):
    test_y2 = list(test_y)
    test_y2.sort(reverse = True)
    th = test_y2[nums-1]
    print (getF1(val_y,test_y,th))
    
def showresults(val_y,test_y):
    print(sum(list(val_y)),len(test_y))
    mf = 0
    print(0.4,getF1(val_y,test_y,0.4))
    print(0.5,getF1(val_y,test_y,0.5))
    for th in np.arange(0.35,0.55,0.005):
        F1score,num,p,r = getF1(val_y,test_y,th)
        #print(th,sum(test_y1),
        #      'MSE = ',metrics.mean_squared_error(val_y, test_y1),
        #      'Precision = ',metrics.precision_score(val_y, test_y1),
        #      'Recall = ',metrics.recall_score(val_y, test_y1),
        #      'F1 = ',F1score)
        if F1score>mf:
            mf = F1score
            mth = th
            nums = num
            pp = p
            rr = r
    print (nums,mth,mf,pp,rr)   
    return mth
    
def showresults1(val_y,test_y):
    print(sum(list(val_y)),len(test_y))
    mf = 0
    test_y2 = list(test_y)
    test_y2.sort()
    for i in range(len(test_y)-19769,len(test_y)-19368):
        th = test_y2[i]
        test_y1 = list(test_y)
        for i in range(len(test_y1)):
            if test_y1[i]>=th:
                test_y1[i]=1
            else:
                test_y1[i]=0

        F1score = metrics.f1_score(val_y, test_y1)
        if F1score>mf:
            mf = F1score
            mth = th
            nums = sum(test_y1)
    print (nums,mth,mf)   
    return mth  
    
def getbest(ids,test_y,th=0.5,rank=-1):
    sub = pd.DataFrame({'user_id': ids,'Y':test_y})
    if rank<0:
        ans = sub[sub['Y']>=th]['user_id'].drop_duplicates()
    else:
        sub = sub.sort_values(by='Y',ascending = False).reset_index(drop = True)
        ans = sub[:rank]['user_id'].drop_duplicates()
        print(sub.loc[rank-1,'Y'])
    print (len(ans),len(test_y))
    return list(ans)
    
def showprecision(val_y,test_y):
    pre = pd.DataFrame({'val_y': val_y,'Y':test_y})
    pre = pre.sort_values(by='Y',ascending = False).reset_index(drop = True)
    pre['s'] = pre['val_y'].cumsum()
    xx = []
    yy = []
    yy1 = []
    zz = []
    for i in range(100,len(pre),100):
        xx.append(i)
        p0 = (pre.loc[i,'s']-pre.loc[i-100,'s'])/100
        p1 = pre.loc[i,'s']/i
        yy.append(p0)
        yy1.append(p1)
        zz.append(pre.loc[i,'Y'])
    yy2 = []
    n = len(xx)
    for i in range(n):
        j = i+10
        if j>n:
            j=n
        yy2.append(np.mean(yy[i:j]))
        
    for i in range(n):
        if yy2[i]<0.4:
            print (xx[i],yy1[i],zz[i])
            break
    plt.subplot(2,1,1)
    plt.plot(xx,yy,'.')
    plt.plot(xx,yy1,'o')
    plt.plot(xx,yy2,'*')
    plt.subplot(2,1,2)
    plt.plot(xx,zz,'.')
    
def showfalse(ids,test_df,val_y,test_y):
    test_df['id'] = ids
    test_df['y'] = test_y
    test_df['val_y'] = val_y
    fs = test_df[(test_df['val_y']==1) & (test_df['y']<0.5)]
    fs = fs.sort_values(by='y',ascending=False)
    fs.to_csv('../data1/false.csv')
    
    
params = {
              'boosting_type': 'gbdt',
              'objective': 'binary',
              'metric':'auc',
              'learning_rate': 0.09,
              'num_leaves': 13, 
              'max_depth': 4, 
              'min_child_samples': 1,  
              'max_bin': 90,  
              'subsample': 0.8,  
              'subsample_freq': 1,  
              'colsample_bytree': 0.9, 
              #'bagging_fraction':0.9,
              'min_child_weight': 0,  
              'min_split_gain': 0,  
              'nthread': 4,
              'verbose': 0,
              #'scale_pos_weight':55, 
              #'is_unbalance' : 'true',
             } 
             
def predict_data(train_df,train_y,fold_k,test_df,importance=0,importance_name = 'importance'):
    print ('training size: ', train_df.shape)
    print ('testing size: ', test_df.shape)
    #importance = 1
    #importance_name = 'importance'
    train_df = train_df.reset_index(drop = True)
    train_y = train_y.reset_index(drop = True)
    lgb_train = lgb.Dataset(train_df, train_y)
    bst=lgb.cv(params,lgb_train,num_boost_round=1000,nfold=10,early_stopping_rounds=30)
    nb = len(bst['auc-mean'])
    print(nb)
    
    a_lists = []
    b_lists = []
    train_len = train_df.shape[0]
    fold_size = int(train_len/fold_k)
    for i in range(fold_k-1):
        a_lists.append(fold_size*i)
        b_lists.append(fold_size*(i+1))
    a_lists.append(fold_size*(fold_k-1))
    b_lists.append(train_len)
      
    pre_test = []
    pre_train = []
    if importance: 
        train_features = train_df.columns.tolist()
        df = pd.DataFrame(train_features, columns=['feature'])
        df1 = pd.DataFrame(train_features, columns=['feature'])
        
    for i in range(fold_k):
        drops = list(range(a_lists[i],b_lists[i]))

        lgb_train = lgb.Dataset(train_df.drop(drops), train_y.drop(drops))
        #lgb_valid = lgb.Dataset(train_df[a_lists[i]:b_lists[i]], train_y[a_lists[i]:b_lists[i]])
        
        #print('Start training...','fold..',i) 
        
        #bst=lgb.cv(params,lgb_train,num_boost_round=1000,nfold=5,early_stopping_rounds=30)
        #print (len(bst['auc-mean']))
        
        model = lgb.train(params, 
                          lgb_train, 
                          #valid_sets=[lgb_train,lgb_valid], 
                          #valid_names=['train','valid'],
                          #evals_result = {},
                          #num_boost_round=len(bst['auc-mean']),
                          num_boost_round=nb,
                          #early_stopping_rounds = 50,
                          verbose_eval=50,
                          feval=None)
        #print('Start predicting...','fold..',i)
        prei = model.predict(train_df[a_lists[i]:b_lists[i]])
        pre_train = pre_train+ list(prei)
        pre_test.append(model.predict(test_df)) 
        if importance:
            df['importance'+str(i)]=list(model.feature_importance())  
    if importance:
        df1['importance'] = df.mean(1)    
        df1 = df1.sort_values(by='importance',ascending=False)    
        print (df1)
        df1.to_csv(importance_name+'.csv',index=False)
        print (list(df1['feature']))
    test_y = pre_test[0]
    for i in range(1,fold_k):
        test_y = test_y + pre_test[i]
    test_y = test_y/fold_k  
    test_y[test_y<0]=0
    test_y[test_y>1]=1    
    del train_df, test_df
    gc.collect()            
    return pre_train,test_y                      
    
if __name__=='__main__':
    getscore1()
    