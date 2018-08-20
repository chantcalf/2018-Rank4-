# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:10:37 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
from sklearn import metrics
import os
import warnings
warnings.filterwarnings("ignore")
from utils import docount,domean,dovar,doiq,domin,domax
from utils import is_active,showresults,predict_data,getbest,showtop,predict_data_val,showfalse,getbest1
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn import preprocessing

val = 1
get_author_feature = 0
get_weekend_feature = 0
path = '../data/'

register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')

register = register[(register['register_day']!=24) | (register['device_type']!=1)]
register = register[(register['register_day']!=24) | (register['device_type']!=83)]
register = register[(register['register_day']!=24) | (register['device_type']!=223)]
#if get_author_feature:
author_info = pd.read_csv('author_info.csv')
act = act.merge(author_info,on=['author_id'],how='left')

#if get_weekend_feature:
def isweekend(x):
    if x in [6,7,13,14,21,22,27,28]:
        return 1
    return 0
    
app['weekend'] = app['day'].apply(isweekend)
video['weekend'] = video['day'].apply(isweekend)
act['weekend'] = act['day'].apply(isweekend)

appadd = []
for i in range(1,31):
    actu = list(act[act.day==i]['user_id'].unique())
    videou = list(video[video.day==i]['user_id'].unique())
    appu = list(app[app.day==i]['user_id'].unique())
    for c in videou:
        if c not in actu:
            actu.append(c)
    for c in actu:
        if c not in appu:
            d={'user_id':c,'day':i}
            appadd.append(d)
appadd = pd.DataFrame(appadd)
app = app.append(appadd)


def get_features(df,ed):
    df['register_time'] = ed-df.register_day+1
    del df['register_day']
    tapp = app[app.day<=ed]
    tact = act[act.day<=ed]
    tvideo = video[video.day<=ed]
    tapp['time'] = ed-tapp.day
    tact['time'] = ed-tact.day
    tvideo['time'] = ed-tvideo.day
    
    
    if get_author_feature:
        gp = tact[(tact.time<14)&(tact.ranks<=500)].groupby(['user_id','ranks']).size().unstack().reset_index()
        cols = list(gp.columns)
        for i in range(1,501):
            if i not in cols:
                print (i)
                gp[i]=0
        for i in range(1,501):
            gp['a'+str(i)] = gp[i]
            del gp[i]
        df = df.merge(gp,on=['user_id'],how='left')
        
    
    df = docount(df,tapp,'app',['user_id'])
    df = domin(df,tapp,'app',['user_id'],'day')
    df = dovar(df,tapp,'app',['user_id'],'day')
    #df = domax(df,tapp,'app',['user_id'],'day')
    #df['app_day_gap'] = df['app$user_id_by_day_max']- df['app$user_id_by_day_min']
    df['app_rate'] = df['app$user_id#']/df['register_time']
    
    df = docount(df,tvideo,'video',['user_id'])
    df = domin(df,tvideo,'video',['user_id'],'day')
    df = doiq(df,tvideo,'video',['user_id'],'day')
    df = doiq(df,tvideo[tvideo.time<16],'video16',['user_id'],'day')

    df['video_rate'] = df['video$user_id_by_day_iq']/df['register_time']
    df['video_rate1'] = df['video$user_id_by_day_iq']/df['app$user_id#']
    df['video_mean'] = df['video$user_id#']/df['register_time']
    df['video_mean1'] = df['video$user_id#']/df['app$user_id#']
    df['video_mean2'] = df['video$user_id#']/df['video$user_id_by_day_iq']
    
    df = docount(df,tact,'act',['user_id'])
    df = domin(df,tact,'act',['user_id'],'day')
    df = doiq(df,tact,'act',['user_id'],'day')
    df = doiq(df,tact[tact.time<16],'act16',['user_id'],'day')
    df['act_rate'] = df['act$user_id_by_day_iq']/df['register_time']
    df['act_rate1'] = df['act$user_id_by_day_iq']/df['act$user_id#']
    df['act_mean'] = df['act$user_id#']/df['register_time']
    df['act_mean1'] = df['act$user_id#']/df['app$user_id#']
    df['act_mean2'] = df['act$user_id#']/df['act$user_id_by_day_iq']
    
    
    #df = docount(df,tapp[(tapp.time<14)&(tapp.weekend==1)],'app14_weekend',['user_id'])
    df = docount(df,tapp[(tapp.time<7)&(tapp.weekend==1)],'app7_weekend',['user_id'])
    
    #df = docount(df,tvideo[(tvideo.time<14)&(tvideo.weekend==1)],'video14_weekend',['user_id'])
    #df = docount(df,tvideo[(tvideo.time<7)&(tvideo.weekend==1)],'video7_weekend',['user_id'])
    
    #df = docount(df,tact[(tact.time<14)&(tact.weekend==1)],'act14_weekend',['user_id'])
    #df = docount(df,tact[(tact.time<7)&(tact.weekend==1)],'act7_weekend',['user_id'])
    
    #df = doiq(df,tact[(tact.time<16)&(tact.ranks<50)],'act16_top50',['user_id'],'author_id')
    #df = doiq(df,tact[(tact.time<16)&(tact.ranks<100)],'act16_top100',['user_id'],'author_id')
    df = doiq(df,tact[(tact.time<16)&(tact.ranks<500)],'act16_top500',['user_id'],'author_id')
    df = doiq(df,tact[(tact.time<16)&(tact.ranks<500)],'act16_top500',['user_id'],'video_id')
    #df = docount(df,tact[(tact.time<16)&(tact.ranks<10)],'act16_top10',['user_id'])
    #df = docount(df,tact[(tact.time<16)&(tact.ranks<100)],'act16_top100',['user_id'])
    #df = docount(df,tact[(tact.time<16)&(tact.ranks<500)],'act16_top500',['user_id'])
    
    for i in range(2,7):
        gp = tapp[tapp.time<=i][['user_id']].groupby(['user_id']).size().rename('last_'+str(i)+'_days_app#').reset_index()
        df = pd.merge(df,gp,on=['user_id'],how='left')
    
    for i in range(2,7):
        gp = tact[tact.time<=i][['user_id']].groupby(['user_id']).size().rename('last_'+str(i)+'_days_act#').reset_index()
        df = pd.merge(df,gp,on=['user_id'],how='left')
    
    gp = tact[['user_id','author_id']].groupby(['user_id'])['author_id'].nunique().rename('act_author_id_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_author_id_u_mean#'] = df['act_author_id_u#']/df['register_time']  
    gp = tact[['user_id','video_id']].groupby(['user_id'])['video_id'].nunique().rename('act_video_id_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    df['act_video_id_u_mean#'] = df['act_video_id_u#']/df['register_time']


    df['video_author_m'] = df['act_video_id_u#']/df['act_author_id_u#']
    df['act_author_id_u_mean1#'] = df['act_author_id_u#']/df['act$user_id_by_day_iq']
    df['act_video_id_u_mean1#'] = df['act_video_id_u#']/df['act$user_id_by_day_iq']
    
    for i in [3,7,14]:
        for c in [0,1,2,3]: 
            gp = tact[tact['time']<i][tact['page']==c][['user_id']].groupby(['user_id']).size().rename('act_'+str(i)+'_author_page_'+str(c)+'_u#').reset_index()
            df = pd.merge(df,gp,on=['user_id'],how='left')
            
    for i in [3,7,14]:
        for c in [0,1,2,3]: 
            gp = tact[tact['time']<i][tact['action_type']==c][['user_id']].groupby(['user_id']).size().rename('act_'+str(i)+'_author_action_type_'+str(c)+'_u#').reset_index()
            df = pd.merge(df,gp,on=['user_id'],how='left')
    
    def get_last_gap(s):
        s = list(s)
        n = len(s)
        if n>1:
            s.sort()
            return s[n-1]-s[n-2]
        return None
        
    gp = tapp[tapp['time']<16].groupby(['user_id'])['day'].unique().apply(get_last_gap).rename('app_last_gap').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tvideo[tvideo['time']<16].groupby(['user_id'])['day'].unique().apply(get_last_gap).rename('video_last_gap').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left') 
    
    gp = tact[tact['time']<16].groupby(['user_id'])['day'].unique().apply(get_last_gap).rename('act_last_gap').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left') 

    df['author_id'] = df['user_id']
    gp = tact[tact['time']<16][['author_id']].groupby(['author_id']).size().rename('author#').reset_index()
    df = pd.merge(df,gp,on=['author_id'],how='left')
    
    for i in [7,14]:
        for c in [0,1,2,3]: 
            gp = tact[tact['time']<i][tact['action_type']==c][['author_id']].groupby(['author_id']).size().rename('act_'+str(i)+'_author_action_type_'+str(c)+'_a#').reset_index()
            df = pd.merge(df,gp,on=['author_id'],how='left')
    
    for c in [1,2,3,4]: 
        gp = tact[tact['time']<16][tact['page']==c][['author_id']].groupby(['author_id']).size().rename('author_act_page_'+str(c)+'#').reset_index()
        df = pd.merge(df,gp,on=['author_id'],how='left')
    
    del df['author_id']     
    
    del df['app$user_id#'],df['video$user_id#'],df['act$user_id#'],df['act_author_id_u#'],df['act_video_id_u#']

    del df['act$user_id_by_day_iq'],df['video$user_id_by_day_iq']
    
    for i in range(16):
        gp = tapp[tapp.time==i].groupby(['user_id']).size().rename('app_'+str(i)).reset_index()
        df = df.merge(gp,on=['user_id'],how = 'left')
    
    for i in range(16):
        gp = tvideo[tvideo.time==i].groupby(['user_id']).size().rename('video_count_'+str(i)).reset_index()
        df = df.merge(gp,on=['user_id'],how = 'left')
    
    for i in range(16):
        gp = tact[tact.time==i].groupby(['user_id']).size().rename('act_count_'+str(i)).reset_index()
        df = df.merge(gp,on=['user_id'],how = 'left')
        
    
    return df

def get_Ridge(train_df,train_y,test_df):
    #train_df = train_df.applymap(lambda x:np.log(x+1))
    #test_df = test_df.applymap(lambda x:np.log(x+1))
    clf = Ridge(alpha=0.001)
    clf.fit(train_df,train_y)
    a = clf.coef_
    for i in range(len(a)):
        if a[i]<0:
            a[i]=0
    clf.coef_ = a
    #print (clf.coef_)
    return clf.predict(train_df),clf.predict(test_df)
    
def get_PCA(df,nn=2,name=''):
    #scl = preprocessing.StandardScaler()
    #df = scl.fit_transform(df)
    pca = PCA(n_components=nn,random_state=2018)
    df = pca.fit_transform(df)
    cc = []
    for i in range(nn):
        cc.append(name+'_PCA'+str(i))   
    df = pd.DataFrame(df,columns=cc)
    
    return df
    
def get_tSVD(df,nn=2,name=''):
    tsvd = TruncatedSVD(n_components=nn, random_state=2018)
    df = tsvd.fit_transform(df)
    cc = []
    for i in range(nn):
        cc.append(name+'_tSVD'+str(i))   
    df = pd.DataFrame(df,columns=cc)
    
    return df

def isnotzero(x):
    if x>0:
        return 1
    return 0

def get_features1(train_df,test_df,train_y):
    #supervised
    cols = []
    for i in range(16):
        cols.append('app_'+str(i))
    
    temp_train_df = train_df[cols]  
    temp_test_df = test_df[cols]
    
    temp_train_df = temp_train_df.fillna(0)
    temp_test_df = temp_test_df.fillna(0)    
    for i in range(4,17):
        train_df['Rapp_'+str(i)],test_df['Rapp_'+str(i)] = get_Ridge(temp_train_df[cols[:i]],train_y,temp_test_df[cols[:i]])
   
        
    cols = []
    for i in range(16):
        cols.append('video_count_'+str(i))
    
    temp_train_df = train_df[cols]  
    temp_test_df = test_df[cols]
    
    temp_train_df = temp_train_df.fillna(0)
    temp_test_df = temp_test_df.fillna(0)    
    for i in range(2,17):
        train_df['Rvideo_count_'+str(i)],test_df['Rvideo_count_'+str(i)] = get_Ridge(temp_train_df[cols[:i]],train_y,temp_test_df[cols[:i]])
        
    cols = []
    for i in range(16):
        cols.append('act_count_'+str(i))
    
    temp_train_df = train_df[cols]  
    temp_test_df = test_df[cols]
    
    train_df['act_day_mean'] = temp_train_df.mean(1)
    test_df['act_day_mean'] = temp_test_df.mean(1)
    
    train_df['act_day_var'] = temp_train_df.var(1)
    test_df['act_day_var'] = temp_test_df.var(1)
    
    temp_train_df = temp_train_df.fillna(0)
    temp_test_df = temp_test_df.fillna(0)    
    for i in range(2,17):
        train_df['Ract_count_'+str(i)],test_df['Ract_count_'+str(i)] = get_Ridge(temp_train_df[cols[:i]],train_y,temp_test_df[cols[:i]])
    
    
    temp_train_df1 = temp_train_df.applymap(isnotzero)
    temp_test_df1 = temp_test_df.applymap(isnotzero)  
    
    for i in range(3,14):
        train_df['Ract_'+str(i)],test_df['Ract_'+str(i)] = get_Ridge(temp_train_df1[cols[:i]],train_y,temp_test_df1[cols[:i]])

    '''
    if get_author_feature:
        cols = []
        for i in range(1,501):
            cols.append('a'+str(i))

        temp_train_df = train_df[cols]  
        temp_test_df = test_df[cols]
        temp_train_df = temp_train_df.fillna(0)
        temp_test_df = temp_test_df.fillna(0)
        train_df['Ra500'],test_df['Ra500'] = get_Ridge(temp_train_df,train_y,temp_test_df)
    '''
    #unsupervised
    lendf = len(train_df)  
    train_df= train_df.append(test_df)
    del test_df
    gc.collect()
    kmeans = pd.read_csv('kmeans.csv')
    train_df = train_df.merge(kmeans,on=['device_type'],how='left')
    '''
    if get_author_feature:
        cols = []
        for i in range(1,501):
            cols.append('a'+str(i))

        temp_train_df = train_df[cols] 
        temp_train_df = temp_train_df.fillna(0)
        from sklearn.feature_extraction.text import TfidfTransformer 
        transformer=TfidfTransformer()
        tfidf = transformer.fit_transform(temp_train_df)
        tfidf = tfidf.toarray()
        tdf = get_PCA(tfidf,nn=50,name='Author_TFIDF')
        for c in tdf.columns:
            train_df[c] = tdf[c]

        train_df.drop(columns=cols,inplace=True)
    '''
    
    cols = []
    for i in range(16):
        cols.append('app_'+str(i))
      
    train_df.drop(columns=cols,inplace=True)
    
    cols = []
    for i in range(16):
        cols.append('video_count_'+str(i))
    
    train_df.drop(columns=cols,inplace=True)    
    cols = []
    for i in range(16):
        cols.append('act_count_'+str(i))  
          
        
    train_df.drop(columns=cols,inplace=True)
    
    
 
    test_df = train_df[lendf:]
    train_df = train_df[:lendf]
    test_df = test_df.reset_index(drop=True)
    return train_df,test_df
    
def get_features_all(df,df1):
    lendf = len(df)
    
    df= df.append(df1)
    del df1
    gc.collect()
    #df = docount(df,df,'All',['register_type']);
    #df = docount(df,df,'All',['device_type']);
    
    #del df['device_type']
    '''
    if get_author_feature:
        cols = []
        for i in range(1,501):
            cols.append('a'+str(i))

        temp_train_df = df[cols] 
        temp_train_df = temp_train_df.fillna(0)
        from sklearn.feature_extraction.text import TfidfTransformer 
        transformer=TfidfTransformer()
        tfidf = transformer.fit_transform(temp_train_df)
        tfidf = tfidf.toarray()
        tdf = get_PCA(tfidf,nn=20,name='Author_TFIDF')
        for c in tdf.columns:
            df[c] = tdf[c]

        df.drop(columns=cols,inplace=True)
    '''
    
    def maxth(s):
        if s>16:
            return 16
        return s
 
    df['register_time'] = df['register_time'].apply(maxth)    
    
    del df['user_id'],df['app7_weekend$user_id#'],df['act16_top500$user_id_by_video_id_iq']
    del df['act16_top500$user_id_by_author_id_iq']


    #cols = cols+cols1     
    #df.drop(columns=cols,inplace=True)
    df1 = df[lendf:]
    df = df[:lendf]
    return df,df1    

if val:
    if os.path.exists(path+'val_df.csv'):
        test_df = pd.read_csv(path+'val_df.csv')
        val_y = pd.read_csv(path+'val_y.csv')
    else:
        test_df = register[register.register_day<=21]
        test_df = get_features(test_df,23)
        val_y = is_active(test_df,24,30,app,video,act)
        test_df.to_csv(path+'val_df.csv',index=False)
        val_y.to_csv(path+'val_y.csv',index=False)
    val_y = val_y['Y']
    if os.path.exists(path+'val_train_df.csv'):
        train_df = pd.read_csv(path+'val_train_df.csv')
        train_y = pd.read_csv(path+'val_train_y.csv')
    else:    
        train_df = pd.DataFrame()   
        train_y = pd.DataFrame()                  
        for i in [16]:
            df = register[register.register_day<=i-2]
            y = is_active(df,i+1,i+7,app,video,act)
            df = get_features(df,i)
            train_df = train_df.append(df)
            train_y = train_y.append(y)
        train_df.to_csv(path+'val_train_df.csv',index=False)
        train_y.to_csv(path+'val_train_y.csv',index=False)
else:
    if os.path.exists(path+'test_df.csv'):
        test_df = pd.read_csv(path+'test_df.csv')
    else:
        test_df = register[register.register_day<=28]
        test_df = get_features(test_df,30)
        test_df.to_csv(path+'test_df.csv',index=False)
                           
    if os.path.exists(path+'train_df.csv'):
        train_df = pd.read_csv(path+'train_df.csv')
        train_y = pd.read_csv(path+'train_y.csv')
    else:            
        if os.path.exists(path+'val_train_df.csv'):
            train_df = pd.read_csv(path+'val_train_df.csv')
            train_y = pd.read_csv(path+'val_train_y.csv') 
            val_df = pd.read_csv(path+'val_train_df.csv')
            val_y = pd.read_csv(path+'val_train_y.csv')
            train_df = train_df.append(val_df)
            train_y = train_y.append(val_y)  
            for i in []:
                df = register[register.register_day<=i-2]
                y = is_active(df,i+1,i+7,app,video,act)
                df = get_features(df,i)
                train_df = train_df.append(df)
                train_y = train_y.append(y)  
        else:
            train_df = pd.DataFrame()   
            train_y = pd.DataFrame()                  
            for i in [16,23]:
                df = register[register.register_day<=i]
                y = is_active(df,i+1,i+7,app,video,act)
                df = get_features(df,i)
                train_df = train_df.append(df)
                train_y = train_y.append(y)  
        train_df.to_csv(path+'train_df.csv',index=False)
        train_y.to_csv(path+'train_y.csv',index=False)                 
train_y = train_y['Y']



if val:
    if os.path.exists(path+'val_df1.csv'):
        test_df = pd.read_csv(path+'val_df1.csv')
        train_df = pd.read_csv(path+'val_train_df1.csv')
    else:
        train_df,test_df = get_features1(train_df,test_df,train_y) 
        train_df.to_csv(path+'val_train_df1.csv',index=False)
        test_df.to_csv(path+'val_df1.csv',index=False)
else:
    if os.path.exists(path+'test_df1.csv'):
        test_df = pd.read_csv(path+'test_df1.csv')
        train_df = pd.read_csv(path+'test_train_df1.csv')
    else:
        train_df,test_df = get_features1(train_df,test_df,train_y) 
        train_df.to_csv(path+'test_train_df1.csv',index=False)
        test_df.to_csv(path+'test_df1.csv',index=False)

ids = test_df['user_id']
train_df,test_df = get_features_all(train_df,test_df) 
cfl = ['device_type', 'kmeans', 'register_type','register_time']
if val:
    pre_train,test_y = predict_data_val(train_df,train_y,10,test_df,val_y,importance=1,cf_list=cfl)
    #pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=1,loss = 1,nb=56)
else:
    pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=0,loss = 1,nb=99)

sp = 1

if val==1:   
    showresults(val_y,test_y) 
    #showtop(val_y,test_y,nums=18223)
    showtop(val_y,test_y,nums=15428)
    showfalse(ids,test_df,val_y,test_y)
else:
    showresults(train_y,pre_train) 
    if sp:
        df_1_28 = register[register.register_day<=28]
        #df_29_30 = register[register.register_day>28]
        ans_1_28 = getbest1(df_1_28,ids,test_y,rank=22088)
        #ans_29_30 = getbest1(df_29_30,ids,test_y,th=0.4)
        #print (len(ans_1_28),len(ans_29_30))
        from predict_30 import predict_30
        from predict_29 import predict_29
        ids29,test_y29,ans29 = predict_29(val,register,app,video,act)
        ids30,test_y30,ans30 = predict_30(val,register,app,video,act)
        ans = ans_1_28+ans29+ans30
        
    else:
        ans = getbest(ids,test_y,rank=22088) 
    print(len(ans))
    import time
    name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
    submission = pd.DataFrame({'user_id': ans})
    submission.to_csv('ksn_submit'+name+'.csv', index=False, header = None)
