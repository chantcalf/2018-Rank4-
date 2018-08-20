# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 14:51:46 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import os
from utils import docount,domean,dovar,doiq,domin,domax
from utils import is_active,showresults,predict_data,getbest,showtop,showprecision,showfalse
import warnings
warnings.filterwarnings("ignore")

def get_features_ks(df,d1,d2,app,video,act):
    tapp = app[(app.day>=d1) & (app.day<=d2)]
    tact = act[(act.day>=d1) & (act.day<=d2)]
    tvideo = video[(video.day>=d1) & (video.day<=d2)]
    tapp.day = tapp.day - d1
    tact.day = tact.day - d1
    tvideo.day = tvideo.day - d1
    lastday = d2-d1
    
    def get_last_gap(s):
        s = np.array(s)
        t = []
        for i in range(len(s)):
            if s[i]>0:
                t.append(i)
        n = len(t)
        if n>1:
            return t[n-1]-t[n-2]
        return None
        
    def get_max_continue_day(s):
        s = np.array(s)
        ans = 1
        t = 0
        for i in range(len(s)):
            if s[i]>0:
                t += 1
            else:
                if t>ans:
                    ans = t
                t = 0
        if t>ans:
            ans = t        
        return ans
    
        
    gp = tapp.groupby(['user_id','day']).size().unstack()
    tdf = pd.DataFrame()
    #tdf['app_last_gap'] = gp.apply(get_last_gap)
    #tdf['app_max_continue_day'] = gp.apply(get_max_continue_day)
    gp = gp.reset_index()
    tdf['user_id'] = gp['user_id']
    df = pd.merge(df,tdf,on=['user_id'],how='left')  
        
    gp = tapp[['user_id']].groupby(['user_id']).size().rename('app#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['app_mean#'] = df['app#']/df['register_time']
    
    for i in range(7):
        gp = tapp[tapp.day>=lastday-i][['user_id']].groupby(['user_id']).size().rename('last_'+str(i)+'_days_app#').reset_index()
        df = pd.merge(df,gp,on=['user_id'],how='left')
        
    #gp = tapp[['user_id','day']].groupby(['user_id'])['day'].nunique().rename('app_u#').reset_index()
    #df = pd.merge(df,gp,on=['user_id'],how='left')  
    #df['app_u_mean#'] = df['app_u#']/df['register_time']
    
    gp = tapp[['user_id','day']].groupby(['user_id'])['day'].min().rename('app_day_min').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tapp[['user_id','day']].groupby(['user_id'])['day'].max().rename('app_day_max').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tapp[['user_id','day']].groupby(['user_id'])['day'].mean().rename('app_day_mean').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tapp[['user_id','day']].groupby(['user_id'])['day'].var().rename('app_day_var').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['app_day_gap'] = df['app_day_max'] - df['app_day_min']
    #df['last_app_day'] = lastday - df['app_day_max']+1
    #del df['app_day_max']
    
    
    gp = tvideo.groupby(['user_id','day']).size().unstack()
    tdf = pd.DataFrame()
    #tdf['video_last_gap'] = gp.apply(get_last_gap)
    #tdf['video_max_continue_day'] = gp.apply(get_max_continue_day)
    gp = gp.reset_index()
    tdf['user_id'] = gp['user_id']
    df = pd.merge(df,tdf,on=['user_id'],how='left')  
    
    gp = tvideo[['user_id','day']].groupby(['user_id'])['day'].min().rename('video_day_min').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tvideo[['user_id','day']].groupby(['user_id'])['day'].max().rename('video_day_max').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tvideo[['user_id','day']].groupby(['user_id'])['day'].mean().rename('video_day_mean').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tvideo[['user_id','day']].groupby(['user_id'])['day'].var().rename('video_day_var').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tvideo[['user_id','day']].groupby(['user_id'])['day'].nunique().rename('video_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left') 
    
    df['video_u_mean#'] = df['video_u#']/df['register_time']
    
    df['video_day_gap'] = df['video_day_max'] - df['video_day_min']
    df['last_video_day'] = lastday - df['video_day_max']+1
    del df['video_day_max']

    #gp = tapp[['user_id']].groupby(['user_id']).size().rename('video#').reset_index()
    #df = pd.merge(df,gp,on=['user_id'],how='left') 
    #df['video_mean#'] = df['video#']/df['register_time']
    #df['video_mean1#'] = df['video#']/df['app#']
    #df['video_mean2#'] = df['video#']/df['video_day_iq']

    #df = docount(df,tvideo[tvideo.day>lastday-2],'video_last_2',['user_id'])
    #df = docount(df,tvideo[tvideo.day>lastday-3],'video_last_3',['user_id'])
    
    def get_actcount_mean(s):
        s = np.array(s)
        t = []
        for i in s:
            if i>0:
                t.append(i)
        return np.mean(t)

    def get_actcount_var(s):
        s = np.array(s)
        t = []
        for i in s:
            if i>0:
                t.append(i)
        return np.var(t)
        
    def get_actcount_max(s):
        s = np.array(s)
        return np.max(s)  
        
    
        
    gp = tact.groupby(['user_id','day']).size().unstack()
    tdf = pd.DataFrame()
    #tdf['actcount_mean'] = gp.apply(get_actcount_mean)
    tdf['actcount_var'] = gp.apply(get_actcount_var)
    #tdf['actcount_max'] = gp.apply(get_actcount_max)
    tdf['act_last_gap'] = gp.apply(get_last_gap)
    #tdf['act_max_continue_day'] = gp.apply(get_max_continue_day)
    gp = gp.reset_index()
    tdf['user_id'] = gp['user_id']
    df = pd.merge(df,tdf,on=['user_id'],how='left')  

    
    
    gp = tact[['user_id']].groupby(['user_id']).size().rename('act#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_mean#'] = df['act#']/df['register_time']
    
    for i in range(7):
        gp = tact[tact.day>=lastday-i][['user_id']].groupby(['user_id']).size().rename('last_'+str(i)+'_days_act#').reset_index()
        df = pd.merge(df,gp,on=['user_id'],how='left')
        df['temp'] = df['register_time'].apply(lambda x: min([x,i+1]))
        df['last_'+str(i)+'_days_act#m'] = df['last_'+str(i)+'_days_act#']/df['temp']
    
    gp = tact[['user_id','day']].groupby(['user_id'])['day'].nunique().rename('act_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_u_mean#'] = df['act_u#']/df['register_time']
    
    gp = tact[['user_id','author_id']].groupby(['user_id'])['author_id'].nunique().rename('act_author_id_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_author_id_u_mean#'] = df['act_author_id_u#']/df['register_time']
    
    gp = tact[['user_id','video_id']].groupby(['user_id'])['video_id'].nunique().rename('act_video_id_u#').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_video_id_u_mean#'] = df['act_video_id_u#']/df['register_time']


    df['video_author_m'] = df['act_video_id_u#']/df['act_author_id_u#']
    df['act_author_id_u_mean1#'] = df['act_author_id_u#']/df['act_u#']
    df['act_video_id_u_mean1#'] = df['act_video_id_u#']/df['act_u#']
    
    gp = tact[['user_id','day']].groupby(['user_id'])['day'].min().rename('act_day_min').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tact[['user_id','day']].groupby(['user_id'])['day'].max().rename('act_day_max').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tact[['user_id','day']].groupby(['user_id'])['day'].mean().rename('act_day_mean').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')  
    
    gp = tact[['user_id','day']].groupby(['user_id'])['day'].var().rename('act_day_var').reset_index()
    df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['act_day_gap'] = df['act_day_max'] - df['act_day_min']
    #df['last_act_day'] = lastday - df['act_day_max']+1
    #del df['act_day_max']
    
    page_list = list(tact['page'].unique())

 
    for c in page_list: 
        gp = tact[tact['page']==c][['user_id']].groupby(['user_id']).size().rename('act_page_'+str(c)+'#').reset_index()
        df = pd.merge(df,gp,on=['user_id'],how='left')
    
    df['author_id'] = df['user_id']
    gp = tact[['author_id']].groupby(['author_id']).size().rename('author#').reset_index()
    df = pd.merge(df,gp,on=['author_id'],how='left')
    

    for c in page_list: 
        gp = tact[tact['page']==c][['author_id']].groupby(['author_id']).size().rename('author_act_page_'+str(c)+'#').reset_index()
        df = pd.merge(df,gp,on=['author_id'],how='left')
    
    
    #show(df)
    del df['author_id'], df['app_day_min'],df['author_act_page_0#'],df['temp']
    
    #print ('feature ok')
    return df

def predict_1_28(val,register,app,video,act):
    path = '../data1/1_28/'
        
    def get_features_all(df,df1):
        lendf = len(df)
        df= df.append(df1)
        del df1
        gc.collect()
        
        df = docount(df,df,'ALL',['register_type'])
        df = docount(df,df,'ALL',['device_type'])
        
        
        del df['user_id']

        df1 = df[lendf:]
        df = df[:lendf]
        return df,df1
        
    df1 = register[register.register_day<15]
    df1['register_time'] = 17-register.register_day
    df2 = register[register.register_day<22]
    df2['register_time'] = 24-register.register_day
    df2[df2['register_time']>16]['register_time'] = 16


    
    test_df = register[register.register_day<29]
    test_df['register_time'] = 31-test_df.register_day
    df2[df2['register_time']>16]['register_time'] = 16

    
    del df1['register_day'],df2['register_day'],test_df['register_day']
    
    if os.path.exists(path+'train_y1.csv'):
        train_y1=pd.read_csv(path+'train_y1.csv')
        
    else:
        train_y1 = is_active(df1,17,23,app,video,act)
        train_y1.to_csv(path+'train_y1.csv',index=False)
    train_y1 = train_y1['Y']
    if os.path.exists(path+'train_y2.csv'):
        train_y2=pd.read_csv(path+'train_y2.csv')
        
    else:
        train_y2 = is_active(df2,24,30,app,video,act)
        train_y2.to_csv(path+'train_y2.csv',index=False)
    train_y2 = train_y2['Y']        
        
    if os.path.exists(path+'df1.csv'):
        df1=pd.read_csv(path+'df1.csv')
    else:
        df1 = get_features_ks(df1,1,16,app,video,act)
        df1.to_csv(path+'df1.csv',index=False)
    
    if os.path.exists(path+'df2.csv'):
        df2=pd.read_csv(path+'df2.csv')
    else:
        df2 = get_features_ks(df2,8,23,app,video,act)
        df2.to_csv(path+'df2.csv',index=False)
        
    if val:
        train_df = df1
        test_df = df2
        train_y = train_y1
        val_y = train_y2
    else:
        if os.path.exists(path+'test_df.csv'):
            test_df=pd.read_csv(path+'test_df.csv')
        else:
            test_df = get_features_ks(test_df,15,30,app,video,act)
            test_df.to_csv(path+'test_df.csv',index=False)
        
        train_df = df1.append(df2)
        train_y = train_y1.append(train_y2)
        #train_df = df2
        #train_y = train_y2
    
    del df1,df2
    gc.collect()
    ids = test_df['user_id']
    train_df,test_df = get_features_all(train_df,test_df)    
    '''
    train_df['Y'] = train_y
    print (len(train_df))
    train_js = train_df[train_df['act_mean#']==0]  
    train_df = train_df[train_df['act_mean#']>0]  
    print (len(train_df))
    train_y = train_df['Y']
    del train_df['Y']
    train_y_js = train_js['Y']
    del train_js['Y']
    
    test_df['Y'] = val_y
    test_js =  test_df[test_df['act_mean#']==0] 
    test_df =  test_df[test_df['act_mean#']>0] 
    val_y = test_df['Y']
    del test_df['Y']
    js_y = test_js['Y']
    del test_js['Y']
    '''
    pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=1)
    #pre_train_js,test_y_js = predict_data(train_js,train_y_js,10,test_js,importance=1)
    '''
    test_df['Y'] = val_y
    test_df['Y1'] = test_y
    test_js =  test_df[test_df['act_mean#']==0] 
    print(len(test_js))
    print(len(test_js[test_js['Y1']>=0.4]))
    print(len(test_js[(test_js['Y1']>=0.4) & (test_js['Y']==1)]))
    test_df[(test_df['act_mean#']==0) & (test_df['Y1']>=0.4)]['Y1'] = 0
    print (len(test_df[(test_df['act_mean#']==0) & (test_df['Y1']>=0.4)]))
    test_y[(test_df['act_mean#']==0) & (test_df['Y1']>=0.4)] = 0
    '''
    
    if val==1:   
        showresults(val_y,test_y) 
        showtop(val_y,test_y,nums=15428)
        showtop(val_y,test_y,nums=15905)
        showfalse(ids,test_df,val_y,test_y)
        #showprecision(val_y,test_y)
        return ids,test_y,getbest(ids,test_y,th=0.4)   
    else:
        showresults(train_y,pre_train)     
        showtop(train_y,pre_train,nums=25713)
    
        #return ids,test_y,getbest(ids,test_y,th=0.4) 
        return ids,test_y,getbest(ids,test_y,rank=22088) 
        
if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
    
    val = 1
    ids,test_y,ans = predict_1_28(val,register,app,video,act)      
    if val==0:
        print(len(ans))
        import time
        name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        submission = pd.DataFrame({'user_id': ans})
        submission.to_csv('1_28_submit'+name+'.csv', index=False, header = None)
        