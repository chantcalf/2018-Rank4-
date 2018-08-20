# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 21:23:18 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import os
from utils import docount,domean,dovar,doiq,domin,domax
from utils import is_active,showresults,predict_data,getbest,showtop
import warnings
warnings.filterwarnings("ignore")

def predict_24_28(val,register,app,video,act):
    
    def get_features(df,d1,d2):
        tapp = app[(app.day>=d1) & (app.day<=d2)]
        tact = act[(act.day>=d1) & (act.day<=d2)]
        tvideo = video[(video.day>=d1) & (video.day<=d2)]
        tapp.day = tapp.day - d1
        tact.day = tact.day - d1
        tvideo.day = tvideo.day - d1
        lastday = d2-d1

        df['register_time'] = d2-df.register_day+1
        del df['register_day']
        
        #app
        df = docount(df,tapp,'app',['user_id'])
        df['app_mean#'] = df['app$user_id#']/df['register_time']
        #df = domax(df,tapp,'app',['user_id'],'day')
        #df['last_app_day'] = lastday - df['app$user_id_by_day_max']+1
        del df['app$user_id#']
        #df['app_day_missing'] = df['register_time'] - df['app$user_id#']
        #df['app$user_id#'] = df['app$user_id#']/df['register_time']
        
        #df = dovar(df,tapp,'app',['user_id'],'day')
        #df = docount(df,tapp[tapp.day>lastday-2],'app_last_2',['user_id'])        
        #df = docount(df,tapp[tapp.day>lastday-1],'app_last_1',['user_id']) 
        #df = docount(df,tapp[tapp.day==lastday],'app_last_1',['user_id'])
        gc.collect()
        #video
        #df = docount(df,tvideo,'video',['user_id'])
        #df['video_mean#'] = df['video$user_id#']/df['register_time']
        #df = domax(df,tvideo,'video',['user_id'],'day')
        #df['last_video_day'] = lastday - df['video$user_id_by_day_max']+1
        #del df['video$user_id_by_day_max']
        #df = doiq(df,tvideo,'video',['user_id'],'day')
        #df['last_video_day'] = lastday - df['video$user_id_by_day_max']+1
        #df['video_day_missing'] = df['register_time'] - df['video$user_id_by_day_iq']
        #df['video$user_id#'] = df['video$user_id#']/df['register_time']
        
        #df = dovar(df,tvideo,'video',['user_id'],'day')     
        df = docount(df,tvideo[tvideo.day>lastday-2],'video_last_2',['user_id'])
        df = docount(df,tvideo[tvideo.day>lastday-3],'video_last_3',['user_id'])
        #df = docount(df,tvideo[tvideo.day==lastday],'video_last_1',['user_id'])
        gc.collect()
        #act
        #gp = act.groupby(['user_id','day']).size().unstack()
        #df = pd.merge(df,gp.max(1).rename('actcount_max').reset_index(),on=['user_id'],how='left')   
        #df = pd.merge(df,gp.mean(1).rename('actcount_mean').reset_index(),on=['user_id'],how='left')
        #df = pd.merge(df,gp.var(1).rename('actcount_var').reset_index(),on=['user_id'],how='left')        
        
        #df = docount(df,tact,'act',['user_id'])
        #df['act_mean#'] = df['act$user_id#']/df['register_time']
        df = domax(df,tact,'act',['user_id'],'day')
        df['last_act_day'] = lastday - df['act$user_id_by_day_max']+1
        del df['act$user_id_by_day_max']
        #df = doiq(df,tact,'act',['user_id'],'day')
        #df['last_act_day'] = lastday - df['act$user_id_by_day_max']+1
        #df['act_day_missing'] = df['register_time'] - df['act$user_id_by_day_iq']
        #df['act$user_id#'] = df['act$user_id#']/df['register_time']
        
        #gp = tact.groupby(['user_id','day']).size().unstack()
        #df = pd.merge(df,gp.max(1).rename('actcount_max').reset_index(),on=['user_id'],how='left')   
        #df = pd.merge(df,gp.mean(1).rename('actcount_mean').reset_index(),on=['user_id'],how='left')
        #df = pd.merge(df,gp.var(1).rename('actcount_var').reset_index(),on=['user_id'],how='left')

        #df = dovar(df,tact,'act',['user_id'],'day')      
        df = docount(df,tact[tact.day>lastday-2],'act_last_2',['user_id']) 
        df = docount(df,tact[tact.day>lastday-3],'act_last_3',['user_id'])
        #df = docount(df,tact[tact.day==lastday],'act_last_1',['user_id'])
        gc.collect()
        
        #page_list = list(tact['page'].unique())
                
        for c in [0,1,2,3]: 
            df = docount(df,tact[(tact['page']==c) & (tact.day>lastday-3)],'act_last_3_page='+str(c),['user_id']) 
            df = docount(df,tact[(tact['page']==c) & (tact.day>lastday-2)],'act_last_2_page='+str(c),['user_id'])
            df = docount(df,tact[(tact['page']==c) & (tact.day>lastday-1)],'act_last_1_page='+str(c),['user_id']) 
        
        df = doiq(df,tact[tact.day>lastday-3],'act_last_3',['user_id'],'author_id')  
        df = doiq(df,tact[tact.day>lastday-3],'act_last_3',['user_id'],'video_id')
        
        df = doiq(df,tact[tact.day>lastday-2],'act_last_2',['user_id'],'author_id')  
        df = doiq(df,tact[tact.day>lastday-2],'act_last_2',['user_id'],'video_id')
        
        df = doiq(df,tact[tact.day>lastday-1],'act_last_1',['user_id'],'author_id')  
        df = doiq(df,tact[tact.day>lastday-1],'act_last_1',['user_id'],'video_id')
        
        for c in [0,1,2,3]: 
            df = docount(df,tact[(tact['action_type']==c) & (tact.day>lastday-3)],'act_last_3_action_type='+str(c),['user_id'])
            df = docount(df,tact[(tact['action_type']==c) & (tact.day>lastday-2)],'act_last_2_action_type='+str(c),['user_id'])
            df = docount(df,tact[(tact['action_type']==c) & (tact.day>lastday-1)],'act_last_1_action_type='+str(c),['user_id'])

        
        gc.collect()
        
        
        return df
    
    
    path = '../data1/24_28/'
    if val:
        if os.path.exists(path+'val_df.csv'):
            test_df = pd.read_csv(path+'val_df.csv')
            val_y = pd.read_csv(path+'val_y.csv')
        else:
            test_df = register[(register.register_day>=17) & (register.register_day<=21)]
            test_df = get_features(test_df,17,23)
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
            for i in range(1,11):
                df = register[(register.register_day>=i) & (register.register_day<=i+4)]
                y = is_active(df,i+7,i+13,app,video,act)
                df = get_features(df,i,i+6)
                train_df = train_df.append(df)
                train_y = train_y.append(y)
            train_df.to_csv(path+'val_train_df.csv',index=False)
            train_y.to_csv(path+'val_train_y.csv',index=False)
    else:
        if os.path.exists(path+'test_df.csv'):
            test_df = pd.read_csv(path+'test_df.csv')
        else:
            test_df = register[(register.register_day>=24) & (register.register_day<=28)]
            test_df = get_features(test_df,24,30)
            test_df.to_csv(path+'test_df.csv',index=False)
                               
        if os.path.exists(path+'train_df.csv'):
            train_df = pd.read_csv(path+'train_df.csv')
            train_y = pd.read_csv(path+'train_y.csv')
        else:            
            if os.path.exists(path+'val_train_df.csv'):
                train_df = pd.read_csv(path+'val_train_df.csv')
                train_y = pd.read_csv(path+'val_train_y.csv') 
                for i in range(11,18):
                    df = register[(register.register_day>=i) & (register.register_day<=i+4)]
                    y = is_active(df,i+7,i+13,app,video,act)
                    df = get_features(df,i,i+6)
                    train_df = train_df.append(df)
                    train_y = train_y.append(y)  
            else:
                train_df = pd.DataFrame()   
                train_y = pd.DataFrame()                  
                for i in range(1,18):
                    df = register[(register.register_day>=i) & (register.register_day<=i+4)]
                    y = is_active(df,i+7,i+13,app,video,act)
                    df = get_features(df,i,i+6)
                    train_df = train_df.append(df)
                    train_y = train_y.append(y)  
            train_df.to_csv(path+'train_df.csv',index=False)
            train_y.to_csv(path+'train_y.csv',index=False)                 
    train_y = train_y['Y']
    #print(sum(train_y)/len(train_y))
        
    def get_features_all(df,df1):
        lendf = len(df)
        
        df= df.append(df1)
        del df1
        gc.collect()
        
        #for c in ['act_last_2$user_id#']:
        #    df = domean(df,df,'All',['device_type'],c);gc.collect()
        #    df = domean(df,df,'All',['register_type'],c);gc.collect()
            
        #del df
            
        #ccc = ['device_type', 'app_mean#', 'register_type', 'register_time', 'act_last_3_page=1$user_id#', 'last_act_day', 'act_last_3$user_id_by_video_id_iq', 'act_last_3_page=2$user_id#', 'act_last_3$user_id_by_author_id_iq', 'act_last_3_action_type=1$user_id#', 'act_last_1$user_id_by_author_id_iq', 'act_last_3_page=3$user_id#', 'act_last_3_page=0$user_id#', 'act_last_1$user_id_by_video_id_iq', 'act_last_2$user_id_by_author_id_iq', 'act_last_3$user_id#', 'act_last_2$user_id_by_video_id_iq', 'act_last_3_action_type=2$user_id#', 'act_last_3_action_type=0$user_id#', 'act_last_2_page=2$user_id#', 'act_last_2_page=1$user_id#', 'act_last_2_page=3$user_id#', 'act_last_1_page=1$user_id#', 'act_last_2$user_id#', 'act_last_2_page=0$user_id#', 'act_last_1_action_type=0$user_id#', 'act_last_2_action_type=1$user_id#', 'act_last_1_page=2$user_id#', 'act_last_3_action_type=3$user_id#', 'act_last_1_page=3$user_id#', 'act_last_2_action_type=0$user_id#', 'video_last_3$user_id#', 'act_last_1_page=0$user_id#', 'act_last_2_action_type=2$user_id#', 'act_last_2_action_type=3$user_id#', 'video_last_2$user_id#', 'act_last_1_action_type=1$user_id#', 'act_last_1_action_type=3$user_id#', 'act_last_1_action_type=2$user_id#']
        #for i in range(38,39):
        #    del df[ccc[i]]            
            
            
        
        del df['user_id']
        #del df['last_app_day'],df['last_video_day'],df['video_last_1$user_id#'],df['app_last_1$user_id#']
        #del df['act_last_1$user_id#'],df['app_last_2$user_id#']
        
 
        df1 = df[lendf:]
        df = df[:lendf]
        return df,df1
        
    
    ids = test_df['user_id']
    train_df,test_df = get_features_all(train_df,test_df)    
    
    pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=1)
    #print(test_y)
    if val==1:   
        print (len(train_y),sum(train_y))
        showresults(val_y,test_y) 
        showtop(val_y,test_y,nums=4723)
        showtop(train_y,pre_train,nums=38507)
        #return ids,test_y,getbest(ids,test_y,rank=4723)
        return ids,test_y,getbest(ids,test_y,th=0.4)
    else:
        showresults(train_y,pre_train)     
        showtop(train_y,pre_train,nums=70275)
        return ids,test_y,getbest(ids,test_y,rank=5498)     
        
if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
    
    val = 1
    ids,test_y,ans = predict_24_28(val,register,app,video,act)   
    if val==0:
        print(len(ans))
        import time
        name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        submission = pd.DataFrame({'user_id': ans})
        submission.to_csv('jt_submit'+name+'.csv', index=False, header = None)     
        