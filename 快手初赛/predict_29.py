# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:46:03 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import os
from utils import docount,domean,dovar,doiq,is_active,showresults,predict_data,getbest,showtop
import warnings
warnings.filterwarnings("ignore")

def predict_29(val,register,app,video,act):
    def get_features(df,d1,d2):
        tapp = app[(app.day>=d1) & (app.day<=d2)]
        tact = act[(act.day>=d1) & (act.day<=d2)]
        tvideo = video[(video.day>=d1) & (video.day<=d2)]
        tapp.day = tapp.day - d1
        tact.day = tact.day - d1
        tvideo.day = tvideo.day - d1
        lastday = d2-d1      
        #df['register_time'] = d2-df.register_day+1
        
        df = docount(df,tapp,'app',['user_id']);gc.collect() 
        df = docount(df,tapp[tapp.day==lastday],'last_day_app',['user_id']);gc.collect()
        #df['app_mean#'] = df['app$user_id#']/2
        df = docount(df,tvideo,'video',['user_id']);gc.collect()
        df['videorate'] = df['video$user_id#']/(tvideo.shape[0]+0.000001)
        #df['video_mean#'] = df['video$user_id#']/2
        df = docount(df,tact,'act',['user_id']);gc.collect()
        df = docount(df,tact[tact.day==lastday],'last_day_act',['user_id']);gc.collect()
        df = docount(df,tact[tact.day==lastday-1],'first_day_act',['user_id']);gc.collect()
        df['actrate'] = df['act$user_id#']/(tact.shape[0]+0.000001)
        df['last_day_actrate'] = df['last_day_act$user_id#']/(tact.shape[0]+0.000001)
        df['first_day_actrate'] = df['first_day_act$user_id#']/(tact.shape[0]+0.000001)
        df['actrate_gap'] = df['last_day_actrate'] - df['first_day_actrate']
        df['act_gap'] = df['last_day_act$user_id#'] - df['first_day_act$user_id#']
        #df['act_mean#'] = df['act$user_id#']/2
        #page_list = list(tact['page'].unique())
        def iszero(s):
            if s==0:
                return 0
            return 1
        df['act0'] = df['act$user_id#'].apply(iszero)
        df['video0'] = df['video$user_id#'].apply(iszero)    
        
        
        
        for c in [1]: 
            df = docount(df,tact[tact.day==lastday][tact['page']==c],'last_day_act_page='+str(c),['user_id']);gc.collect()
        
        for c in [0,1,2,3,4]: 
            df = docount(df,tact[tact['page']==c],'act_page='+str(c),['user_id']);gc.collect()
            df['act_page='+str(c)+'$user_id#rate'] = df['act_page='+str(c)+'$user_id#']/(df['act$user_id#']+0.00001)
        
        df['act_page=23$user_id#'] = df['act_page=2$user_id#'] + df['act_page=3$user_id#']
        df['act_page=023$user_id#'] = df['act_page=2$user_id#'] + df['act_page=3$user_id#']+df['act_page=0$user_id#']

        
        action_list = list(tact['action_type'].unique())
        for c in [0,1,2,3,4,5]: 
            df = docount(df,tact[tact['action_type']==c],'action_type='+str(c),['user_id']);gc.collect()
            df = docount(df,tact[tact.day==lastday][tact['action_type']==c],'last_day_action_type='+str(c),['user_id']);gc.collect()
            df['action_type='+str(c)+'$user_id#rate'] = df['action_type='+str(c)+'$user_id#']/(df['act$user_id#']+0.00001)


        df['author_id'] = df['user_id']
        
        df = doiq(df,tact[tact.day==lastday],'last_day_act',['user_id'],'video_id');gc.collect()
        df = doiq(df,tact[tact.day==lastday],'last_day_act',['user_id'],'author_id');gc.collect()
        df['last_day_act$author_video_m'] = df['last_day_act$user_id_by_video_id_iq']/df['last_day_act$user_id_by_author_id_iq']
        
        df = doiq(df,tact[tact.day==lastday-1],'first_day_act',['user_id'],'video_id');gc.collect()
        df = doiq(df,tact[tact.day==lastday-1],'first_day_act',['user_id'],'author_id');gc.collect()
        df['first_day_act$author_video_m'] = df['first_day_act$user_id_by_video_id_iq']/df['first_day_act$user_id_by_author_id_iq']

        
        df = doiq(df,tact[tact.day>=lastday-1],'last2_day_act',['user_id'],'video_id');gc.collect()
        df = doiq(df,tact[tact.day>=lastday-1],'last2_day_act',['user_id'],'author_id');gc.collect()
        df['last2_day_act$author_video_m'] = df['last2_day_act$user_id_by_video_id_iq']/df['last2_day_act$user_id_by_author_id_iq']

        
        
        del df['register_day'],df['author_id']
        return df

    def get_features_all(df,df1):
        lendf = len(df)
        df= df.append(df1)
        del df1
        gc.collect()
        df = docount(df,df,'ALL',['register_type']) 

        del df['user_id']
        
        ccc = ['device_type', 'register_type', 'action_type=0$user_id#rate', 'act_page=1$user_id#', 'first_day_act$user_id_by_author_id_iq', 'action_type=2$user_id#rate', 'act_page=0$user_id#rate', 'last_day_act$author_video_m', 'action_type=1$user_id#rate', 'act_page=2$user_id#', 'actrate', 'last_day_act$user_id_by_author_id_iq', 'app$user_id#', 'last_day_act_page=1$user_id#', 'act_page=3$user_id#rate', 'last_day_action_type=0$user_id#', 'first_day_act$user_id_by_video_id_iq', 'videorate', 'act_page=1$user_id#rate', 'last2_day_act$user_id_by_author_id_iq', 'last2_day_act$user_id_by_video_id_iq', 'first_day_actrate', 'act_page=2$user_id#rate', 'last_day_actrate', 'first_day_act$author_video_m', 'last2_day_act$author_video_m', 'ALL$register_type#', 'act_page=0$user_id#', 'actrate_gap', 'action_type=3$user_id#rate', 'last_day_act$user_id#', 'act$user_id#', 'last_day_act$user_id_by_video_id_iq', 'action_type=0$user_id#', 'action_type=1$user_id#', 'act_gap', 'action_type=2$user_id#', 'action_type=3$user_id#', 'first_day_act$user_id#', 'act_page=3$user_id#', 'act_page=4$user_id#rate', 'video$user_id#', 'last_day_action_type=1$user_id#', 'act_page=23$user_id#', 'act_page=023$user_id#', 'act_page=4$user_id#', 'last_day_action_type=2$user_id#', 'last_day_action_type=3$user_id#', 'action_type=5$user_id#rate', 'action_type=5$user_id#', 'last_day_app$user_id#', 'last_day_action_type=4$user_id#', 'action_type=4$user_id#', 'last_day_action_type=5$user_id#', 'act0', 'action_type=4$user_id#rate', 'video0']
        ccc1 = [ ]
        
        ddd = ['action_type=2$user_id#rate','action_type=1$user_id#rate','last_day_act$user_id_by_author_id_iq',
               'last_day_act_page=1$user_id#','act_page=3$user_id#rate','first_day_act$user_id_by_video_id_iq',
               'videorate','act_page=1$user_id#rate','last2_day_act$user_id_by_author_id_iq','last2_day_act$user_id_by_video_id_iq',
               'act_page=2$user_id#rate','last_day_actrate', 'first_day_act$author_video_m','last2_day_act$author_video_m',
               'ALL$register_type#','act_page=0$user_id#','actrate_gap','action_type=3$user_id#rate',
               'last_day_act$user_id#','act$user_id#','last_day_act$user_id_by_video_id_iq', 'action_type=0$user_id#', 
               'action_type=1$user_id#','act_gap', 'action_type=2$user_id#','action_type=3$user_id#',
               'first_day_act$user_id#', 'act_page=3$user_id#','act_page=4$user_id#rate', 'video$user_id#', 
               'last_day_action_type=1$user_id#','act_page=23$user_id#', 'act_page=023$user_id#','act_page=4$user_id#', 
               'last_day_action_type=2$user_id#','last_day_action_type=3$user_id#', 'action_type=5$user_id#rate',
               'action_type=5$user_id#', 'last_day_app$user_id#','last_day_action_type=4$user_id#',
               'action_type=4$user_id#','last_day_action_type=5$user_id#', 'act0', 'action_type=4$user_id#rate', 'video0']
        
        used = ['device_type', 'register_type', 'action_type=0$user_id#rate', 'act_page=1$user_id#',
                'first_day_act$user_id_by_author_id_iq', 'act_page=0$user_id#rate','last_day_act$author_video_m',
                'act_page=2$user_id#','actrate','app$user_id#', 'last_day_action_type=0$user_id#',
                'first_day_actrate', 'action_type=5$user_id#rate', ]
        
        df = df[used]
        
        
         
        df1 = df[lendf:]
        df = df[:lendf]
        return df,df1
    
    path = '../data1/29/'
    
    if val:
        if os.path.exists(path+'val_df.csv'):
            test_df = pd.read_csv(path+'val_df.csv')
            val_y = pd.read_csv(path+'val_y.csv')
        else:
            test_df = register[(register.register_day==22)]
            test_df = get_features(test_df,22,23)
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
            for i in range(1,22):
                df = register[(register.register_day==i)]
                y = is_active(df,i+2,i+8,app,video,act)
                df = get_features(df,i,i+1)
                train_df = train_df.append(df)
                train_y = train_y.append(y)
            train_df.to_csv(path+'val_train_df.csv',index=False)
            train_y.to_csv(path+'val_train_y.csv',index=False)
    else:
        if os.path.exists(path+'test_df.csv'):
            test_df = pd.read_csv(path+'test_df.csv')
        else:
            test_df = register[(register.register_day==29)]
            test_df = get_features(test_df,29,30)
            test_df.to_csv(path+'test_df.csv',index=False)
                               
        if os.path.exists(path+'train_df.csv'):
            train_df = pd.read_csv(path+'train_df.csv')
            train_y = pd.read_csv(path+'train_y.csv')
        else:            
            if os.path.exists(path+'val_train_df.csv'):
                train_df = pd.read_csv(path+'val_train_df.csv')
                train_y = pd.read_csv(path+'val_train_y.csv')
                val_df = pd.read_csv(path+'val_df.csv')
                val_y = pd.read_csv(path+'val_y.csv')
                train_df = train_df.append(val_df)
                train_y = train_y.append(val_y)
            else:
                train_df = pd.DataFrame()   
                train_y = pd.DataFrame()                  
                for i in range(1,23):
                    df = register[(register.register_day==i)]
                    y = is_active(df,i+2,i+8,app,video,act)
                    df = get_features(df,i,i+1)
                    train_df = train_df.append(df)
                    train_y = train_y.append(y)  
            train_df.to_csv(path+'train_df.csv',index=False)
            train_y.to_csv(path+'train_y.csv',index=False)                 
    train_y = train_y['Y']

    ids = test_df['user_id']
    train_df,test_df = get_features_all(train_df,test_df)
    
    pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=1)
    
    if val==1:   
        print (len(train_y),sum(train_y))
        showresults(train_y,pre_train)
        showresults(val_y,test_y) 
        showtop(val_y,test_y,nums=1337)
        showtop(train_y,pre_train,nums=19589)
        return ids,test_y,getbest(ids,test_y,th=0.4)
    else:
        showresults(train_y,pre_train)     
        showtop(train_y,pre_train,nums=20926)
        return ids,test_y,getbest(ids,test_y,rank=1294)
        #return ids,test_y,getbest(ids,test_y,th=0.4)
    
    
    
if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
    
    val =1
    #ans = list(register[register.register_day==29]['user_id'])
    ids,test_y,ans = predict_29(val,register,app,video,act)
    if val==0:
        print(len(ans))
        import time
        name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        submission = pd.DataFrame({'user_id': ans})
        submission.to_csv('29_submit'+name+'.csv', index=False, header = None)