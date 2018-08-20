# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 20:02:05 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import gc
import lightgbm as lgb
import os
from utils import docount,domean,dovar,doiq,is_active,showresults,predict_data,getbest,showfalse
from utils import showtop,showprecision
def predict_30(val,register,app,video,act):
    def get_features0(df,d):
        #tapp = app[app.day==d]
        tvideo = video[video.day==d]
        tact = act[act.day==d]        
        #df = docount(df,tapp,'app',['user_id']);gc.collect()        
        df = docount(df,tvideo,'video',['user_id']);gc.collect()
        df['videorate'] = df['video$user_id#']/(tvideo.shape[0]+0.000001)
        df = docount(df,tact,'act',['user_id']);gc.collect()
        df['actrate'] = df['act$user_id#']/(tact.shape[0]+0.000001)
        
        page_list = list(tact['page'].unique())
        for c in [0,1,2,3,4]: 
            df = docount(df,tact[tact['page']==c],'act_page='+str(c),['user_id']);gc.collect()
            df['act_page='+str(c)+'$user_id#rate'] = df['act_page='+str(c)+'$user_id#']/(df['act$user_id#']+0.00001)
        
        df['act_page=23$user_id#'] = df['act_page=2$user_id#'] + df['act_page=3$user_id#']
        df['act_page=023$user_id#'] = df['act_page=2$user_id#'] + df['act_page=3$user_id#']+df['act_page=0$user_id#']
        
        action_list = list(tact['action_type'].unique())
        for c in [0,1,2,3,4,5]: 
            df = docount(df,tact[tact['action_type']==c],'action_type='+str(c),['user_id']);gc.collect()
            df['action_type='+str(c)+'$user_id#rate'] = df['action_type='+str(c)+'$user_id#']/(df['act$user_id#']+0.00001)
            
        df['action_type=01$user_id#'] = df['action_type=0$user_id#'] + df['action_type=1$user_id#']
        
        def iszero(s):
            if s==0:
                return 0
            return 1
        
        df['pageall'] = df['act_page=0$user_id#'].apply(iszero)
        for c in [1,2,3,4]: 
            df['pageall'] = df['pageall'] * df['act_page=0$user_id#']
        df['pageall'] = df['act_page=0$user_id#'].apply(iszero)
        
        df['actionall'] = df['action_type=0$user_id#'].apply(iszero)
        for c in [1,2,3,4,5]: 
            df['pageall'] = df['pageall'] * df['action_type=0$user_id#']
        df['actionall'] = df['action_type=0$user_id#'].apply(iszero)
        
        df['act0'] = df['act$user_id#'].apply(iszero)
        df['video0'] = df['video$user_id#'].apply(iszero)
                
        def bigact(s):
            if s>=50:
                return 5
            else:
                return int(s/10)
        df['act$user_id#10'] = df['act$user_id#'].apply(bigact)
        
        df['author_id'] = df['user_id']
        df = docount(df,tact,'act',['author_id']);gc.collect()
        df = doiq(df,tact,'act',['user_id'],'video_id');gc.collect()
        df = doiq(df,tact,'act',['user_id'],'author_id');gc.collect()
        
        df['act$author_video_m'] = df['act$user_id_by_video_id_iq']/df['act$user_id_by_author_id_iq']

        
        
        del df['register_day'],df['author_id']
        return df

    def get_features_all(df,df1):
        lendf = len(df)
        df= df.append(df1)
        del df1
        gc.collect()
         
        
        for c in ['act$user_id#']:
            #df = domean(df,df,'All',['device_type'],c);gc.collect()
            df = domean(df,df,'All',['register_type'],c);gc.collect()
            #df = dovar(df,df,'All',['register_type'],c);gc.collect()
        df = docount(df,df,'ALL',['register_type'])
        df = docount(df,df,'ALL',['device_type'])
        
        del df['user_id'],
        
        ccc = ['device_type', 'actrate', 'All$register_type_by_act$user_id#_mean', 'act_page=1$user_id#', 'action_type=0$user_id#rate', 'action_type=1$user_id#rate', 'register_type', 'act$user_id_by_author_id_iq', 'act$user_id_by_video_id_iq', 'videorate', 'act_page=1$user_id#rate', 'act$author_video_m', 'action_type=2$user_id#rate', 'act_page=3$user_id#rate', 'act_page=0$user_id#', 'action_type=0$user_id#', 'act_page=2$user_id#', 'act_page=2$user_id#rate', 'action_type=1$user_id#', 'act$user_id#', 'act_page=4$user_id#rate', 'act_page=0$user_id#rate', 'pageall', 'act_page=4$user_id#', 'action_type=3$user_id#rate', 'act_page=23$user_id#', 'act_page=3$user_id#', 'video$user_id#', 'action_type=2$user_id#', 'action_type=3$user_id#', 'act_page=023$user_id#', 'act$author_id#', 'action_type=01$user_id#', 'action_type=5$user_id#rate', 'ALL$register_type#', 'action_type=5$user_id#', 'act$user_id#10', 'action_type=4$user_id#', 'actionall', 'action_type=4$user_id#rate', 'act0', 'video0']
        ccc1 = [         ]

        ddd = ['All$register_type_by_act$user_id#_mean','act_page=1$user_id#','action_type=1$user_id#rate',
               'act$user_id_by_author_id_iq','act$user_id_by_video_id_iq', 'act$author_video_m',
               'act_page=2$user_id#','act_page=2$user_id#rate','action_type=1$user_id#','act$user_id#',
               'act_page=4$user_id#rate','act_page=4$user_id#','action_type=3$user_id#rate', 'act_page=23$user_id#',
               'act_page=3$user_id#','video$user_id#','action_type=2$user_id#', 'action_type=3$user_id#',
               'act$author_id#','action_type=01$user_id#','ALL$register_type#','ALL$device_type#',
               'action_type=5$user_id#rate','action_type=5$user_id#','act$user_id#10','action_type=4$user_id#',
               'actionall', 'action_type=4$user_id#rate', 'act0',]
               
        used = ['device_type','register_type','actrate','action_type=0$user_id#rate',
                'videorate','act_page=1$user_id#rate','action_type=2$user_id#rate',
                'act_page=3$user_id#rate','act_page=0$user_id#', 'action_type=0$user_id#',
                'act_page=0$user_id#rate','pageall', 'act_page=023$user_id#', 'video0',
                'All$register_type_by_act$user_id#_mean','ALL$register_type#',]
        
        df = df[used]
        
        
        
        
        

 
        df1 = df[lendf:]
        df = df[:lendf]
        return df,df1
       
        
    path = '../data1/30/'
    if os.path.exists(path+'train_df.csv'):
        train_df = pd.read_csv(path+'train_df.csv')
        train_y = pd.read_csv(path+'train_y.csv')
        
    else:
        train_df = pd.DataFrame()
        train_y = pd.DataFrame()
        for i in range(1,24):
            df = register[register.register_day==i]
            y = is_active(df,i+1,i+7,app,video,act)
            df = get_features0(df,i)
            train_df = train_df.append(df)
            train_y = train_y.append(y)
            if i==22:
                valst = len(train_df)
                print (valst)
                
        train_df.to_csv(path+'train_df.csv',index=False)
        train_y.to_csv(path+'train_y.csv',index=False)
    
    train_y = train_y['Y']    
    if val:
        #35134
        valst = 35134
        test_df = train_df[valst:]
        val_y = train_y[valst:]
        train_df = train_df[:valst]
        train_y = train_y[:valst]
    else:
        if os.path.exists(path+'test_df.csv'):
            test_df = pd.read_csv(path+'test_df.csv')
        else:
            test_df = register[register.register_day==30]
            test_df = get_features0(test_df,30)
            test_df.to_csv(path+'test_df.csv',index=False)

    #train_df['Y'] = train_y 
    #act0train = train_df[train_df['act$user_id#']==0]
    #print(len(act0train),len(act0train[act0train['Y']==1]))
    #del train_df['Y']     
    #act0ids = test_df[test_df['act$user_id#']==0]['user_id']
            
    ids = test_df['user_id']
    train_df,test_df = get_features_all(train_df,test_df)
    
    pre_train,test_y = predict_data(train_df,train_y,10,test_df,importance=1)
    
    if val==1:   
        print (len(train_y),sum(train_y))
        showresults(train_y,pre_train)
        showresults(val_y,test_y) 
        showfalse(ids,test_df,val_y,test_y)
        showtop(val_y,test_y,nums=1457)
        showtop(train_y,pre_train,nums=23260)
        #showtop(train_y,pre_train,nums=15485)
        #showprecision(val_y,test_y)
        #showprecision(train_y,pre_train)
        return ids,test_y,getbest(ids,test_y,th=0.4)
    else:
        showresults(train_y,pre_train)     
        showtop(train_y,pre_train,nums=24717)
        #showtop(train_y,pre_train,nums=16943)
        #showprecision(train_y,pre_train)
        return ids,test_y,getbest(ids,test_y,rank = 1490)
        #return ids,test_y,getbest(ids,test_y,th=0.4)
    
    
    
if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
    
    val=1
    ids,test_y,ans = predict_30(val,register,app,video,act)
    if val==0:
        print(len(ans))
        import time
        name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        submission = pd.DataFrame({'user_id': ans})
        submission.to_csv('30_submit'+name+'.csv', index=False, header = None)