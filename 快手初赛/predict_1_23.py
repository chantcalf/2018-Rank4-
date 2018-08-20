# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 18:48:01 2018

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

def predict_1_23(val,register,app,video,act):
    path = '../data1/1_23/'
    
    
    
    def get_features(df,d1,d2):
        tapp = app[(app.day>=d1) & (app.day<=d2)]
        tact = act[(act.day>=d1) & (act.day<=d2)]
        tvideo = video[(video.day>=d1) & (video.day<=d2)]
        tapp.day = tapp.day - d1
        tact.day = tact.day - d1
        tvideo.day = tvideo.day - d1
        lastday = d2-d1
        #app
        df = docount(df,tapp,'app',['user_id'])
        #df = domin(df,tapp,'app',['user_id'],'day')
        df = domax(df,tapp,'app',['user_id'],'day')

        df['last_app_day'] = lastday - df['app$user_id_by_day_max']+1
        #df['app_day_gap'] = df['app$user_id_by_day_max']- df['app$user_id_by_day_min']+1
        df['app_day_missing'] = df['register_time'] - df['app$user_id#']
        df['app_mean#'] = df['app$user_id#']/df['register_time']
        del df['app$user_id#'], df['app$user_id_by_day_max']
        
        df = dovar(df,tapp,'app',['user_id'],'day')
        #df = domean(df,tapp[tapp.day>lastday-8],'app_last_8',['user_id'],'day')
        #df = dovar(df,tapp[tapp.day>lastday-8],'app_last_8',['user_id'],'day')
         
        for i in range(8):
            df = docount(df,tapp[tapp.day>=lastday-i],'app_last_'+str(i),['user_id'])
            if i>=3:
                df = domean(df,tapp[tapp.day>=lastday-i],'app_last_'+str(i),['user_id'],'day')
                df = dovar(df,tapp[tapp.day>=lastday-i],'app_last_'+str(i),['user_id'],'day')
        #df = docount(df,tapp[tapp.day>lastday-7],'app_last_7',['user_id'])        
        #df = docount(df,tapp[tapp.day>lastday-3],'app_last_3',['user_id']) 
        #df = docount(df,tapp[tapp.day==lastday],'app_last_1',['user_id'])
        
        
            
        gc.collect()
        #video
        df = docount(df,tvideo,'video',['user_id'])
        df = domin(df,tvideo,'video',['user_id'],'day')
        df = domax(df,tvideo,'video',['user_id'],'day')
        df = doiq(df,tvideo,'video',['user_id'],'day')
        df['last_video_day'] = lastday - df['video$user_id_by_day_max']+1
        df['first_video_day'] = lastday - df['video$user_id_by_day_min']+1
        df['video_day_gap'] = df['video$user_id_by_day_max']- df['video$user_id_by_day_min']+1
        #df['video_day_missing'] = df['register_time'] - df['video$user_id_by_day_iq']
        df['video_mean#'] = df['video$user_id#']/df['register_time']
        del df['video$user_id#'], df['video$user_id_by_day_max'],df['video$user_id_by_day_min']

        df = dovar(df,tvideo,'video',['user_id'],'day')
        df = domean(df,tvideo[tvideo.day>lastday-8],'video_last_8',['user_id'],'day')
        df = dovar(df,tvideo[tvideo.day>lastday-8],'video_last_8',['user_id'],'day')
         
        df = docount(df,tvideo[tvideo.day>lastday-8],'video_last_8',['user_id'])        
        #df = docount(df,tvideo[tvideo.day>lastday-3],'video_last_3',['user_id']) 
        #df = docount(df,tvideo[tvideo.day==lastday],'video_last_1',['user_id'])
        gc.collect()
        #act
        gp = tact.groupby(['user_id','day']).size().unstack()
        df = pd.merge(df,gp.max(1).rename('actcount_max').reset_index(),on=['user_id'],how='left')   
        df = pd.merge(df,gp.mean(1).rename('actcount_mean').reset_index(),on=['user_id'],how='left')
        df = pd.merge(df,gp.var(1).rename('actcount_var').reset_index(),on=['user_id'],how='left')
        
        df = docount(df,tact,'act',['user_id'])
        df = domin(df,tact,'act',['user_id'],'day')
        df = domax(df,tact,'act',['user_id'],'day')
        df = doiq(df,tact,'act',['user_id'],'day')
        #df['last_act_day'] = lastday - df['act$user_id_by_day_max']+1
        df['act_day_gap'] = df['act$user_id_by_day_max']- df['act$user_id_by_day_min']+1
        df['act_day_missing'] = df['register_time'] - df['act$user_id_by_day_iq']
        df['act_mean#'] = df['act$user_id#']/df['register_time']
        del df['act$user_id#']

        df = dovar(df,tact,'act',['user_id'],'day')
        #df = domean(df,tact[tact.day>lastday-8],'act_last_8',['user_id'],'day')
        #df = dovar(df,tact[tact.day>lastday-8],'act_last_8',['user_id'],'day')
         
        for i in range(8):
            df = docount(df,tact[tact.day>=lastday-i],'act_last_'+str(i),['user_id'])
            if i>=3:
                df = domean(df,tact[tact.day>=lastday-i],'act_last_'+str(i),['user_id'],'day')
                df = dovar(df,tact[tact.day>=lastday-i],'act_last_'+str(i),['user_id'],'day')
            
                gp = tact[tact.day>=lastday-i].groupby(['user_id','day']).size().unstack()
                df = pd.merge(df,gp.max(1).rename('act_last_'+str(i)+'_actcount_max').reset_index(),on=['user_id'],how='left')   
                df = pd.merge(df,gp.mean(1).rename('act_last_'+str(i)+'_actcount_mean').reset_index(),on=['user_id'],how='left')
                df = pd.merge(df,gp.var(1).rename('act_last_'+str(i)+'_actcount_var').reset_index(),on=['user_id'],how='left')
        #df = docount(df,tact[tact.day>lastday-7],'act_last_7',['user_id'])        
        #df = docount(df,tact[tact.day>lastday-3],'act_last_3',['user_id']) 
        #df = docount(df,tact[tact.day==lastday],'act_last_1',['user_id'])
        gc.collect()
        
        page_list = list(tact['page'].unique())
        for c in page_list: 
            df = docount(df,tact[tact['page']==c],'act_page='+str(c),['user_id'])
            df['act_page='+str(c)+'$user_id#'] = df['act_page='+str(c)+'$user_id#']/df['register_time']
        
        for c in page_list: 
            df = docount(df,tact[(tact['page']==c) & (tact.day>lastday-8)],'act_last_8_page='+str(c),['user_id'])
        for c in page_list: 
            df = docount(df,tact[(tact['page']==c) & (tact.day>lastday-3)],'act_last_3_page='+str(c),['user_id'])
    
        df['author_id'] = df['user_id']
        df = docount(df,tact,'act',['author_id'])
        df['act$author_id#'] = df['act$author_id#']/df['register_time']
        
        df = doiq(df,tact,'act',['user_id'],'author_id')  
        df['act$user_id_by_author_id_iq'] = df['act$user_id_by_author_id_iq']/df['register_time']

        df = doiq(df,tact,'act',['user_id'],'video_id')  
        df['act$user_id_by_video_id_iq'] = df['act$user_id_by_video_id_iq']/df['register_time']
        
        for i in range(8):
            df = doiq(df,tact[tact.day>=lastday-i],'act_last_'+str(i),['user_id'],'author_id')  
            df = doiq(df,tact[tact.day>=lastday-i],'act_last_'+str(i),['user_id'],'video_id')
        
        
        #action_list = list(tact['action_type'].unique())
        for c in [0,1,2,3,5]: 
            df = docount(df,tact[tact['action_type']==c],'action_type='+str(c),['user_id']);gc.collect()
            df['action_type='+str(c)+'$user_id#'] = df['action_type='+str(c)+'$user_id#']/df['register_time']
        for c in [0,1,2,3]: 
            df = docount(df,tact[(tact['action_type']==c) & (tact.day>lastday-8)],'act_last_8_action_type='+str(c),['user_id'])
        for c in [0,1,2,3]: 
            df = docount(df,tact[(tact['action_type']==c) & (tact.day>lastday-3)],'act_last_3_action_type='+str(c),['user_id'])
  
        ''' 
        def getmaxcontinuedays(s):
            s = np.array(s)
            ans = 0
            t = 0
            for i in s:
                if i>0:
                    t =  t+ 1
                else:
                    if t>ans:
                        ans = t
                    t = 0
            if t>ans:
                ans=t
            return ans
  
        gp = tapp.groupby(['user_id','day']).size().unstack()
        gp = gp.fillna(0)
        
        #print (gp)
        gp['app_max_continue_days'] = gp.apply(getmaxcontinuedays,axis=1)
        #print (gp)
        df = pd.merge(df,gp.reset_index()[['user_id','app_max_continue_days']],on=['user_id'],how='left') 
         
        gp = tact.groupby(['user_id','day']).size().unstack()
        gp = gp.fillna(0)
        
        #print (gp)
        gp['act_max_continue_days'] = gp.apply(getmaxcontinuedays,axis=1)
        #print (gp)
        df = pd.merge(df,gp.reset_index()[['user_id','act_max_continue_days']],on=['user_id'],how='left') 
        '''
         
         
        del df['author_id']
        gc.collect()
        
        
        return df
        
    def get_features_all(df,df1):
        lendf = len(df)
        df= df.append(df1)
        del df1
        gc.collect()
        
        
        #ccc = ['app_mean#', 'last_app_day', 'app$user_id_by_day_var', 'act$user_id_by_day_var', 'device_type', 'act$user_id_by_video_id_iq', 'app_last_4$user_id_by_day_var', 'act_last_0$user_id_by_author_id_iq', 'app_last_4$user_id#', 'register_type', 'act$user_id_by_day_max', 'actcount_var', 'act_last_0$user_id#', 'act_mean#', 'actcount_max', 'act_last_7$user_id_by_day_var', 'app_last_7$user_id_by_day_var', 'app_last_1$user_id#', 'action_type=2$user_id#', 'act_page=1$user_id#', 'action_type=0$user_id#', 'act_last_1$user_id#', 'app_last_5$user_id#', 'act$user_id_by_day_min', 'act_page=3$user_id#', 'act$user_id_by_day_iq', 'actcount_mean', 'act_last_0$user_id_by_video_id_iq', 'act_last_2$user_id_by_author_id_iq', 'app_last_7$user_id_by_day_mean', 'act_last_8_action_type=2$user_id#', 'act_last_8_page=1$user_id#', 'act_last_4$user_id_by_day_mean', 'act$user_id_by_author_id_iq', 'app_last_5$user_id_by_day_mean', 'act_day_gap', 'app_day_missing', 'act_last_7_actcount_var', 'action_type=3$user_id#', 'act_last_4_actcount_var', 'act_last_1$user_id_by_author_id_iq', 'app_last_3$user_id_by_day_var', 'act_last_3_actcount_var', 'act_last_1$user_id_by_video_id_iq', 'act_last_3_page=1$user_id#', 'act_page=2$user_id#', 'act_page=0$user_id#', 'act_last_3$user_id_by_video_id_iq', 'act_last_6_actcount_max', 'app_last_2$user_id#', 'act_last_2$user_id#', 'app_last_6$user_id_by_day_mean', 'act_last_6_actcount_var', 'act_last_3_action_type=2$user_id#', 'act_last_6$user_id_by_video_id_iq', 'act_last_7$user_id_by_video_id_iq', 'act_last_5_actcount_var', 'act_last_3$user_id#', 'act_last_7$user_id_by_author_id_iq', 'act_last_2$user_id_by_video_id_iq', 'act_last_8_page=3$user_id#', 'act_page=4$user_id#', 'act_last_7_actcount_max', 'act_last_5$user_id_by_day_var', 'act_last_7$user_id_by_day_mean', 'act_last_8_action_type=0$user_id#', 'act_last_3_actcount_max', 'app_last_5$user_id_by_day_var', 'app_last_0$user_id#', 'app_last_6$user_id_by_day_var', 'act_day_missing', 'action_type=1$user_id#', 'act_last_6_actcount_mean', 'act_last_6$user_id_by_day_mean', 'act_last_3$user_id_by_author_id_iq', 'act_last_8_page=0$user_id#', 'act_last_3_actcount_mean', 'act_last_6$user_id_by_author_id_iq', 'video_last_8$user_id_by_day_var', 'act_last_5$user_id_by_day_mean', 'act_last_3_page=0$user_id#', 'register_time', 'act_last_3$user_id_by_day_var', 'last_video_day', 'act_last_6$user_id_by_day_var', 'act_last_4$user_id#', 'act_last_5$user_id_by_author_id_iq', 'act_last_4$user_id_by_author_id_iq', 'first_video_day', 'video_mean#', 'act_last_8_action_type=3$user_id#', 'act_last_3_action_type=0$user_id#', 'act_last_3_page=3$user_id#', 'app_last_4$user_id_by_day_mean', 'app_last_3$user_id#', 'act_last_8_page=4$user_id#', 'act_last_6$user_id#', 'act_last_3$user_id_by_day_mean', 'act_last_7$user_id#', 'act_last_5$user_id_by_video_id_iq', 'video_last_8$user_id_by_day_mean', 'act_last_4$user_id_by_day_var', 'act_last_7_actcount_mean', 'app_last_7$user_id#', 'video$user_id_by_day_var', 'act_last_5_actcount_max', 'act_last_3_page=4$user_id#', 'act_last_8_page=2$user_id#', 'act_last_5$user_id#', 'act_last_4_actcount_max', 'video$user_id_by_day_iq', 'act_last_4$user_id_by_video_id_iq', 'act_last_5_actcount_mean', 'act$author_id#', 'app_last_6$user_id#', 'act_last_4_actcount_mean', 'act_last_8_action_type=1$user_id#', 'video_day_gap', 'act_last_3_action_type=1$user_id#', 'act_last_3_page=2$user_id#', 'app_last_3$user_id_by_day_mean', 'action_type=5$user_id#', 'video_last_8$user_id#', 'act_last_3_action_type=3$user_id#']
        #for i in range(100,124):
        #    del df[ccc[i]]
        
        del df['user_id']

 
        df1 = df[lendf:]
        df = df[:lendf]
        return df,df1
        
    df1 = register[register.register_day<10]
    df1['register_time'] = 17-register.register_day
    df2 = register[register.register_day<17]
    df2['register_time'] = 24-register.register_day


    
    test_df = register[register.register_day<24]
    test_df['register_time'] = 31-test_df.register_day

    
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
        df1 = get_features(df1,1,16)
        df1.to_csv(path+'df1.csv',index=False)
    
    if os.path.exists(path+'df2.csv'):
        df2=pd.read_csv(path+'df2.csv')
    else:
        df2 = get_features(df2,1,23)
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
            test_df = get_features(test_df,1,30)
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
        showtop(val_y,test_y,nums=10705)
        return ids,test_y,getbest(ids,test_y,rank=10705)   
    else:
        showresults(train_y,pre_train)     
        showtop(train_y,pre_train,nums=16449)
        return ids,test_y,getbest(ids,test_y,th=0.4)
        #return ids,test_y,getbest(ids,test_y,rank=16590)     
        
if __name__=='__main__':
    register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
    app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
    video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')
    act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
    
    val = 0
    #ans = list(register[register.register_day<=23]['user_id'])
    ids,test_y,ans = predict_1_23(val,register,app,video,act)      
    if val==0:
        print(len(ans))
        import time
        name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
        submission = pd.DataFrame({'user_id': ans})
        submission.to_csv('1_23_submit'+name+'.csv', index=False, header = None)
        