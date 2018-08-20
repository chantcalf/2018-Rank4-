# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 19:51:14 2018

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
from utils import is_active,showresults
from predict_30 import predict_30
from predict_29 import predict_29
from predict_1_23 import predict_1_23
from predict_24_28 import predict_24_28
from predict_1_28 import predict_1_28

register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')

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

val = 0
v4 = 0

ids29,test_y29,ans29 = predict_29(val,register,app,video,act)

ids30,test_y30,ans30 = predict_30(val,register,app,video,act)

if v4:
    ids1_23,test_y1_23,ans1_23 = predict_1_23(val,register,app,video,act)

    ids24_28,test_y24_28,ans24_28 = predict_24_28(val,register,app,video,act)
    ans = ans1_23+ans24_28+ans29+ans30
else:
    ids1_28,test_y1_28,ans1_28 = predict_1_28(val,register,app,video,act)
    ans = ans1_28+ans29+ans30

if val:
    
    
    val_df = register[register.register_day<24].reset_index(drop=True)
    val_y = is_active(val_df,24,30,app,video,act)

    val_df['Y'] = val_y['Y']
    
    trueans = list(val_df[val_df['Y']==1]['user_id'])
    val_y = val_df['Y']
    if v4:
        res1 = pd.DataFrame({'user_id':ids1_23,'Y1':test_y1_23})
        res2 = pd.DataFrame({'user_id':ids24_28,'Y1':test_y24_28})
        res3 = pd.DataFrame({'user_id':ids29,'Y1':test_y29})  
        res4 = pd.DataFrame({'user_id':ids30,'Y1':test_y30})
        res = res1.append(res2).append(res3).append(res4)
    else:
        res1 = pd.DataFrame({'user_id':ids1_28,'Y1':test_y1_28})
        res3 = pd.DataFrame({'user_id':ids29,'Y1':test_y29})  
        res4 = pd.DataFrame({'user_id':ids30,'Y1':test_y30})
        res = res1.append(res3).append(res4)
    val_df = pd.merge(val_df,res,on=['user_id'],how='left')
    test_y = val_df['Y1']
    

    showresults(val_y,test_y)
    n1 = len(val_df[(val_df['Y']==1) & (val_df['Y1']>=0.4)])
    print(n1)
    n = 0
    for i in ans:
        if i in trueans:
            n = n+1
    print(n,len(ans),len(trueans))
    p = n/len(ans)
    r = n/len(trueans)
    f1 = 2*p*r/(p+r+0.000001)
    print (p,r,f1)
else:

    print(len(ans))
    import time
    name = time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))
    submission = pd.DataFrame({'user_id': ans})
    submission.to_csv('ks_sub'+name+'.csv', index=False, header = None)