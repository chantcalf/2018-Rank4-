# -*- coding: utf-8 -*-
"""
Created on Mon May 28 21:02:39 2018

@author: chantcalf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import is_active

register = pd.read_csv('../user_register_log.txt',sep="\t",names = ['user_id','register_day','register_type','device_type'])
app = pd.read_csv('../app_launch_log.txt',names=['user_id','day'],sep='\t',)
act = pd.read_csv('../user_activity_log.txt',names=['user_id','day','page','video_id','author_id','action_type'],sep='\t')
video = pd.read_csv('../video_create_log.txt',names=['user_id','day'],sep='\t')

ans = list(register[(register.register_day>=24)&(register.register_day<=28)]['user_id'])
print (len(ans))
#aaa =[]
#print (np.mean(aaa),np.var(aaa))

'''
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
print (appadd.shape)
print (app.shape)
app = app.append(appadd)
print (app.shape)
'''
'''
df = register
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
  
gp = app.groupby(['user_id','day']).size().unstack()
gp = gp.fillna(0)

#print (gp)
gp['max_continue_days'] = gp.apply(getmaxcontinuedays,axis=1)
#print (gp)
df = pd.merge(df,gp.reset_index()[['user_id','max_continue_days']],on=['user_id'],how='left')   
print (df)
'''

#plt.subplot(2,1,1)
#gp = register.groupby(['register_day']).size().rename('reg').reset_index()
#plt.plot(range(30),gp['reg'],'o')

#plt.subplot(2,1,2)
#gp = register[register['device_type']==3].groupby(['register_day']).size().rename('reg3').reset_index()
#plt.plot(range(30),gp['reg3'],'*')


'''
app = pd.merge(app,register,on=['user_id'],how='left')
app = app[app['day']!=app['register_day']]
app = app[app['register_type']!=1]
gp1 = app.groupby(['day']).size().rename('regtype1').reset_index()
print(gp1)
#plt.plot(range(29),gp1['regtype1'],'o')
'''

'''
x = register[register['register_type']==1]
gp1 = register.groupby(['register_day']).size().rename('regtype1').reset_index()
print(gp1)
plt.plot(range(30),gp1['regtype1'],'o')
'''


'''
print(register.shape)
test_df = register[(register.register_day>=17) & (register.register_day<=22)]
print(test_df.shape)
'''


#print (register['user_id'].head(10))
#for i in range(1,31):
#    temp = app[app.day==i]
#    print (len(temp),temp['user_id'].nunique())
'''
plt.subplot(4,1,1)
gp1 = register.groupby(['register_day']).size().rename('dayreg').reset_index()
plt.plot(range(30),gp1['dayreg'],'o')

plt.subplot(4,1,2)
activerate = []
for i in range(1,24):
    us = register[register.register_day==i]
    y = is_active(us,i+1,i+7,app,video,act)
    y = list(y['Y'])
    activerate.append(sum(y)/us.shape[0])

am = np.mean(activerate)
for i in range(7):    
    activerate.append(am)

plt.plot(range(30),activerate,'*')
'''
'''
plt.subplot(4,1,2)
appcount = []
for i in range(1,31):
    appcount.append(app[app['day']==i]['user_id'].nunique())

plt.plot(range(30),appcount,'*')

plt.subplot(4,1,3)
videocount = []
for i in range(1,31):
    videocount.append(video[video['day']==i]['user_id'].nunique())

plt.plot(range(30),appcount,'*')

plt.subplot(4,1,4)
actcount = []
for i in range(1,31):
    actcount.append(act[act['day']==i]['user_id'].nunique())

plt.plot(range(30),appcount,'*')

'''