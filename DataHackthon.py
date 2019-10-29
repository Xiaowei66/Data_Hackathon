
# coding: utf-8

# In[3]:


# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd
import re

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# In[4]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import PolynomialFeatures


# In[5]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
li = []
for filename in ['train.csv','test.csv']:
    df = pd.read_csv(filename,index_col=None, header=0)
    li.append(df)
    
train_df = pd.concat(li, axis=0, ignore_index=True,sort=False)


# In[6]:


train_df2 = train_df.replace(np.nan, '', regex=True)


# In[7]:


gender_list=[]
age_list=[]
for s in train_df2['sex and age']:
    matchObj = re.match( r'^(.*), (.*)$', s, re.M|re.I)
    if matchObj:
        if re.match(r"[FMfm][a-z]*",(matchObj.group(1)) ):
            
            gender_list.append(matchObj.group(1))
            if re.match(r"\d+",matchObj.group(2)):
                age_list.append( int(float(matchObj.group(2))) )
            else:
                age_list.append(50)
           
        else:
            
            gender_list.append(matchObj.group(2))
            if re.match(r"\d+",matchObj.group(1)):
                age_list.append( int(float(matchObj.group(1))) ) 
            else:
                age_list.append(50)
            
    else:
        gender_list.append(0)
        age_list.append(0)
#print ("Finishing")
train_df2["gender"]=gender_list
train_df2["age"]=age_list


# In[8]:


job_list=[]
area_list=[]
#for s in train_df2['job_status and living_area']:
#for s in ['Remote?government','Remote?government','?government','Remote?','?city','R?government',""]:
for s in train_df2['job_status and living_area']:

    matchObj = re.match( r'^(.*)\?(.*)$', s, re.M|re.I)
    if matchObj:
        if re.match(r"^[cCRr][^\?]*",(matchObj.group(1)) ):
            
            # area exists
            area = matchObj.group(1)
            
            # check whether job exists
            if not matchObj.group(2):
                job = 0
            else:
                job = matchObj.group(2)
            
        else:
            
            # check whether group(1) is null, if not null it's job
            if matchObj.group(1):
                job = matchObj.group(1)
                
                # check whether group(2) is null
                if not matchObj.group(2):
                    area = 0
                else:
                    area = matchObj.group(2) 
                
            #if group(1) is null
            else: 
                # check whether group(2) is City?
                if re.match(r"^[cCRr][^\?]*",(matchObj.group(2)) ):
                    area = matchObj.group(2)
                    job = 0
                else:
                    job = matchObj.group(2)
                    area = 0
    else:
        job = 0
        area = 0
    
    job_list.append(job)
    area_list.append(area)


train_df2["job"]=job_list
train_df2["area"]=area_list


# In[9]:


step1_clear_df = train_df2.drop(['sex and age','job_status and living_area'],1)


# In[10]:


int_smoker_status= []
for s in train_df2['smoker_status']:
    if s == "non-smoker":
        int_smoker_status.append(1)
    elif s == "quit":
        int_smoker_status.append(2)
    
    elif s == "active_smoker":
        int_smoker_status.append(3)
    
    else:
        int_smoker_status.append(0)


# In[11]:


# drop smoker_status
step2_clear_df = step1_clear_df.drop(['smoker_status'],1)
step2_clear_df['smoker_status']=int_smoker_status


# In[12]:


# get the mean of smoker_status
mean_smokerstatus = int((step2_clear_df['smoker_status']).mean())
second_smker_list = step2_clear_df['smoker_status'].replace(0,mean_smokerstatus)
# replace the 0 by mean

step3_clear_df = step2_clear_df.drop(['smoker_status'],1)
step3_clear_df['smoker_status']=second_smker_list


# In[13]:


# get the mean of gender
ggg = step3_clear_df['gender']
ggg.value_counts().head(1)

second_gender_list = step3_clear_df['gender'].replace(0,"F")
step4_clear_df = step3_clear_df.drop(['gender'],1)
step4_clear_df['gender']=second_gender_list


# In[14]:


# ge the mean of age
mean_age = int((step4_clear_df['age']).mean())

second_age_list = (step4_clear_df['age'].replace(0,mean_age))
step5_clear_df = step4_clear_df.drop(['age'],1)
step5_clear_df['age']=second_age_list


# In[15]:


# get the mean of BMI
new_BMI=[]
for i in(step5_clear_df['BMI']):
    try:
        if re.match(r"^[\d\.]+$",i):
            #print(i)
            new_BMI.append(int(float(i)))
        else:
            new_BMI.append(0)
    except ValueError:
        new_BMI.append(0)


# In[16]:


step6_clear_df = step5_clear_df.drop(['BMI'],1)
step6_clear_df['BMI']=new_BMI


# In[17]:


mean_bmi = int((step6_clear_df['BMI']).mean())
second_bim_list = (step6_clear_df['BMI'].replace(0,mean_bmi))

step7_clear_df = step6_clear_df.drop(['BMI'],1)
step7_clear_df['BMI']=second_bim_list


# In[18]:


new_blood=[]
for i in(step7_clear_df['average_blood_sugar']):
    try:
        new_blood.append(int(float(i)))
    except ValueError:
        new_blood.append(0)

step8_clear_df = step7_clear_df.drop(['average_blood_sugar'],1)
step8_clear_df['average_blood_sugar']=new_blood


# In[19]:


mean_blood = int((step8_clear_df['average_blood_sugar']).mean())
second_blood_list = (step8_clear_df['average_blood_sugar'].replace(0,mean_blood))

step9_clear_df = step8_clear_df.drop(['average_blood_sugar'],1)
step9_clear_df['average_blood_sugar']=second_blood_list


# In[20]:


new_BP=[]
for i in(step7_clear_df['high_BP']):
    #print(type(i))
    try:
        new_BP.append(int(i))
    except ValueError:
        new_BP.append(0)
        
new_heart_condition=[]
for i in(step7_clear_df['heart_condition_detected_2017']):
    #print(type(i))
    try:
        new_heart_condition.append(int(i))
    except ValueError:
        new_heart_condition.append(0)
        
new_married=[]
for i in(step7_clear_df['married']):
    #print(type(i))
    try:
        new_married.append(int(i))
    except ValueError:
        new_married.append(0)
        


# In[21]:


step10_clear_df = step9_clear_df.drop(['high_BP','heart_condition_detected_2017','married'],1)
step10_clear_df['high_BP']=new_BP
step10_clear_df['heart_condition_detected_2017']=new_heart_condition
step10_clear_df['married']=new_married


# In[22]:


aaa = step10_clear_df.drop(['TreatmentA','TreatmentB','TreatmentC','TreatmentD','id'],1)
n_job=[]
for i in aaa['job']:
    
    if type(i) == int:
        n_job.append(1)
    else:
        i.strip()
        #print(i)
        if re.search(r"go|G",i):
            #print (1)
            n_job.append(1)
        elif re.search(r"pr|PR",i):
            #print (2)
            n_job.append(2)
        elif re.search(r"bu|BUS|Biz|biz",i):
            #print (3)
            n_job.append(3)
        elif re.search(r"pa|PA",i):
            #print (4)
            n_job.append(4)
        elif re.search(r"un",i):
            n_job.append(4)
            #print (5)
        else:
            n_job.append(1)
n_area=[]
for i in aaa['area']:
    
    if type(i) == int:
        n_area.append(1)
    else:
        i.strip()
        #print(i)
        if re.search(r"Ci|ci|c|C",i):
            #print (1)
            n_area.append(1)
        elif re.search(r"Re|re|R|r",i):
            #print (2)
            n_area.append(2)
        else:
            n_area.append(1)
            #print(i)
n_gender=[]
for i in aaa['gender']:
    if re.search(r"F|f",i):
        n_gender.append(1)
    
    elif re.search(r"M|m",i):
        n_gender.append(2)
    else:
        n_gender.append(2)
        #print(i)


# In[23]:


bbb = aaa.drop(['job','area','gender'],1)
bbb['job']=n_job
bbb['area']=n_area
bbb['gender']=n_gender


# In[24]:



train_p = bbb[:34872]

#train_p

train_y_p = train_p['stroke_in_2018']
#train_y_p

new_stroke=[]
count = 0
for i in train_y_p:
    try:
        new_stroke.append(int(i))
    except ValueError:
        
        if re.match(r"N|n",i):
            #print(i)
            new_stroke.append(0)
        else:
            new_stroke.append(-1)
            count +=1
        
print (count)

train_p = train_p.drop(['stroke_in_2018'],1)
train_p['stroke_in_2018']=new_stroke


# In[25]:


mmm = train_p[train_p['stroke_in_2018'] != -1 ]


# In[26]:


train_y = mmm['stroke_in_2018']
train_x = mmm.drop(['stroke_in_2018'],1)


# In[27]:


train_y.value_counts()


# In[28]:


test_x = bbb[34872:]
test_x = test_x.drop(['stroke_in_2018'],1)
#test_x


# In[29]:


count = 0
for lambd in [0.03,0.04,0.05,0.06,0.07,0.08,1,5,10,50,100]:
    for poly_degree in range(1,8):
        # Create polynomial features
        poly = PolynomialFeatures(poly_degree)
        X_train_poly = poly.fit_transform(train_x)
        X_test_poly = poly.fit_transform(test_x)

        clf = LogisticRegression(C=1/lambd, class_weight='balanced')        
        clf.fit(X_train_poly, train_y)
        train_predictions = clf.predict(X_test_poly)
        #acc = accuracy_score(train_x, train_y)
        a = pd.Series(train_predictions)
        #print(acc)
        count = count+1
        path = '/Users/Luyao/Desktop/DataHack/new/sample{}.csv'.format(count)
        f = open(path,'w')
        
        ID = test_df["id"]
        ID = pd.DataFrame(ID)
        ID['stroke_in_2018']=a
        ID.to_csv(path,index=False)

