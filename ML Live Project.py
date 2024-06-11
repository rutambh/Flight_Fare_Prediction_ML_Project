#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as  plt


# In[2]:


#To Ignore the warnings/Remove the Warnings
import warnings
warnings.filterwarnings("ignore") 


# In[3]:


#Import the File
data = pd.read_excel(r"D:\Work\ML\Live Project\ML Live Flight Fare Resourses16963295320.xlsx")


# In[4]:


#Read the file
data


# In[5]:


#Reads the first rows of data
data.head() 


# In[6]:


#-----------------------------------------------------------------------------------------------------------------------


# In[7]:


#--------------------------------------------Data Cleaning-------------------------------------------------------------#


# In[8]:


#-----------------------------------------------------------------------------------------------------------------------


# In[9]:


#Remove Unncessary columns
data.drop(["Additional_Info","Route"],axis=1,inplace=True)


# In[10]:


#Find Blank values in all columns
data.isnull().sum()


# In[11]:


#Find which row has the NULL value
data[data["Total_Stops"].isnull()]


# In[12]:


#Find the Index of that row
data[data["Total_Stops"].isnull()].index


# In[13]:


#Drop that particular row
data.drop([9039],inplace=True)


# In[14]:


#Check again for NULL Values, Validation
data.isnull().sum()


# In[15]:


#Know the data types, Mainly  for data  type
data.info() 


# In[16]:


#COLUMN : Date of Journcey---


# In[17]:


#Changed data type to date and time
pd.to_datetime(data["Date_of_Journey"])


# In[18]:


#Seperated Month and day and created new column
data["month"]=pd.to_datetime(data["Date_of_Journey"]).dt.month
data["date"]=pd.to_datetime(data["Date_of_Journey"]).dt.day


# In[19]:


#COLUMN : Dep_Time and Arrival_time------


# In[20]:


pd.to_datetime(data["Dep_Time"])
#This will show today's  dates because  we don't have date in the data,
# We  have only time in this column


# In[21]:


pd.to_datetime(data["Arrival_Time"])
#Here we have the date in the data so its showing the date


# In[22]:


#Seperated Hour and minutes from the Dep time and Arrival_Time columns
data["d_hour"]=pd.to_datetime(data["Dep_Time"]).dt.hour
data["d_min"]=pd.to_datetime(data["Dep_Time"]).dt.minute
data["A_hour"]=pd.to_datetime(data["Arrival_Time"]).dt.hour
data["A_min"]=pd.to_datetime(data["Arrival_Time"]).dt.minute


# In[23]:


data


# In[24]:


#COLUMN : Duration------


# In[25]:


data["Duration"]
#Challenges :  
#we need to seperate Hours & minutes,but by simple Index, both will show  in one only
#Some values has Hours and some values have minutes, we need to fix that


# In[26]:


lis=data["Duration"]


# In[27]:


#If data is like 2h then in column it should be 2h 0m
#If data is like 2m then in column it should be 0h 2m 

new_lis=[]
for i in lis:
    if len(i.split(" "))==1: #Split by Space
        if "m" in i:
            i="0h "+i
            print(i)   #for 0h 2m
        else:
            i=i+" 0m" #for 2h 0m
    new_lis.append(i) #Insert the data in the list


# In[28]:


new_lis
#New list created where all values are same, some values didnt have hours/minutes, 
#So now it will show 0


# In[29]:


#Save thed data in the column
data["Duration"]=new_lis


# In[30]:


data


# In[31]:


#Now remove words "h" and "m" from the data so we can have interger data only
data["dur_h"]=data["Duration"].str.split(" ").str[0].replace("[h]","",regex=True)
data["dur_m"]=data["Duration"].str.split(" ").str[1].replace("[m]","",regex=True)                                       


# In[32]:


data


# In[33]:


#Remove Data cleaned Columns
data.drop(["Date_of_Journey","Dep_Time","Arrival_Time","Duration"],axis=1,inplace=True)


# In[34]:


data


# In[35]:


#COLUMN : Total_Stops------


# In[36]:


data["Total_Stops"].unique()


# In[37]:


#Replace data with Integer
data["Total_Stops"].replace(['non-stop', '2 stops', '1 stop', '3 stops', '4 stops'],[0,2,1,3,4],inplace=True)


# In[38]:


#COLUMN : Airline,Source,Destination---------
#We have just labels in this columns so we can use "Apply" function


# In[39]:


#Import ML Label Encoder
from sklearn.preprocessing import LabelEncoder


# In[40]:


#Saving the Encoder
enc=LabelEncoder()


# In[41]:


#Apply function will apply same logic to all metioned columns
data[["Airline","Source","Destination"]]=data[["Airline","Source","Destination"]].apply(enc.fit_transform)


# In[42]:


data


# In[43]:


#Validate that all data types are same for all columns
data.info()


# In[44]:


#Change  Data type of remaining ones, Object ones


# In[45]:


data[["dur_h","dur_m"]]=data[["dur_h","dur_m"]].astype(int)


# In[46]:


#Validate that all data types are same for all columns
data.info()


# In[47]:


#-----------------------------------------------------------------------------------------------------------------------


# In[48]:


#--------------------------------------------ML DATA TRAINING-------------------------------------------------------------#


# In[49]:


#-----------------------------------------------------------------------------------------------------------------------


# In[50]:


x = data.drop("Price", axis=1)
y = data["Price"]


# In[51]:


#Spliting Data for Training and Testing
from sklearn.model_selection import train_test_split


# In[52]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)


# In[53]:


#Importing Linear Regression Model
from sklearn.linear_model import LinearRegression


# In[54]:


model = LinearRegression()
model.fit(x_train,y_train)


# In[55]:


model.score(x_train,y_train)


# In[56]:


model.score(x_test,y_test)


# In[57]:


#Comparing all Regressor Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[59]:


lis=[DecisionTreeRegressor,KNeighborsRegressor,RandomForestRegressor,GradientBoostingRegressor]


# In[60]:


for i in lis:
    model=i()
    model.fit(x_train,y_train)
    print(i)
    print(model.score(x_train,y_train))
    print(model.score(x_test,y_test))


# In[61]:


#Random Forest - has a high training score and the highest testing score among the four models, 
#suggesting that it generalizes well to new data without overfitting.


# In[70]:


#Another way of Comparing is Comparing all their means


# In[62]:


from sklearn.model_selection import KFold, cross_val_score


# In[65]:


kf=KFold(n_splits=5)


# In[66]:


score=[]
for i in lis:
    score.append(cross_val_score(i(),x,y,cv=kf))


# In[67]:


score


# In[68]:


import numpy as np


# In[69]:


for i in range(len(score)):
    print(lis[i])
    print(score[i].mean())


# In[71]:


rf = RandomForestRegressor()


# In[72]:


from sklearn.model_selection import RandomizedSearchCV


# In[77]:


n_estimators=[400,500,600,700,800,900,1000]


# In[78]:


n_depth=[2,3,4,5]
random=[10,20,30,40]
crireia=["squared_error","absolute_error"]


# In[79]:


dic={"n_estimators":n_estimators,"max_depth":n_depth,"random_state":random,
     "criterion":crireia}


# In[82]:


random_model=RandomizedSearchCV(estimator=rf,param_distributions=dic,cv=kf,n_iter=20,
                  n_jobs=2,verbose=2)


# In[83]:


random_model.fit(x,y)


# In[85]:


random_model.best_params_


# In[86]:


model=RandomForestRegressor(random_state=30,n_estimators=1000,max_depth=5)
model.fit(x_train,y_train)
print(model.score(x_train,y_train))
print(model.score(x_test,y_test))


# In[ ]:




