
# coding: utf-8

# ## Loading 1M rows of the dataset

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('train.csv',nrows=1000000)


# ## Explore and Analyse the DATA

# In[3]:


df.dtypes


# In[4]:


#We don't need such big datatypes to represent our dataset efficiently
types1 = {'fare_amount': 'float32',
         'pickup_longitude': 'float32',
         'pickup_latitude': 'float32',
         'dropoff_longitude': 'float32',
         'dropoff_latitude': 'float32',
         'passenger_count': 'uint8'}


# In[5]:


df = df.astype(types1)


# In[6]:


df.dtypes


# In[7]:


df.head()


# #### Missing Value Treatment

# In[8]:


df.describe()


# In[9]:


#Check how many rows have null values
df.isnull().sum()


# In[10]:


#Drop null since it is negligible in this case 
df.dropna(inplace=True)


# In[11]:


df.describe()


# #### Outlier treatment

# In[12]:


print(f"There are {len(df[df['fare_amount'] < 0])} negative fares.")
print(f"There are {len(df[df['fare_amount'] == 0])} $0 fares.")
print(f"There are {len(df[df['fare_amount'] > 100])} fares greater than $100.")


# In[13]:


df = df[(df['fare_amount'] > 0) & (df['fare_amount'] < 100)]


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df['passenger_count'].value_counts().plot.bar(color = 'b', edgecolor = 'k')
plt.title('Passenger Counts'); plt.xlabel('Number of Passengers'); plt.ylabel('Count')


# In[16]:


df = df[(df['passenger_count'] > 0) & (df['passenger_count'] < 6)]


# In[17]:


for col in ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']:
    print(f'PERCENTILES OF {col.capitalize():17}: 2.5% = {round(np.percentile(df[col], 2.5), 2):5}\t\t97.5% = {round(np.percentile(df[col], 97.5), 2)}\n')


# In[18]:


df = df.loc[df['pickup_latitude'].between(40, 44)]
df = df.loc[df['pickup_longitude'].between(-75, -72)]
df = df.loc[df['dropoff_latitude'].between(40, 44)]
df = df.loc[df['dropoff_longitude'].between(-75, -72)]


# In[19]:


df.describe()


# ## Feature Engineering

# In[20]:


from math import sin, cos, sqrt, atan2, radians


# In[21]:


#Get useable date for feature engineering
df['pickup_datetime'] = df['pickup_datetime'].str.replace(" UTC", "")
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'], format='%Y-%m-%d %H:%M:%S')


# #### From Timestamp we can get below new features

# In[22]:


#Getting interger numbers from the pickup_datetime
df["hour"] = df.pickup_datetime.dt.hour
df["weekday"] = df.pickup_datetime.dt.weekday
df["month"] = df.pickup_datetime.dt.month
df["year"] = df.pickup_datetime.dt.year


# #### Distance is aslo another crucial attribute

# In[23]:


#Quicker but slightly less accurate
def dist_calc(df):
    R = 6373.0
    for i,row in df.iterrows():

        lat1 = radians(row['pickup_latitude'])
        lon1 = radians(row['pickup_longitude'])
        lat2 = radians(row['dropoff_latitude'])
        lon2 = radians(row['dropoff_longitude'])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        distance = R * c
        df.at[i,'distance'] = distance


# In[24]:


dist_calc(df)


# #### Another concept Hotspot proximity can serve as useful feature

# In[25]:


#Function for distance calculation between coordinates as mapped variables
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    #Define earth radius (km)
    R_earth = 6371
    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = map(np.radians,
                                                             [pickup_lat, pickup_lon, 
                                                              dropoff_lat, dropoff_lon])
    #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon
    
    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    
    return 2 * R_earth * np.arcsin(np.sqrt(a))


# In[26]:


#Function for calculating distance between newly obtained distances from the hotspots.
def add_airport_dist(dataset):
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    
    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']
    
    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], dropoff_lat, dropoff_lon) 
    
    dataset['jfk_dist'] = pd.concat([pickup_jfk, dropoff_jfk], axis=1).min(axis=1)
    dataset['ewr_dist'] = pd.concat([pickup_ewr, dropoff_ewr], axis=1).min(axis=1)
    dataset['lga_dist'] = pd.concat([pickup_lga, dropoff_lga], axis=1).min(axis=1)
    
    return dataset


# In[27]:


#Run the functions to add the features to the dataset
df = add_airport_dist(df)


# In[28]:


df.dtypes


# In[29]:


#We don't need such big datatypes to represent our dataset efficiently plus it is computationally costly
types2 = {'hour': 'uint8',
         'weekday': 'uint8',
         'month': 'uint8',
         'year': 'uint8',
         'distance': 'float32'}


# In[30]:


df = df.astype(types2)
df.dtypes


# In[31]:


df.head()


# ## Attribute selection for regression

# In[32]:


import seaborn as sns


# #### Correlation between attributes

# In[33]:


#Plot heatmap of value correlations
plt.figure(figsize=(15,8))
sns.heatmap(df.drop(['key','pickup_datetime'],axis=1).corr(),annot=True,fmt='.4f')


# #### Scatter plot of 1K records

# In[34]:


d = df[['fare_amount','distance','jfk_dist','lga_dist']]
d = d[:1000]


# In[35]:


sns.set(style='whitegrid', context = 'notebook')
sns.pairplot(d,size=2.5)


# In[36]:


X = df[['distance','jfk_dist']]
y = df[['fare_amount']]


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size=0.2, random_state=42)


# ### Ordinary least squares

# In[39]:


w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_training.T, X_training)), X_training.T), y_training)
print(w_OLS)


# In[40]:


y_predict = np.matmul(X_testing, w_OLS).round(decimals = 2)


# In[41]:


from sklearn.metrics import mean_squared_error


# In[42]:


print('Mean Squared Error using Ordinary Least Square for two variables: %.2f' % mean_squared_error(y_testing, y_predict))


# In[43]:


y_predict1 = np.matmul(X_training, w_OLS).round(decimals = 2)


# In[44]:


print('Mean Squared Error using Ordinary Least Square for two variables ( Training Error ): %.2f' % mean_squared_error(y_training, y_predict1))


# ### Linear Regression

# In[45]:


from sklearn.linear_model import LinearRegression


# In[46]:


lr = LinearRegression()
lr_predict = lr.fit(X_training,y_training)


# In[47]:


lin_predict = lr.predict(X_testing)
print('Mean Squared Error using Linear Regression for two variables: %.2f' % mean_squared_error(y_testing, lin_predict))


# In[48]:


lr_predict.score(X_testing,y_testing)


# ## Now lets take it to other way around

# In[49]:


y_data = df[['fare_amount']]
X_data = df.drop(['key','fare_amount','pickup_datetime'],axis=1)


# In[50]:


X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)


# In[51]:


w_OLS = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.T, X_train)), X_train.T), y_train)
print(w_OLS)


# In[52]:


y_pred = np.matmul(X_test, w_OLS).round(decimals = 2)


# In[53]:


print('Mean Squared Error using Ordinary Least Square for all possible attributes: %.2f' % mean_squared_error(y_test, y_pred))


# In[54]:


lr_predictions = lr.fit(X_train,y_train)


# In[55]:


linpred = lr.predict(X_test)
print('Mean Squared Error using Linear Regression for all possible attributes: %.2f' % mean_squared_error(y_test, linpred))


# In[56]:


lr_predictions.score(X_test,y_test)


# ### XGBoost

# In[57]:


import xgboost as xgb


# In[58]:


#Define a XGB model and parameters
def XGBoost(X_train,X_test,y_train,y_test):
    dtrain = xgb.DMatrix(X_train,label=y_train)
    dtest = xgb.DMatrix(X_test,label=y_test)

    return xgb.train(params={'objective':'reg:linear','eval_metric':'mae'}
                    ,dtrain=dtrain,num_boost_round=400, 
                    early_stopping_rounds=30,evals=[(dtest,'test')])


# In[59]:


#Fit data and optimise the model, generate predictions
xgbm = XGBoost(X_train,X_test,y_train,y_test)


# In[60]:


XGBPredictions = xgbm.predict(xgb.DMatrix(X_test), ntree_limit = xgbm.best_ntree_limit)


# In[61]:


print('Mean Squared Error using XGBoost for all possible attributes: %.2f' % mean_squared_error(y_test, XGBPredictions))


# # Apparently ensemble approache (i.e. XGBoost) improved the performance over baseline approaches (i.e.Ordinary least squares and Linear Regression) significantly.
