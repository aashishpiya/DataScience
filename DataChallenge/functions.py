#!/usr/bin/env python
# coding: utf-8

# ## Zillow dataset functions

# In[5]:


def check_non_us(dataframe, country, country_code):
    """
    This function takes 3 arguments: the dataframe, the country name and country code and then
    prints the number of records in the dataframe which does not belong the the specified country 
    and country code in the dataframe
    """
    
    ##checking if there is any country other than United States 
    Country_count = 0
    for i in dataframe.country:

        if i !=country:
            Country_count+1

    print('There are {} countries other than United States in the dataset'.format(Country_count))   

    ##checking if there is any country other than US 
    code_count = 0
    for i in dataframe.country_code:

        if i !=country_code:
            code_count+1

    print('There are {} country_code other than US in the dataset'.format(code_count))   


# In[7]:


def prelim_preprocessing(dataframe, features):
    
    """
    prelim_preprocessing function takes in two arguemenst: first a dataframe and second a list of features 
    selected to be in dataframe based upon the assumptions and scope of analysis
    
    """
    ##print the feature names  
    print("Features selected after preliminary data preprocessing are: ")
    for i in features:
        print(i)
    
    ##creting a dataframe with only the required features
    dataframe = dataframe[features]
    
    #output the dataframe
    return dataframe


# In[9]:


def missing_value(dataframe):
    """
    missing_value function takes in one arguement dataframe and return a tables as well as a plotly bar chart 
    showing the total null values and as well as the percentage of null values in each columns in the dataframe
    """
    import pandas as pd
    
    missing = pd.DataFrame()
    missing['null_percentage'] = dataframe.isna().mean().round(4) * 100
    missing['total_null_values'] = dataframe.isna().sum()
   
    display(missing)
    
    import plotly.express as px

    fig = px.bar(missing, x=missing.index, y='null_percentage',
         hover_data=['total_null_values'], color='total_null_values',
          height=500, width=750)
    fig.show()


# In[ ]:


def zero_value(dataframe , col):
    """
    missing_value function takes in one arguement dataframe and return a tables as well as a plotly bar chart 
    showing the total null values and as well as the percentage of null values in each columns in the dataframe
    """
    import pandas as pd
    
    missing = pd.DataFrame()
    missing['null_percentage'] = dataframe.isna().mean().round(4) * 100
    missing['total_null_values'] = dataframe.isna().sum()
   
    display(missing)
    
    import plotly.express as px

    fig = px.bar(missing, x=missing.index, y='null_percentage',
         hover_data=['total_null_values'], color='total_null_values',
          height=500, width=750)
    fig.show()


# ## Zillow dataset functions

# In[1]:



def zillow_city_filter(df, State, Metro):

    """
    zillow_city_filter function takes in  three arguments and returns dataframe with values filterred based on the arguements provided.
    The first argument df is the dataframe from which the user intends to filter values
    The second argument State is the name of the state State from which the user intends to filter values
    The third argument Metro is the Metro name from which the user intends to filter values
    """
    
    # Filter data for New York City
    #we are using State and MEtro to filter city because the city columns has a lot of mistakes in it
    df = df[(df["State"] == State) & (df["Metro"] == Metro)]
    
    return df


# In[2]:



def zillow_preprocess(df, n):
    
    """
    zillow_preprocess function takes in  two arguments and returns dataframe with required values for time series forecasting. 
    The first argument df is the dataframe which needs to be proprocessed.
    The second argument n is the number of recent months required to be present in the dataframe.
    """
    import numpy as np
    
    # Dropping unnecessary columns..
    df = df.drop(["RegionID","City","State","Metro","CountyName","SizeRank"],axis=1)
    
    # Renaming RegionName column to Zipcode for better readability as we only need zipcode and date values for times series forecasting
    df.rename(columns={"RegionName": "zipcode"},inplace=True)

    #changing zipcode into string 
    df['zipcode']=df['zipcode'].astype(str)
    
    
    #n is the numbers of months that user requires to have in the dataframe
    n = n

    cols = [] #cols is an empty list which is appened with columns to be dropped
    
    #the first row in the df is the zipcode so the loop starts from second column
    # total number of columns minus total required columns gives the nof of columns to be dropped which is obtained from the following
    for i in np.arange(1,(len(df.columns)-n)):
        cols.append(i)
        
    
    #we only keep the required date columns
    df = df.drop(df.columns[cols],axis=1)
    
    return df
    


# In[3]:


def zillow_prophet(df, period):
    
    """
    zillow_prophet function takes in two arguements: first preprocessed dataframe and second the period over which forecasting is to be done and returns the dataframe with forecasted current value of the property as "currentPrice".
    Facebook prophet is used for the time series forecasting.f
    Facebook prophet is various robust and can handle missing values as well and perform very well with large scale of data as well.

    """
    
        #loop through each zipcode
        #for each zipcode a dataframe with 'ds' and 'y' column are created as required by Fb Prophet
        #'ds' is the date and 'y' is the value
        
    import numpy as np
    import pandas as pd
    from fbprophet import Prophet
    from fbprophet.plot import plot_plotly
    
    predval=[] # empty list to store the predicted current value of the property
        
    for i in range(len(df)):  
        
        ts = pd.melt(df.iloc[[i]], id_vars=['zipcode'], value_vars=list(df.columns[1:]))
        #melt create a dataframe with date and value columns for each zipcode
        #transforms row into columns similar to matrix transpose
        
        train_dataset = pd.DataFrame() #empty dataframe
        train_dataset['ds'] = pd.to_datetime(ts["variable"]) #fill the 'ds' with date which is 'variable' in ts 
        train_dataset['y'] = ts['value'] #fill the 'y' with median price which is 'value' in ts 
        
        prophet_basic = Prophet() #instantiate fb prophet
        prophet_basic.fit(train_dataset) #train the model with training data
        future = prophet_basic.make_future_dataframe(periods=period, freq='M') #create future timeperiod in dataframe
        forecast = prophet_basic.predict(future) #predict the future values for the future timeperiods
        pred = forecast['yhat'].astype(int).values[-1]  #get the latest predicted value
        predval.append(pred) #append the value in the predval list
    
    #creates a new column "currentPrice" in the df and filles it with the saved 'predval' 
    df['currentPrice'] = np.array(predval)
    
    print(future.tail(1))
    
    return df





