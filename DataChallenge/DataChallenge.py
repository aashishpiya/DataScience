#!/usr/bin/env python
# coding: utf-8

# # CapitalOne Data Challenge

# ## Business Problem

# <font size="4">*You are consulting for a real estate company that has a niche in purchasing properties to rent out short-term as part of their business model specifically within New York City.  The real estate company has already concluded that two bedroom properties are the most profitable; however, they do not know which zip codes are the best to invest in. The real estate company has engaged your firm to build out a data product and provide your conclusions to help them understand which zip codes would generate the most profit on short term rentals within New York City.*</font>
# 

# ## Assumptions

# - Investors have already identified that 2 bedroom properties are most profitable and want to invest in only those properties
# - Investers are interested to buy properties only in New York City
# - The occupancy rate is assumed to be 75%   
# - The investor will pay for the property in cash (i.e. no mortgage/interest rate will need to be accounted for).
# - The time value of money discount rate is 0% (i.e. $1 today is worth the same 100 years from now).
# - All properties and all square feet within each locale can be assumed to be homogeneous.
# - Only the monthly date will be considered (i.e. all the data will be consolidated into monthly data wherever daily data is provided)

# In[ ]:





# ### install and import required packages

# In[1]:


#unccomment the following line to install required packages
#pip install -r requirements.txt


# In[1]:


#import all the required packages

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import dash
import dash_core_components as dcc
import dash_html_components as html
import gzip


from fbprophet import Prophet
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


import missingno as msno 
import plotly.express as px


# ### importing the functions created for the purpose of this analysis

# In[2]:


import functions


# In[ ]:





# ### Load the dataset
# 

# In[3]:


airbnb = pd.read_csv('listings.csv',low_memory=False) #load the airbnb revenue dataset
zillow = pd.read_csv('Zip_Zhvi_2bedroom.csv',low_memory=False) # load the zillow cost dataset


# .

# ### Quick view of both the dataset

# ### Airbnb data (Revenue data)

# In[4]:



airbnb.head(3)


# ### No of rows and columns in the dataset

# In[5]:


# no of rows and columns present in the airbnb data 
airbnb_rows = airbnb.shape[0]
airbnb_columns = airbnb.shape[1]
print('There are {} rows and {} columns in the airbnb listings dataset.'.format(airbnb_rows,airbnb_columns))


# .

# ### Zillow data (cost data)

# In[6]:


zillow.head()


# ### No of rows and columns in the dataset

# In[7]:


# no of rows and columns present in the zillow data
zillow_rows = zillow.shape[0]
zillow_columns = zillow.shape[1]
print('There are {} rows and {} columns in the zillow dataset.'.format(zillow_rows,zillow_columns))


# .

# # Data Cleaning

# ## Airbnb data

# ### Data Scrape information

# In[8]:


#unique values in the 'last_scraped column'
scrape = [x for x in airbnb.last_scraped.unique()]
print("The data in airbnb dataset were scraped on: ")
for i in scrape:
    print(i)


# <font size="4">*The 'scrape_id' and 'last_scraped' columns provides us the scrape id and the latest date when the data was scraped. This is a valuable information for us as it provides us with the date regarding the data. From the above code we can see that the data has be scraped in two days whose dates are '2019-07-09' and '2019-07-08'. We have already made the assumption that for our analysis we will consolidating the daily data into monthly data. Therefore, it is safe for us to consider the data to be scraped on '2019-07' or 'July 2019'
# Why do we have to keep this in mind? This is useful if we have to forecast the data from the zillow dataset as it is best for us to have the data from same time period for further analysis.*</font>

# ### Inspecting for multiple nation data

# In[9]:


## using the check_non_us function created and saved under the functions module
##check_non_us functions gives the user the numbers of country and country code present in the data besides the specified ones

df = airbnb
country = 'United States'
country_code = 'US'
functions.check_non_us(df, country, country_code)


# ## Basis of preliminary data preprocessing and feature selection

# 
# <font size="4">*After confirming that we have data only for US cities we can drop country and country_code column. Furthermore, Since the investor wants us to specifically analyse only New York City, we can select features such that we can filter our data for New York city. Also, 'neighbourhood_cleansed', 'neighbourhood_cleansed_grouped' are verified information about neighbourhood so we can just choose those.We will select only the columns required as having too many features that have same information is reduntant.* </font>
# 
# <font size="4">*After inspecting the AirBnB dataset and Metadata document further, it is apparent that there are a lot of columns in the dataset; 106 columns to be exact. Considering the nature of our analysis, instruction from the investors and assumptions, we can conduct a preliminary data preprocessing in order to get a dataset with smaller and relevant features without running any statistical analysis.* </font>
# 
# 
# <font size="4">*The purpose of the analysis is to find the zipcodes which are most profitable. For this, we need to focus only on the macro-economic factors or variables rather than the micro variables. The investing company is looking to buy properties and rent it out through AirBnB so all the variables like , house rules, amenities, access, security deposit, cleaning fee, host related data, host interaction data, extra people and guest related data are not needed as they all vary from renters and the investing company can work to make it the best interest of guest so they sell faster.  Also, there are various data related to the urls for the airbnb listing and images which we do not need.*</font> 
# 
# <font size="4">*The listings data also has various identifier columns and scrape information columns. For our purpose, we do not need any unique identifier columns as the data will be grouped by zipcode and there is no need to have granularity in the data beyond the zipcode. Furthermore, there are various review related features but only review score related to location is relevant to our analysis. So, based on above criteria I have done preliminary feature selection with purpose of retaining only the important features.*</font>
# 
# <font size="4">*Also, we only need to find the profitable zipcodes. Therefore, we can remove latitude and longitude from the dataset as we do not need in depth analysis of the location.*</font>
# 
# <font size="4">*Simialrly, the investors already know that 2 bedroom properties are most profitable so their only criteria is 2 bedroom and the type of property doesn't matter. Therefore we can remove the type of property as well as all other features of property exceept the number of bedroom.*</font>
# 
# <font size="4">*From the beginning, one of our assuption is that the occupancy rate of 75%. Therefore, we do not have to take the availability data into consideration and we can drop them.*</font>
# 
# <font size="4">*There are various columns having unique string data and need help of  Natural Language Processing models to process those dataand meaninful insights from them. So for our analysis, we will drop those columns as they add no value to us without NLP.*</font>
# 
# <font size="4">*There are various review scores data but we only need the "review_scores_location" columns as it provides us with the customers score of the location. All other review scores are dependent on the accuracy of description posted, cleanliness, communication and so on which depends on the host and after purchasing the property, the investing company can ensure that all of those are in best interest of renters. The only review score that the investor cannot control is the location review score, so we will add this column for our analysis*</font>
# 
# 
# 
# 
# 

# .

# ### Features selected after preliminary data preprocessing

# In[10]:


#fearutes identified as important ones based on the assumptions and scope of analysis
features = ['neighbourhood_group_cleansed', 'bedrooms','city',
       'state', 'zipcode', 'square_feet', 'price', 'weekly_price',
       'monthly_price', 'review_scores_location']


# ### creating a dataframe with only the required columns and printing the column names

# In[11]:


#using the prelim_preprocessing function saved under functions module
df1_airbnb = functions.prelim_preprocessing(airbnb, features)


# In[12]:


#viewing the new dataframe
df1_airbnb


# In[13]:


#data = df1_airbnb.isna()


# ### Plotting the completeness of the data

# In[14]:


# Visualize the completeness/ missing values, hte bar graphs shows how complete the data is
#Higher the bar, higher the data completeness and lower the NULL values and vice versa
# values as a bar chart 
msno.bar(df1_airbnb) 


# ### Plotting the missing/ NULL values

# In[15]:


#using the missing_value function saved under functions module
#missing_value functions takes in a dataframe and gives a table and plot of the total nulls and percentage of nulls for each columns in the dataframe
functions.missing_value(df1_airbnb)


# <font size="4">*From the above table and bar chart it clear that the columns "square_feet","weekly_price" and "monthly_price" have maximum null values. "square_feet" has 99.17% of the data as null value,"weekly_price" has 87.72% of the data as null value and "monthly_price"has 89.27% of the data as null value and there is not way to impute data for those missing values. Therefore, we should drop those columns from further analysis. Furthermore, 'neighbourhood_group_cleansed' is not necessary as our granulity level is zicode*</font>
# <font size="4">*Thus, we will be dropping following columns:*</font>
# - square_feet
# - weekly_price
# - monthly_price
# - neighbourhood_group_cleansed

# In[16]:


df2_airbnb = df1_airbnb.drop(columns=['neighbourhood_group_cleansed','square_feet', 'weekly_price', 'monthly_price'], axis = 1)


# In[17]:


df2_airbnb


# <font size="4">*After inspecting the state column, we can see that there are multiple patterns of NY and there are null values as well. Therefore, we will rectify the patterns of state column for NY and finter the dataframe for NY only as required by the investor*</font>

# In[18]:


df2_airbnb.state.unique()


# In[19]:


df2_airbnb.bedrooms.dtype


# In[20]:


#changing the datatype to string
df2_airbnb.state = df2_airbnb.state.astype(str).copy() 

#stripping excess whitespaces from the column
df2_airbnb['city'] = df2_airbnb['city'].str.strip()

#stripping excess whitespaces from the column
df2_airbnb['state'] = df2_airbnb['state'].str.strip()

#Makes all "NY"s uniform
df2_airbnb.state = df2_airbnb.state.replace(dict.fromkeys(['NY', 'ny', 'Ny','New York'],'NY')) 

#filtering the dataframe to get NewYork data only
df2_airbnb = df2_airbnb[df2_airbnb['state']=='NY']

#filtering the dataframe to get 2 bedroom property only
df2_airbnb = df2_airbnb[df2_airbnb['bedrooms']==2]


# In[21]:


#unique values in state column
df2_airbnb.state.unique()

#this shows us that the datafram now has only NewYork data


# In[22]:


#unique values in bedroom column
df2_airbnb.bedrooms.unique()
#this shows us that the datafram now has only 2 bedroom property


# In[23]:



# no of rows and columns present in the airbnb data after filtering with state = New York
ny_airbnb_rows = df2_airbnb.shape[0]
ny_airbnb_columns = df2_airbnb.shape[1]
print('There are {} rows airbnb listings dataset after filtering with New York.'.format(ny_airbnb_rows))


# In[24]:


df2_airbnb.to_csv('df2airbnbnew.csv')


# In[25]:


#using the missing_value function saved under functions module
#missing_value functions takes in a dataframe and gives a table and plot of the total nulls and percentage of nulls for each columns in the dataframe
functions.missing_value(df2_airbnb)


# .

# ### Correcting  zipcode and price values and filtering the dataset with state = NewYork 

# - <font size="4">*There are miltiple entries for zipcode where there is more that 5 digits and also 1.06% of the zidcode data is null. Since, the percentage of null data is very insignificant we will just drop it and reassign the zipcode by fetching only the first 5 digit of the zipcode*</font>
# 
# - <font size="4">*We will then remove the '$' and',' from the price data*</font>
# 
# - <font size="4">*Also, 0.12% of the city data is null. Since, the percentage of null data is very insignificant we will just drop it*</font>
# 
# 

# In[26]:


#copying the dataframe into a newone 
df3_airbnb = df2_airbnb.copy()

#reassigning only the real zipcode (first5 digit of zipcode) to the zipcode column of the data 
df3_airbnb.zipcode = df2_airbnb.zipcode.str[:5].copy()


# In[27]:


# filter out rows in dataframe with column zipcode values NA/NAN
df3_airbnb = df3_airbnb[df1_airbnb.zipcode.notnull()]


# In[28]:


#inspecting the unique values in zipcode to see if there is any null values or irrelavant values
df3_airbnb.zipcode.unique()


# In[29]:


#Removing the "$" and "," signs from the price column and changing the datatype to float
df3_airbnb['price'] = df3_airbnb['price'].replace( '[\$,)]','', regex=True ).astype(float).copy()


# In[30]:


# filter out rows in dataframe with column city values NA/NAN
df3_airbnb = df3_airbnb[df1_airbnb.city.notnull()]


# In[31]:


df3_airbnb.head(20)


# In[ ]:





# In[32]:


#check the data
#using the missing_value function saved under functions module
#missing_value functions takes in a dataframe and gives a table and plot of the total nulls and percentage of nulls for each columns in the dataframe
functions.missing_value(df3_airbnb)


# ### Imputing the missing data 
# 
# <font size="4">*After dealing with most of the features, we have one column i.e "review_scores_location" with 22.64% of its data as missing value. This features seems important for analysis and can be easily imputed by using regressor based on zip code and price. We will use MICE to impute the missing zipcodes*</font>

# In[33]:


#Mice only take numerical values of preparing a dataset for imputation
review=df3_airbnb[['zipcode', 'price','review_scores_location']]


# In[34]:


review.head(3)


# In[35]:


# Import IterativeImputer from fancyimpute
from fancyimpute import IterativeImputer

# Copy diabetes to diabetes_mice_imputed
review_mice_imputed = review.copy(deep=True)

# Initialize IterativeImputer
mice_imputer = IterativeImputer()

# Impute using fit_tranform on diabetes
review_mice_imputed.iloc[:, :] = mice_imputer.fit_transform(review)

#rounding off the imputed data
review_mice_imputed.review_scores_location = round(review_mice_imputed.review_scores_location,0)

#view the data
review_mice_imputed.head(3)


# In[36]:


#replacing the null values with the imputed value in the dataset
df3_airbnb.review_scores_location = review_mice_imputed.review_scores_location.copy()


# In[37]:


df3_airbnb


# In[38]:


#df3_airbnb.to_csv('df3airbnbnew.csv')


# .

# ### Dealing with extremely large and zero values in Prices
# 
# <font size="4">*After inspecting the price column, there were many rows with 0 as price so we will filter the dataset such that it will contain only prices greater than 0*</font>

# In[39]:


df3_airbnb = df3_airbnb[df3_airbnb['price']>0].copy()


# In[40]:


df3_airbnb


# In[ ]:





# ### Checking  for other outliers
# 

# In[41]:


# Percentile of the flattened array  
print("\n99th Percentile of price: ",  
      np.percentile(df3_airbnb.price, 99))


# In[42]:


import plotly.express as px

fig = px.box(df3_airbnb, y="price", height = 700)
fig.show()


# 
# <font size="4">*After inspecting the price column and plotting the box plot chart it was clear that there are many outliers. After calculating the 99th percentile of the price column we found out that 99th percentile of prices fall below 968. So, I have set the maximum threshold for the price to be 1000 as people will rarely buy any houses costing more that 1000 daily.*</font>

# In[43]:


#filtering the data for daily price less than or eual to $1000 
df3_airbnb = df3_airbnb[df3_airbnb['price']<=1000].copy()


# ### Box plot

# In[44]:


#the following code if for generating box plot with plotly

import plotly.express as px

fig = px.box(df3_airbnb, y="price", height = 700)
fig.show()


# ### Grouping the data by zipcode

# In[45]:


#count the number of unique zipcodes in our data and number of records
count = len(df3_airbnb.zipcode.unique())
print('There are {} unique zipcodes.'.format(count))

rows = df3_airbnb.shape[0]
print('There are {} records.'.format(rows))


# In[46]:


## Grouping the data
airbnb_grouped = df3_airbnb.groupby('zipcode').mean()


# In[47]:


airbnb_grouped


# .

# ## Zillow Data

# In[48]:


zillow


# In[49]:



# no of rows and columns present in the zillow data

zillow_rows = zillow.shape[0]
zillow_columns = zillow.shape[1]
print('There are {} rows and {} columns in the zillow dataset'.format(zillow_rows,zillow_columns))


# .

# 
# <font size="4">*We are not using data before 2012 in our model for 2 reasons: First, We have lots of null values for these time frame. Second, because of 2008-2011 recession, the prices of real estate properties has declined. If we use this data, It would potentially mislead our model against predicting correctly.*</font>
# 
# <font size="4">*We are considering current year as 2017 and forecasting for July 2019 as our revenue prices were scraped on July 2019 and it would be better to have data from simialr time period for accurate analysis. By using Facebook Prophet model, we will try to predict the value of the properties in year 2018 in order to calculate the profit percentage.*</font>
# 
# 
# 

# In[50]:


df = zillow


# ### Filtering the data to get New York data 

# In[51]:


# zillow_city_filter function takes in  three arguments and returns dataframe with values filterred based on the arguements provided.
# The first argument df is the dataframe from which the user intends to filter values
# The second argument State is the name of the state State from which the user intends to filter values
# The third argument Metro is the Metro name from which the user intends to filter values

df = zillow
State = "NY"
Metro = "New York"
df1 = functions.zillow_city_filter(df, State, Metro)


# ### Preprocessing the data to get median cost values zipcodes and time periods

# In[52]:


# zillow_preprocess function takes in  two arguments and returns dataframe with required values for time series forecasting. 
# The first argument df is the dataframe which needs to be proprocessed.
# The second argument n is the number of recent months required to be present in the dataframe.

#number of months if denoted by n
n = 61

df2 = functions.zillow_preprocess(df1, 61)


# In[53]:


df2


# .

# .

# ### Forecasting property cost using Facebook Prophet

# In[54]:


# zillow_prophet function takes in  a preprocessed dataframe and returns the dataframe with forecasted current value of the property as "currentPrice".
# Facebook prophet is used for the time series forecasting.
# Facebook prophet is various robust and can handle missing values as well and perform very well with large scale of data as well.


# In[55]:


#df_currentPrice = functions.zillow_prophet(df2, 26)


# In[57]:


df_currentPrice


# In[58]:


#df_currentPrice.to_csv('crrntprce.csv')


# # Analysis

# ## Merge the datasets

# In[59]:


airbnb_grouped


# In[ ]:





# In[60]:


#getting a dataframe with zipcode and property cost 
cost =  df_currentPrice[['zipcode', 'currentPrice' ]]
cost = cost.rename(columns={'currentPrice': 'Property_Cost'})            
cost
            


# In[61]:


## Based on our asssumption of 75% occupancy rate, we are calculating occupied days
occupancy_rate = 0.75
occupancy_days = occupancy_rate*365


# ### Calculating Yearly revenue

# In[62]:




revenue = airbnb_grouped.copy()
revenue = revenue.reset_index()
revenue=revenue[['zipcode','price','review_scores_location']]
revenue['yearly_revenue']= occupancy_days*revenue['price']
revenue 


# ## Merging Revenue and Cost dataframe on zipcode

# In[63]:


#merging two dataframe on a common key
df_merge = pd.merge(cost, revenue, on='zipcode')


# In[64]:


#calculating breakeven period
df_merge['breakeven_period'] = df_merge['Property_Cost']/df_merge['yearly_revenue']


# In[65]:


#merged dataset
df_merge


# ## Dataframe sorted by breakeven period, cost and revenue

# ### sorted by Breakeven Period

# In[66]:


#sorted by breakeven period
sort_breakeven = df_merge.sort_values(by='breakeven_period', ascending=True)
sort_breakeven


# ### sorted by Yearly Revenue

# In[67]:


#sorted by yearly revenue
sort_revenue = df_merge.sort_values(by='yearly_revenue', ascending=False)
sort_revenue


# ### sorted by Property Cost

# In[68]:


#sorted by property cost
sort_cost = df_merge.sort_values(by='Property_Cost', ascending=False)
sort_cost


# ## Top 10 zipcodes based on Cost and Revenue

# In[69]:


sort_cost_top = sort_cost.head(10)
print("Top 10 zipcodes with cost: ")
display(sort_cost_top)

sort_breakeven_top = sort_breakeven.head(10)
print("Top 10 zipcodes with fastest breakeven period: ")
display(sort_breakeven_top)



sort_revenue_top = sort_revenue.head(10)
print("Top 10 zipcodes with highest revenue: ")
display(sort_revenue_top)



# ## Visualizations

# ### Property Cost

# <font size="4">*The below table and graph identifies the property cost for different zipcodes. Zipcodes 10013, 10011, 10014 cost the highest with 3 Million, 2.8 million and 2.79 million respectively whereas, zipcode 10036, 10128 and 10021 have the lowest cost  with values less than 500,000.*</font>
# 

# In[70]:


sort_cost_top = sort_cost.head(10)
print("Top 10 zipcodes with cost: ")
display(sort_cost_top)

sort_cost_bot= sort_cost.tail(10)
print("zipcodes with  lowest cost: ")
display(sort_cost_bot)


# In[71]:


import plotly.express as px

fig = px.scatter(sort_revenue, x="Property_Cost", y="zipcode", color="zipcode",
                 size='Property_Cost', hover_data=['Property_Cost'])
fig.show()


# ### Yearly revenue

# <font size="4">*The below tables and graph identifies the yearly revenue for different zipcodes. Zipcodes 10013, 10011, 10036 yield the highest yearly revenue with 99379 dollars, 95382 dollars and 90511 dollars respectively whereas, zipcode 10314, 10309 and 10304 have the yearly revenue  with values 19983 dollars, 23268 dollars and 25550 dollars respectively.*</font>
# 

# In[72]:


sort_revenue_top = sort_revenue.head(10)
print("Top 10 zipcodes with highest revenue: ")
display(sort_revenue_top)

sort_revenue_bot = sort_revenue.tail(10)
print("Top 10 zipcodes with lowest revenue: ")
display(sort_revenue_bot)


# In[73]:


import plotly.express as px

fig = px.scatter(sort_revenue, x="yearly_revenue", y="zipcode", color="zipcode",
                 size='yearly_revenue', hover_data=['yearly_revenue'])
fig.show()


# ### Breakeven period

# <font size="4">*The below tables and graph identifies the breakeven period for different zipcodes. Zipcodes 10013, 10011, 10036 take the least time to breakeven with breakeven period of 8 years, 13 years and 13 years respectively whereas, zipcode 10021, 10028 and 10014 take the longest to breakeven with breakeven period of 36, 34 and 32 years respectively.*</font>
# 
# 

# In[74]:


sort_breakeven_top = sort_breakeven.head(10)
print("Top 10 zipcodes with fastest breakeven period: ")
display(sort_breakeven_top)

sort_breakeven_bot = sort_breakeven.tail(10)
print("Top 10 zipcodes with maximum breakeven period: ")
display(sort_breakeven_bot)


# In[75]:


import plotly.express as px

fig = px.scatter(sort_revenue, x="breakeven_period", y="zipcode", color="zipcode",
                 size='breakeven_period', hover_data=['breakeven_period'])
fig.show()


# ### Location Review Score

# ### Review_score_location and Yearly Revenue

# <font size="4">*The review score based on the location seems to be all over the chart and is less conclusive in determining the revenue as the properties in zipcodes yeilding least revenue also have high ratings along with low ratings. However, most of the high revenue yielding zipcodes have higher rating*</font>
# 
# 

# In[76]:


import plotly.express as px

fig = px.scatter(sort_revenue, x="yearly_revenue", y="review_scores_location", color="zipcode",
                 size='review_scores_location', hover_data=['review_scores_location'])
fig.show()


# ### Review_score_location and Property cost

# <font size="4">*This chart does show that most of the zipcode having properties that cost high tend to have highers review score based on location. However, there are few zipcode having high review score based on location even when they contain least expensive properties.*</font>

# In[77]:


import plotly.express as px

fig = px.scatter(sort_revenue, x="Property_Cost", y="review_scores_location", color="zipcode",
                 size='review_scores_location', hover_data=['review_scores_location'])
fig.show()


# In[ ]:





# # Conclusion and Recommendation

# <font size="4">*The following conclusions can be derived from the analysis:*</font>
# 
# - Properties in zip codes that represent Manhattan are the most expensive followed by properties in Brooklyn.
# - Properties in zipcodes belonging to Manhattan have highest daily price.
# - Properties near zipcodes belonging to Manhattan has the highest yearly revenue ranging from 75k-100k dollars per year
# - It can also be concluded that zipcodes nearby Central Park cost the most and yeild the highest revenue as well.
# - Properties in zipcodes belonging to Staten Island have the least cost as well as yield the least revenue followed by properties in Rochdale.
# - Cheaper properties in zip codes belonging to Staten Island and Rochdale acieve breakeven earlier rather than properties in zipcodes belonging to Manhattan.
# - Review scores based on location tend to be higher for properties having high cost as well as yeilding high revenue but this relation is not exclusive.
# 
# 
# <font size="4">*The following Recommendations can be derived from the analysis:*</font>
# 
# - Since, the investor is seeking High short term Return on Investment, s/he can purchase properties in Manhattan which provide the highest yearly revenue. 
# - Zipcodes providing highest short-term return on investment with yearly revenue more than 75k dollars are:
#     - 10013
#     - 10011
#     - 10036
#     - 10014
#     - 10003
#     - 10022

# # Further  Possibilities 

# <font size="4">*There are additional analyis that can be done to make more concrete and better decisions. Some of the possible additional steps are:*</font>
# - Visuallizations incorporating latitude and longitude data can be used for mapping and better visualization which was not done due to time constraints
# - Collect and incorporate additional data to make better analysis. Factors like House Price Index and inflation are very important and should be incorporated.
# - One of the major factors in real state industry is safety. Open source Crime data can be added to the dataseet to further narow down profitable properties in safe neighborhood. 
# - Data related to availability of transportation, natural scenary, entertainment, shopping malls, restaurants, hospitals, groceries among other can affect the choice of property for renters and incorporating such data will provide us with more realistic insights.
# - Conducting Competitor analysis is very important while making any business decisions. So, adding details/ data regarding nearby competitors can be very useful anf informative.
# - Segmenting the property type and target customersand conducting analysis for seperately may prove to be more effective. The needs and requirements of a business person will be totally different than that of a student. If we specific customers at a timeand conduct analysis, it may lead to effective insights. 
# - Adding additional data regarding the costs such as property tax, property maintenace cost, utility cost and so on must be incorporated to provide more realistic business insights.
# - Machine learning models can be used for better and accurate predictions such as property price prediction based on the features rather than just longitudinal data of one feature.
# - Natural Language Processing algoriths can be used to get better and more insights from the unique text data.
# - Also, sentiment analysis can be dome from various peoples social media data inorder to get the sentiment of different categories of customers towards different properties which can help us in making more informed decisions.
