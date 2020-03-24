#!/usr/bin/env python


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
