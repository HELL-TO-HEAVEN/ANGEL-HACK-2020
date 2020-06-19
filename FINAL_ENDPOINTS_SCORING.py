import json
import requests
import pandas as pd

#conversion to pandas
def conv_(ch): 
    res = ch.strip('][').split(', ') 
    res = [s.strip('["]') for s in res]
    res_pd = pd.DataFrame(res)
    res_pd.columns = ['pred']
    res_pd['pred'] = pd.to_numeric(res_pd['pred'])
    return res_pd


#scoring function
def score_url(df): 
# URL for the web service.
    scoring_uri = 'http://41698060-7459-44a9-97d6-6b4da799ad9a.southeastasia.azurecontainer.io/score'
    # Convert to JSON string.
    input_data_json = df.to_json()
    # Set the content type.
    headers = {'Content-Type': 'application/json'}
    # Make the request and display the response.
    resp = requests.post(scoring_uri, input_data_json, headers=headers)
    # print(resp.text)
    pred = conv_(resp.text)
    return pred

#scoring function - by batch of 1000 
def score_batch(df):
    pd_all = pd.DataFrame()
    if df.shape[0]>1000:
        for i in range(1,len(df)//1000+1,1):
            a = 1000*(i-1); b = i*1000
            temp = df.iloc[a:b,:]
            pred_temp = score_url(temp)
            pd_all = pd.concat([pd_all, pred_temp], axis = 0)
    else:
        pd_all= score_url(df)
        
    return pd_all

# KINDLY KEY IN THE DIRECTORY
# Load the data from CSV (in pandas format)
inp = ""
data = pd.read_csv(inp+'sample_all.csv')
sample = pd.read_csv(inp+'sample.csv') #IMPORT THE SAMPLE
target = pd.read_csv(inp+'target_all.csv')


test_all = score_batch(sample)

# Calculation of RMSE
from sklearn.metrics import mean_squared_error

mean_squared_error(test_all.pred,target["eta"])**0.5

