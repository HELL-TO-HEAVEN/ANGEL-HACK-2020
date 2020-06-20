
import json
import numpy as np
import pandas as pd, datetime
import joblib
import geopandas as gpd
from functools import reduce
from sklearn.preprocessing import LabelEncoder
from math import radians
import networkx as nx
import lightgbm as lgb
from category_encoders.cat_boost import CatBoostEncoder

from azureml.core.model import Model
######################################################################################################################
# Functions
# Functions
def getbaseLineETA(df,columns=["pickup_id","destination_id","trj_id"]):
    predictedTime=[]
    predictedPath=[]
    predictedPath2=[]
    predictedTime2=[]
    for item in df[columns].values.tolist():
        if G.has_node(item[0]) and G.has_node(item[1]):
            if nx.has_path(G,item[0],item[1]):
                shortestPath=nx.dijkstra_path(G,item[0],item[1], weight="popularity+time")
                shortestPath2=nx.dijkstra_path(G,item[0],item[1], weight="mean")
                total_time=0
                for index in range(len(shortestPath)-1):
                    a=shortestPath[index]
                    b=shortestPath[index+1]
                    total_time+=G[a][b]['mean']
                predictedTime.append(total_time)
                predictedPath.append(shortestPath)
                predictedTime2.append(nx.dijkstra_path_length(G,item[0],item[1], weight="mean"))
                predictedPath2.append(shortestPath2)
            else:
                predictedTime.append("")
                predictedPath.append([])
                predictedTime2.append("")
                predictedPath2.append([])
        else:
            predictedTime.append("")
            predictedPath.append([])
            predictedTime2.append("")
            predictedPath2.append([])
    return predictedTime,predictedPath,predictedTime2,predictedPath2

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns 

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)
    
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def euclidean_distance(x,y):
 
    return np.sqrt((np.sum(np.power(x-y,2),axis=1)))

def manhattan_distance(x,y):
 
    return np.sum(np.abs(x-y),axis=1)

def classify(x):
    if x >=8 and x<=9:
        return 10
    elif x>=17 and x<=19:
        return 50
    return 100

######################################################################################################################

def init():

    global df_geo
    global G
    global weather_data_filter
    global loaded_model_1
    global loaded_model_2
    global destination_pivot_df
    global pickup_pivot_df
    global h 
    global a
    global inCentrality
    global outCentrality
    global loadCentrality
    
    model_path = Model.get_model_path('df_geo')
    df_geo = joblib.load(model_path)    
    
    model_path = Model.get_model_path('G')
    G = joblib.load(model_path)
    
    model_path = Model.get_model_path('weather_data_filter')
    weather_data_filter = joblib.load(model_path)
    
    model_path = Model.get_model_path('loaded_model_1')
    loaded_model_1 = joblib.load(model_path)
    
    model_path = Model.get_model_path('loaded_model_2')
    loaded_model_2 = joblib.load(model_path)
    
    model_path = Model.get_model_path('destination_pivot_df')
    destination_pivot_df = joblib.load(model_path)
    
    model_path = Model.get_model_path('pickup_pivot_df')
    pickup_pivot_df = joblib.load(model_path)
    
    model_path = Model.get_model_path('h')
    h = joblib.load(model_path)

    model_path = Model.get_model_path('a')
    a = joblib.load(model_path)

    model_path = Model.get_model_path('inCentrality')
    inCentrality = joblib.load(model_path)
    
    model_path = Model.get_model_path('outCentrality')
    outCentrality = joblib.load(model_path)
    
    model_path = Model.get_model_path('loadCentrality')
    loadCentrality = joblib.load(model_path)

def run(df_json):
    sample_test_data = pd.read_json(df_json)
    sample_test_data["sequence"]=sample_test_data.index
    gdf_origin = gpd.GeoDataFrame(sample_test_data.copy(),   geometry=gpd.points_from_xy(sample_test_data.longitude_origin, sample_test_data.latitude_origin),crs={'init': 'epsg:4326'})
    gdf_destination = gpd.GeoDataFrame(sample_test_data.copy(),   geometry=gpd.points_from_xy(sample_test_data.longitude_destination, sample_test_data.latitude_destination),crs={'init': 'epsg:4326'})  

    sjoined_origin = gpd.sjoin(gdf_origin, df_geo, op="within",how="left")
    sjoined_origin['id']=sjoined_origin['id'].fillna(-1)

    sjoined_destination = gpd.sjoin(gdf_destination, df_geo, op="within",how="left")
    sjoined_destination['id']=sjoined_destination['id'].fillna(-1)

    sjoined_origin_selected= sjoined_origin[['latitude_origin','longitude_origin','timestamp','hour_of_day','day_of_week','id','REGION_N','sequence']]
    sjoined_destination_selected= sjoined_destination[['latitude_destination','longitude_destination','id','REGION_N','sequence']]

    sjoined_origin_selected.rename(columns={'id': 'pickup_id','REGION_N':'REGION_N_pickup'}, inplace=True)
    sjoined_destination_selected.rename(columns={'id': 'destination_id','REGION_N':'REGION_N_dest'}, inplace=True)

    data_joined= pd.merge(sjoined_origin_selected,sjoined_destination_selected, how='left', on='sequence')
    data_joined['Date'] = pd.to_datetime(data_joined.timestamp,unit='s')
    data_joined.rename(columns={'sequence': 'trj_id'}, inplace=True)

    predictedTime,predictedPath,predictedTime2,predictedPath2 =getbaseLineETA(data_joined)
    data_joined["popularity+time"]=pd.Series(predictedTime)
    data_joined["popularity+time_path"]=pd.Series(predictedPath)
    data_joined["timeOnly"]=pd.Series(predictedTime2)
    data_joined["timeOnlyPath"]=pd.Series(predictedPath2)

    data_joined["dayOfWeek"]=data_joined["Date"].dt.dayofweek
    data_joined["Day"]=data_joined["Date"].dt.day
    data_joined["month"]=data_joined["Date"].dt.month
    data_joined["hour"]=data_joined['Date'].dt.hour

    data_joined["pathHIndex"]=data_joined['popularity+time_path'].apply(lambda x:sum(map(lambda item:h[item],x)))
    data_joined["pathaIndex"]=data_joined['popularity+time_path'].apply(lambda x:sum(map(lambda item:a[item],x)))
    data_joined["pathInDegreeIndex"]=data_joined['popularity+time_path'].apply(lambda x:sum(map(lambda item:inCentrality[item],x)))
    data_joined["pathOutDegreeIndex"]=data_joined['popularity+time_path'].apply(lambda x:sum(map(lambda item:outCentrality[item],x)))
    data_joined["CountOfGrid"]=data_joined['popularity+time_path'].apply(len)
    data_joined["destLoad"]=data_joined.destination_id.map(loadCentrality)
    data_joined["destHIndex"]=data_joined.destination_id.map(h)
    data_joined["pickUpLoad"]=data_joined.pickup_id.map(loadCentrality)
    data_joined["flowType"]=data_joined.REGION_N_pickup +"->"+data_joined.REGION_N_dest

    data_joined['Date_round'] = data_joined['Date'].dt.floor('h')

    data_joined1= pd.merge(data_joined,weather_data_filter, how='left', left_on='Date_round', right_on='NewDT_round')
    data_joined1['Temp'] = data_joined1['Temp'].str.replace('°F', '')
    data_joined1['Weather'] = data_joined1['Weather'].str.replace('.', '')
    data_joined1['Wind'] = data_joined1['Wind'].str.replace('mph', '')
    data_joined1['Humidity'] = data_joined1['Humidity'].str.replace('%', '')
    data_joined1['Barometer'] = data_joined1['Barometer'].str.replace('"Hg', '')
    data_joined1['Visibility'] = data_joined1['Visibility'].str.replace('mi', '')
    data_joined1['Wind']=np.where((data_joined1['Wind'].isnull())|(data_joined1['Wind']=='No wind'),'0',data_joined1['Wind'] )
    data_joined1['Visibility']=np.where((data_joined1['Visibility'].isnull())|(data_joined1['Visibility']=='N/A'),'0',data_joined1['Visibility'] )
    data_joined1['Temp'] = data_joined1['Temp'].astype(np.float64)
    data_joined1['Wind'] = data_joined1['Wind'].astype(np.float64)
    data_joined1['Humidity'] = data_joined1['Humidity'].astype(np.float64)
    data_joined1['Barometer'] = data_joined1['Barometer'].astype(np.float64)
    data_joined1['Visibility'] = data_joined1['Visibility'].astype(np.float64)

    df_join2= pd.merge(data_joined1, pickup_pivot_df, how='left', left_on='pickup_id', right_on='pickup_id_pick')
    df_join3= pd.merge(df_join2,destination_pivot_df, how='left', left_on='destination_id', right_on='destination_id_des')
    df_original=df_join3[["trj_id",'latitude_origin','longitude_origin','latitude_destination','longitude_destination','timestamp','hour_of_day','day_of_week']]

    df_join3["Festival"]=np.where(df_join3['day'].isin(['2019-04-01','2019-04-18','2019-04-19','2019-04-20','2019-04-21','2019-05-01','2019-05-11','2019-05-12','2019-05-17','2019-05-18','2019-05-19','2019-05-20']),1,0)
    df_geo["X"]=df_geo.geometry.centroid.x
    df_geo["y"]=df_geo.geometry.centroid.y
    df_join3=df_join3.merge(df_geo[["id","X","y"]],left_on="pickup_id",right_on="id",how="left").merge(df_geo[["id","X","y"]],left_on="destination_id",right_on="id",suffixes=("_pickup","_dest"),how="left")
    df_join3[(df_join3['popularity+time'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3['timeOnlyPath'].astype(str).apply(lambda x: len(x)) == 0)]
    df_join3['rawlat_pickup']=df_join3['latitude_origin']
    df_join3['rawlng_pickup']=df_join3['longitude_origin']
    df_join3['rawlat_dest']=df_join3['latitude_destination']
    df_join3['rawlng_dest']=df_join3['longitude_destination']
    df_join3["rawlng_pickup2"] = df_join3["rawlng_pickup"].apply(radians)
    df_join3["rawlat_pickup2"] = df_join3["rawlat_pickup"].apply(radians)
    df_join3["rawlng_dest2"] = df_join3["rawlng_dest"].apply(radians)
    df_join3["rawlat_dest2"] = df_join3["rawlat_dest"].apply(radians)
    df_join3["HarvsineDistance"]=haversine(df_join3["rawlng_pickup2"], df_join3["rawlat_pickup2"], df_join3["rawlng_dest2"], df_join3["rawlat_dest2"])
    df_join3["EuclideanDistance"]=euclidean_distance(df_join3[["rawlng_pickup","rawlat_pickup"]].values, df_join3[["rawlng_dest","rawlat_dest"]].values)
    df_join3["ManhattanDistance"]=manhattan_distance(df_join3[["rawlng_pickup","rawlat_pickup"]].values, df_join3[["rawlng_dest","rawlat_dest"]].values)
    df_join3["TimeInterval"]=df_join3.hour.apply(classify)
    df_join3["PopularPlace"]=0
    df_join3["Weekend"]=df_join3.dayOfWeek.map({6:1,5:1}).fillna(0)
    df_join3.loc[df_join3.destination_id.isin([6815,6816,6885,6886,517,587,588,658,659,590,519,589,2877,2876,2875,2947,2946,2945]),'PopularPlace']=1

    # can match
    data_for_model_1=df_join3[~((df_join3['popularity+time'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3['timeOnlyPath'].astype(str).apply(lambda x: len(x)) == 0))]
    var_drop=['latitude_origin','longitude_origin','timestamp','pickup_id','REGION_N_pickup','destination_id','latitude_destination',
         'longitude_destination','Date','popularity+time_path','timeOnlyPath','hour_of_day','day_of_week','month','Date_round',
         'Time','day','Time24','NewDT','session','NewDT_round','pickup_id_pick','destination_id_des',
         'id_pickup','X_pickup','y_pickup','id_dest','X_dest','y_dest','rawlng_pickup2','rawlat_pickup2','rawlng_dest2','rawlat_dest2',
           'EuclideanDistance','ManhattanDistance','Day']
    data_for_model_1 = data_for_model_1.drop(columns = var_drop, axis = 1)
    data_for_model_1.rename(columns={"REGION_N_dest": "destination_Region"}, inplace = True)
    data_for_model_1['Weather'].fillna('Not Available', inplace=True)
    trip_Data=data_for_model_1['trj_id'].values
    data_for_model_1_re=data_for_model_1[['destination_Region',  'popularity+time',  'timeOnly',  'dayOfWeek',  'hour',  'pathHIndex',  'pathaIndex',  'pathInDegreeIndex',  'pathOutDegreeIndex',  'CountOfGrid',  'destLoad',  'destHIndex',  'pickUpLoad',  'flowType',  'Festival',  'Temp',  'Weather',  'Wind',  'Humidity',  'Barometer',  'Visibility',  'Bus Terminal_pick',  'Business Area_pick',  'HDB_pick',  'Hospital_pick',  'LRT_pick',  'MRT_pick',  'Mall/Supermarket_pick',  'Other Residentials_pick',  'School/Kindergarten_pick',  'University/College_pick',  'Bus Terminal_des',  'Business Area_des',  'HDB_des',  'Hospital_des',  'LRT_des',  'MRT_des',  'Mall/Supermarket_des',  'Other Residentials_des',  'School/Kindergarten_des',  'University/College_des',  'rawlat_pickup',  'rawlng_pickup',  'rawlat_dest',  'rawlng_dest',  'HarvsineDistance',  'TimeInterval',  'PopularPlace',  'Weekend']]
    join_predicted_1=pd.DataFrame()
    if not data_for_model_1_re.empty:
        data_for_model_1_re['eta']= loaded_model_1.predict(data_for_model_1_re)

        predicted_values_1=data_for_model_1_re[['eta']]
        predicted_values_1["trj_id"]=trip_Data
        join_predicted_1= pd.merge(df_original, predicted_values_1,on="trj_id")

    data_for_model_2=df_join3[(df_join3['popularity+time'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3['timeOnlyPath'].astype(str).apply(lambda x: len(x)) == 0)]
    if data_for_model_2.shape[0]>=1:
        data_for_model_2['eta']= loaded_model_2.predict(data_for_model_2[["Festival",'HarvsineDistance']])
        predicted_values_2=data_for_model_2[['eta']]
        predicted_values_2["trj_id"]=data_for_model_2["trj_id"].values
        join_predicted_2= pd.merge(df_original, predicted_values_2,on="trj_id")
        finalprediction= join_predicted_1.append(join_predicted_2)

    else: 
        finalprediction= join_predicted_1.copy()

    return json.dumps(finalprediction.sort_values("trj_id").drop("trj_id",axis=1).eta.tolist())
