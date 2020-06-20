#!/usr/bin/env python
# coding: utf-8

# In[1]:


# REF
# https://github.com/Azure/MachineLearningNotebooks/blob/master/how-to-use-azureml/training/train-within-notebook/train-within-notebook.ipynb
# https://github.com/janakiramm/azureml-tutorial/blob/master/Azure/score.py
# https://azure.github.io/learnAnalytics-UsingAzureMachineLearningforAIWorkloads/lab09-collect_and_analyze_data_from_a_scoring_service/0_README.html
# https://github.com/avanish-fullstack/CarSalesPrediction-AzureMLService/blob/master/Car%20Price%20Prediction%20using%20Azure%20ML%20Service.ipynb


# # Import standard Python modules
# 

# In[2]:


import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
# from sklearn.externals import joblib


# # Import Azure ML SDK modules
# 

# In[3]:


import azureml.core
from azureml.core import Workspace,Environment
from azureml.core.model import Model
from azureml.core import Experiment
from azureml.core.webservice import Webservice
from azureml.core.image import ContainerImage
from azureml.core.webservice import AciWebservice
from azureml.core.conda_dependencies import CondaDependencies


# In[4]:


print(azureml.core.VERSION)


# In[7]:


inp = "D:/Kaggle/AngelHack/stupid azure/FinalModel/"


# # Create Workspace

# In[ ]:


# AZ_SUBSCRIPTION_ID=' '
# ws = Workspace.create(name='final_model',
#                       subscription_id=AZ_SUBSCRIPTION_ID, 
#                       resource_group='ML',
#                       create_resource_group=True,
#                       location='southeastasia'
#                      )



ws = Workspace.from_config()


# In[6]:


print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Subscription id: ' + ws.subscription_id, 
      'Resource group: ' + ws.resource_group, sep='\n')


# # Register Model
# 
# External data can be serialised and posted inside

# In[104]:


df_geo = Model.register(model_path = inp + "df_geo.pkl",
                       model_name = "df_geo",
                       tags = {"key": "1"},
                       description = "df_geo",
                       workspace = ws)
G = Model.register(model_path = inp + "G.pkl",
                       model_name = "G",
                       tags = {"key": "1"},
                       description = "G",
                       workspace = ws)
weather_data_filter = Model.register(model_path = inp + "weather_data_filter.pkl",
                       model_name = "weather_data_filter",
                       tags = {"key": "1"},
                       description = "weather_data_filter",
                       workspace = ws)
loaded_model_1 = Model.register(model_path = inp + "loaded_model_1.pkl",
                       model_name = "loaded_model_1",
                       tags = {"key": "1"},
                       description = "model",
                       workspace = ws)
loaded_model_2 = Model.register(model_path = inp + "loaded_model_2.pkl",
                       model_name = "loaded_model_2",
                       tags = {"key": "1"},
                       description = "model",
                       workspace = ws)
destination_pivot_df = Model.register(model_path = inp + "destination_pivot_df.pkl",
                       model_name = "destination_pivot_df",
                       tags = {"key": "1"},
                       description = "destination_pivot_df",
                       workspace = ws)
pickup_pivot_df = Model.register(model_path = inp + "pickup_pivot_df.pkl",
                       model_name = "pickup_pivot_df",
                       tags = {"key": "1"},
                       description = "pickup_pivot_df",
                       workspace = ws)
h = Model.register(model_path = inp + "h.pkl",
                       model_name = "h",
                       tags = {"key": "1"},
                       description = "h",
                       workspace = ws)
a = Model.register(model_path = inp + "a.pkl",
                       model_name = "a",
                       tags = {"key": "1"},
                       description = "a",
                       workspace = ws)
inCentrality = Model.register(model_path = inp + "inCentrality.pkl",
                       model_name = "inCentrality",
                       tags = {"key": "1"},
                       description = "inCentrality",
                       workspace = ws)
outCentrality = Model.register(model_path = inp + "outCentrality.pkl",
                       model_name = "outCentrality",
                       tags = {"key": "1"},
                       description = "outCentrality",
                       workspace = ws)
loadCentrality = Model.register(model_path = inp + "loadCentrality.pkl",
                       model_name = "loadCentrality",
                       tags = {"key": "1"},
                       description = "loadCentrality",
                       workspace = ws)


# # Define Azure ML Deploymemt configuration
# 

# In[105]:


aciconfig = AciWebservice.deploy_configuration(cpu_cores=1, 
                                               memory_gb=4, 
                                               tags={"data": "A-B",  "method" : "lgb+dt"}, 
                                               description='Predict ETA')


# # Create enviroment configuration file

# In[106]:


salenv = CondaDependencies()
salenv.add_conda_package("scikit-learn")
salenv.add_conda_package("geopandas")
salenv.add_conda_package("lightgbm")
salenv.add_conda_package("pandas")
salenv.add_conda_package("numpy")
salenv.add_conda_package("joblib")
salenv.add_conda_package("pyproj")
salenv.add_conda_package("networkx")
salenv.add_conda_package("category_encoders")

salenv.remove_channel("anaconda")



with open("ml.yml","w") as f:
    f.write(salenv.serialize_to_string())
with open("ml.yml","r") as f:
    print(f.read())


# # Create Azure ML Scoring file

# In[107]:


get_ipython().run_cell_magic('writefile', 'score.py', '\nimport json\nimport numpy as np\nimport pandas as pd, datetime\nimport joblib\nimport geopandas as gpd\nfrom functools import reduce\nfrom sklearn.preprocessing import LabelEncoder\nfrom math import radians\nimport networkx as nx\nimport lightgbm as lgb\nfrom category_encoders.cat_boost import CatBoostEncoder\n\nfrom azureml.core.model import Model\n######################################################################################################################\n# Functions\n# Functions\ndef getbaseLineETA(df,columns=["pickup_id","destination_id","trj_id"]):\n    predictedTime=[]\n    predictedPath=[]\n    predictedPath2=[]\n    predictedTime2=[]\n    for item in df[columns].values.tolist():\n        if G.has_node(item[0]) and G.has_node(item[1]):\n            if nx.has_path(G,item[0],item[1]):\n                shortestPath=nx.dijkstra_path(G,item[0],item[1], weight="popularity+time")\n                shortestPath2=nx.dijkstra_path(G,item[0],item[1], weight="mean")\n                total_time=0\n                for index in range(len(shortestPath)-1):\n                    a=shortestPath[index]\n                    b=shortestPath[index+1]\n                    total_time+=G[a][b][\'mean\']\n                predictedTime.append(total_time)\n                predictedPath.append(shortestPath)\n                predictedTime2.append(nx.dijkstra_path_length(G,item[0],item[1], weight="mean"))\n                predictedPath2.append(shortestPath2)\n            else:\n                predictedTime.append("")\n                predictedPath.append([])\n                predictedTime2.append("")\n                predictedPath2.append([])\n        else:\n            predictedTime.append("")\n            predictedPath.append([])\n            predictedTime2.append("")\n            predictedPath2.append([])\n    return predictedTime,predictedPath,predictedTime2,predictedPath2\n\nclass MultiColumnLabelEncoder:\n    def __init__(self,columns = None):\n        self.columns = columns \n\n    def fit(self,X,y=None):\n        return self\n\n    def transform(self,X):\n        output = X.copy()\n        if self.columns is not None:\n            for col in self.columns:\n                output[col] = LabelEncoder().fit_transform(output[col])\n        else:\n            for colname,col in output.iteritems():\n                output[colname] = LabelEncoder().fit_transform(col)\n        return output\n\n    def fit_transform(self,X,y=None):\n        return self.fit(X,y).transform(X)\n    \ndef haversine(lon1, lat1, lon2, lat2):\n    """\n    Calculate the great circle distance between two points \n    on the earth (specified in decimal degrees)\n    """\n    # haversine formula \n    dlon = lon2 - lon1 \n    dlat = lat2 - lat1 \n    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2\n    c = 2 * np.arcsin(np.sqrt(a)) \n    r = 6371 # Radius of earth in kilometers. Use 3956 for miles\n    return c * r\n\ndef euclidean_distance(x,y):\n \n    return np.sqrt((np.sum(np.power(x-y,2),axis=1)))\n\ndef manhattan_distance(x,y):\n \n    return np.sum(np.abs(x-y),axis=1)\n\ndef classify(x):\n    if x >=8 and x<=9:\n        return 10\n    elif x>=17 and x<=19:\n        return 50\n    return 100\n\n######################################################################################################################\n\ndef init():\n\n    global df_geo\n    global G\n    global weather_data_filter\n    global loaded_model_1\n    global loaded_model_2\n    global destination_pivot_df\n    global pickup_pivot_df\n    global h \n    global a\n    global inCentrality\n    global outCentrality\n    global loadCentrality\n    \n    model_path = Model.get_model_path(\'df_geo\')\n    df_geo = joblib.load(model_path)    \n    \n    model_path = Model.get_model_path(\'G\')\n    G = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'weather_data_filter\')\n    weather_data_filter = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'loaded_model_1\')\n    loaded_model_1 = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'loaded_model_2\')\n    loaded_model_2 = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'destination_pivot_df\')\n    destination_pivot_df = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'pickup_pivot_df\')\n    pickup_pivot_df = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'h\')\n    h = joblib.load(model_path)\n\n    model_path = Model.get_model_path(\'a\')\n    a = joblib.load(model_path)\n\n    model_path = Model.get_model_path(\'inCentrality\')\n    inCentrality = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'outCentrality\')\n    outCentrality = joblib.load(model_path)\n    \n    model_path = Model.get_model_path(\'loadCentrality\')\n    loadCentrality = joblib.load(model_path)\n\ndef run(df_json):\n    sample_test_data = pd.read_json(df_json)\n    sample_test_data["sequence"]=sample_test_data.index\n    gdf_origin = gpd.GeoDataFrame(sample_test_data.copy(),   geometry=gpd.points_from_xy(sample_test_data.longitude_origin, sample_test_data.latitude_origin),crs={\'init\': \'epsg:4326\'})\n    gdf_destination = gpd.GeoDataFrame(sample_test_data.copy(),   geometry=gpd.points_from_xy(sample_test_data.longitude_destination, sample_test_data.latitude_destination),crs={\'init\': \'epsg:4326\'})  \n\n    sjoined_origin = gpd.sjoin(gdf_origin, df_geo, op="within",how="left")\n    sjoined_origin[\'id\']=sjoined_origin[\'id\'].fillna(-1)\n\n    sjoined_destination = gpd.sjoin(gdf_destination, df_geo, op="within",how="left")\n    sjoined_destination[\'id\']=sjoined_destination[\'id\'].fillna(-1)\n\n    sjoined_origin_selected= sjoined_origin[[\'latitude_origin\',\'longitude_origin\',\'timestamp\',\'hour_of_day\',\'day_of_week\',\'id\',\'REGION_N\',\'sequence\']]\n    sjoined_destination_selected= sjoined_destination[[\'latitude_destination\',\'longitude_destination\',\'id\',\'REGION_N\',\'sequence\']]\n\n    sjoined_origin_selected.rename(columns={\'id\': \'pickup_id\',\'REGION_N\':\'REGION_N_pickup\'}, inplace=True)\n    sjoined_destination_selected.rename(columns={\'id\': \'destination_id\',\'REGION_N\':\'REGION_N_dest\'}, inplace=True)\n\n    data_joined= pd.merge(sjoined_origin_selected,sjoined_destination_selected, how=\'left\', on=\'sequence\')\n    data_joined[\'Date\'] = pd.to_datetime(data_joined.timestamp,unit=\'s\')\n    data_joined.rename(columns={\'sequence\': \'trj_id\'}, inplace=True)\n\n    predictedTime,predictedPath,predictedTime2,predictedPath2 =getbaseLineETA(data_joined)\n    data_joined["popularity+time"]=pd.Series(predictedTime)\n    data_joined["popularity+time_path"]=pd.Series(predictedPath)\n    data_joined["timeOnly"]=pd.Series(predictedTime2)\n    data_joined["timeOnlyPath"]=pd.Series(predictedPath2)\n\n    data_joined["dayOfWeek"]=data_joined["Date"].dt.dayofweek\n    data_joined["Day"]=data_joined["Date"].dt.day\n    data_joined["month"]=data_joined["Date"].dt.month\n    data_joined["hour"]=data_joined[\'Date\'].dt.hour\n\n    data_joined["pathHIndex"]=data_joined[\'popularity+time_path\'].apply(lambda x:sum(map(lambda item:h[item],x)))\n    data_joined["pathaIndex"]=data_joined[\'popularity+time_path\'].apply(lambda x:sum(map(lambda item:a[item],x)))\n    data_joined["pathInDegreeIndex"]=data_joined[\'popularity+time_path\'].apply(lambda x:sum(map(lambda item:inCentrality[item],x)))\n    data_joined["pathOutDegreeIndex"]=data_joined[\'popularity+time_path\'].apply(lambda x:sum(map(lambda item:outCentrality[item],x)))\n    data_joined["CountOfGrid"]=data_joined[\'popularity+time_path\'].apply(len)\n    data_joined["destLoad"]=data_joined.destination_id.map(loadCentrality)\n    data_joined["destHIndex"]=data_joined.destination_id.map(h)\n    data_joined["pickUpLoad"]=data_joined.pickup_id.map(loadCentrality)\n    data_joined["flowType"]=data_joined.REGION_N_pickup +"->"+data_joined.REGION_N_dest\n\n    data_joined[\'Date_round\'] = data_joined[\'Date\'].dt.floor(\'h\')\n\n    data_joined1= pd.merge(data_joined,weather_data_filter, how=\'left\', left_on=\'Date_round\', right_on=\'NewDT_round\')\n    data_joined1[\'Temp\'] = data_joined1[\'Temp\'].str.replace(\'°F\', \'\')\n    data_joined1[\'Weather\'] = data_joined1[\'Weather\'].str.replace(\'.\', \'\')\n    data_joined1[\'Wind\'] = data_joined1[\'Wind\'].str.replace(\'mph\', \'\')\n    data_joined1[\'Humidity\'] = data_joined1[\'Humidity\'].str.replace(\'%\', \'\')\n    data_joined1[\'Barometer\'] = data_joined1[\'Barometer\'].str.replace(\'"Hg\', \'\')\n    data_joined1[\'Visibility\'] = data_joined1[\'Visibility\'].str.replace(\'mi\', \'\')\n    data_joined1[\'Wind\']=np.where((data_joined1[\'Wind\'].isnull())|(data_joined1[\'Wind\']==\'No wind\'),\'0\',data_joined1[\'Wind\'] )\n    data_joined1[\'Visibility\']=np.where((data_joined1[\'Visibility\'].isnull())|(data_joined1[\'Visibility\']==\'N/A\'),\'0\',data_joined1[\'Visibility\'] )\n    data_joined1[\'Temp\'] = data_joined1[\'Temp\'].astype(np.float64)\n    data_joined1[\'Wind\'] = data_joined1[\'Wind\'].astype(np.float64)\n    data_joined1[\'Humidity\'] = data_joined1[\'Humidity\'].astype(np.float64)\n    data_joined1[\'Barometer\'] = data_joined1[\'Barometer\'].astype(np.float64)\n    data_joined1[\'Visibility\'] = data_joined1[\'Visibility\'].astype(np.float64)\n\n    df_join2= pd.merge(data_joined1, pickup_pivot_df, how=\'left\', left_on=\'pickup_id\', right_on=\'pickup_id_pick\')\n    df_join3= pd.merge(df_join2,destination_pivot_df, how=\'left\', left_on=\'destination_id\', right_on=\'destination_id_des\')\n    df_original=df_join3[["trj_id",\'latitude_origin\',\'longitude_origin\',\'latitude_destination\',\'longitude_destination\',\'timestamp\',\'hour_of_day\',\'day_of_week\']]\n\n    df_join3["Festival"]=np.where(df_join3[\'day\'].isin([\'2019-04-01\',\'2019-04-18\',\'2019-04-19\',\'2019-04-20\',\'2019-04-21\',\'2019-05-01\',\'2019-05-11\',\'2019-05-12\',\'2019-05-17\',\'2019-05-18\',\'2019-05-19\',\'2019-05-20\']),1,0)\n    df_geo["X"]=df_geo.geometry.centroid.x\n    df_geo["y"]=df_geo.geometry.centroid.y\n    df_join3=df_join3.merge(df_geo[["id","X","y"]],left_on="pickup_id",right_on="id",how="left").merge(df_geo[["id","X","y"]],left_on="destination_id",right_on="id",suffixes=("_pickup","_dest"),how="left")\n    df_join3[(df_join3[\'popularity+time\'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3[\'timeOnlyPath\'].astype(str).apply(lambda x: len(x)) == 0)]\n    df_join3[\'rawlat_pickup\']=df_join3[\'latitude_origin\']\n    df_join3[\'rawlng_pickup\']=df_join3[\'longitude_origin\']\n    df_join3[\'rawlat_dest\']=df_join3[\'latitude_destination\']\n    df_join3[\'rawlng_dest\']=df_join3[\'longitude_destination\']\n    df_join3["rawlng_pickup2"] = df_join3["rawlng_pickup"].apply(radians)\n    df_join3["rawlat_pickup2"] = df_join3["rawlat_pickup"].apply(radians)\n    df_join3["rawlng_dest2"] = df_join3["rawlng_dest"].apply(radians)\n    df_join3["rawlat_dest2"] = df_join3["rawlat_dest"].apply(radians)\n    df_join3["HarvsineDistance"]=haversine(df_join3["rawlng_pickup2"], df_join3["rawlat_pickup2"], df_join3["rawlng_dest2"], df_join3["rawlat_dest2"])\n    df_join3["EuclideanDistance"]=euclidean_distance(df_join3[["rawlng_pickup","rawlat_pickup"]].values, df_join3[["rawlng_dest","rawlat_dest"]].values)\n    df_join3["ManhattanDistance"]=manhattan_distance(df_join3[["rawlng_pickup","rawlat_pickup"]].values, df_join3[["rawlng_dest","rawlat_dest"]].values)\n    df_join3["TimeInterval"]=df_join3.hour.apply(classify)\n    df_join3["PopularPlace"]=0\n    df_join3["Weekend"]=df_join3.dayOfWeek.map({6:1,5:1}).fillna(0)\n    df_join3.loc[df_join3.destination_id.isin([6815,6816,6885,6886,517,587,588,658,659,590,519,589,2877,2876,2875,2947,2946,2945]),\'PopularPlace\']=1\n\n    # can match\n    data_for_model_1=df_join3[~((df_join3[\'popularity+time\'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3[\'timeOnlyPath\'].astype(str).apply(lambda x: len(x)) == 0))]\n    var_drop=[\'latitude_origin\',\'longitude_origin\',\'timestamp\',\'pickup_id\',\'REGION_N_pickup\',\'destination_id\',\'latitude_destination\',\n         \'longitude_destination\',\'Date\',\'popularity+time_path\',\'timeOnlyPath\',\'hour_of_day\',\'day_of_week\',\'month\',\'Date_round\',\n         \'Time\',\'day\',\'Time24\',\'NewDT\',\'session\',\'NewDT_round\',\'pickup_id_pick\',\'destination_id_des\',\n         \'id_pickup\',\'X_pickup\',\'y_pickup\',\'id_dest\',\'X_dest\',\'y_dest\',\'rawlng_pickup2\',\'rawlat_pickup2\',\'rawlng_dest2\',\'rawlat_dest2\',\n           \'EuclideanDistance\',\'ManhattanDistance\',\'Day\']\n    data_for_model_1 = data_for_model_1.drop(columns = var_drop, axis = 1)\n    data_for_model_1.rename(columns={"REGION_N_dest": "destination_Region"}, inplace = True)\n    data_for_model_1[\'Weather\'].fillna(\'Not Available\', inplace=True)\n    trip_Data=data_for_model_1[\'trj_id\'].values\n    data_for_model_1_re=data_for_model_1[[\'destination_Region\',  \'popularity+time\',  \'timeOnly\',  \'dayOfWeek\',  \'hour\',  \'pathHIndex\',  \'pathaIndex\',  \'pathInDegreeIndex\',  \'pathOutDegreeIndex\',  \'CountOfGrid\',  \'destLoad\',  \'destHIndex\',  \'pickUpLoad\',  \'flowType\',  \'Festival\',  \'Temp\',  \'Weather\',  \'Wind\',  \'Humidity\',  \'Barometer\',  \'Visibility\',  \'Bus Terminal_pick\',  \'Business Area_pick\',  \'HDB_pick\',  \'Hospital_pick\',  \'LRT_pick\',  \'MRT_pick\',  \'Mall/Supermarket_pick\',  \'Other Residentials_pick\',  \'School/Kindergarten_pick\',  \'University/College_pick\',  \'Bus Terminal_des\',  \'Business Area_des\',  \'HDB_des\',  \'Hospital_des\',  \'LRT_des\',  \'MRT_des\',  \'Mall/Supermarket_des\',  \'Other Residentials_des\',  \'School/Kindergarten_des\',  \'University/College_des\',  \'rawlat_pickup\',  \'rawlng_pickup\',  \'rawlat_dest\',  \'rawlng_dest\',  \'HarvsineDistance\',  \'TimeInterval\',  \'PopularPlace\',  \'Weekend\']]\n    join_predicted_1=pd.DataFrame()\n    if not data_for_model_1_re.empty:\n        data_for_model_1_re[\'eta\']= loaded_model_1.predict(data_for_model_1_re)\n\n        predicted_values_1=data_for_model_1_re[[\'eta\']]\n        predicted_values_1["trj_id"]=trip_Data\n        join_predicted_1= pd.merge(df_original, predicted_values_1,on="trj_id")\n\n    data_for_model_2=df_join3[(df_join3[\'popularity+time\'].astype(str).apply(lambda x: len(x)) == 0)|(df_join3[\'timeOnlyPath\'].astype(str).apply(lambda x: len(x)) == 0)]\n    if data_for_model_2.shape[0]>=1:\n        data_for_model_2[\'eta\']= loaded_model_2.predict(data_for_model_2[["Festival",\'HarvsineDistance\']])\n        predicted_values_2=data_for_model_2[[\'eta\']]\n        predicted_values_2["trj_id"]=data_for_model_2["trj_id"].values\n        join_predicted_2= pd.merge(df_original, predicted_values_2,on="trj_id")\n        finalprediction= join_predicted_1.append(join_predicted_2)\n\n    else: \n        finalprediction= join_predicted_1.copy()\n\n    return json.dumps(finalprediction.sort_values("trj_id").drop("trj_id",axis=1).eta.tolist())')


# # Deploy the model to Azure Container Instance
# 

# In[108]:


get_ipython().run_cell_magic('time', '', 'image_config = ContainerImage.image_configuration(execution_script="score.py", \n                                                  runtime="python", \n                                                  conda_file="ml.yml")')


# In[ ]:


# from azureml.core.model import InferenceConfig, Model
# from azureml.core.webservice import AciWebservice, Webservice

# inference_config = InferenceConfig(entry_script="score.py",
#                                    environment=myenv)


# # Expose web service
# 

# In[109]:


from azureml.core.webservice import AciWebservice

aciconfig = AciWebservice.deploy_configuration(cpu_cores=2, 
                                               memory_gb=3, 
                                               tags={'sample': 'FinalModel_V1'}, 
                                               description='AZURE_SCORING_ROUTE_ETA')


# In[110]:


get_ipython().run_cell_magic('time', '', "service = Webservice.deploy_from_model(workspace=ws,\n                                       name='final-deploy',\n                                       deployment_config=aciconfig,\n                                       models=[df_geo,G,weather_data_filter,loaded_model_1,loaded_model_2,destination_pivot_df,pickup_pivot_df,h,a,inCentrality,outCentrality,loadCentrality],\n                                       image_config=image_config, overwrite=True)\n\nservice.wait_for_deployment(show_output=True)")


# In[111]:


get_ipython().run_cell_magic('time', '', '\nprint(service.state)\nprint(service.get_logs())')


# In[ ]:


# facing some issues with the new syntaxes
# from azureml.core.model import InferenceConfig, Model
# from azureml.core.webservice import AciWebservice, Webservice
# from azureml.core.environment import Environment

# myenv = Environment(name="myenv")
# # Combine scoring script & environment in Inference configuration
# inference_config = InferenceConfig(entry_script="score.py",
#                                    environment=myenv)

# conda_dep = CondaDependencies()
# conda_dep.add_conda_package("scikit-learn")
# conda_dep.add_conda_package("geopandas")
# conda_dep.add_conda_package("lightgbm")
# conda_dep.add_conda_package("pandas")
# conda_dep.add_conda_package("numpy")
# conda_dep.add_conda_package("joblib")


# # Adds dependencies to PythonSection of myenv
# myenv.python.conda_dependencies=conda_dep

# from azureml.core import ScriptRunConfig, Experiment
# from azureml.core.environment import Environment

# exp = Experiment(name="myexp", workspace = ws)
# # Instantiate environment
# myenv = Environment(name="myenv")

# # Add training script to run config
# runconfig = ScriptRunConfig(source_directory=".", script="score.py")

# # Attach compute target to run config
# runconfig.run_config.target = "local"

# # Attach environment to run config
# runconfig.run_config.environment = myenv

# # Submit run 
# run = exp.submit(runconfig)

# from azureml.core.model import InferenceConfig, Model
# from azureml.core.webservice import AciWebservice, Webservice
# # Combine scoring script & environment in Inference configuration
# inference_config = InferenceConfig(entry_script="score.py",
#                                    environment=myenv)
# # Set deployment configuration
# deployment_config = AciWebservice.deploy_configuration(cpu_cores = 2,
#                                                        memory_gb = 4)

# # Define the model, inference, & deployment configuration and web service name and location to deploy
# service = Model.deploy(workspace = ws,
#                        name = "mywebservice",
#                        models = [weather_data, df_geo, df_poi, model_0, model_1, model_2, model_3, model_4],
#                        inference_config = inference_config,
#                        deployment_config = deployment_config)


# # Get the Web Service URL¶
# 

# In[112]:


print(service.scoring_uri)


# In[113]:


inp = "D:/Kaggle/AngelHack/stupid azure/FinalModel/"
data = pd.read_csv(inp+'sample_all.csv')
sample = pd.read_csv(inp+'sample.csv')


# In[67]:


get_ipython().run_cell_magic('time', '', 'import json\nimport requests\n\n#conversion to pandas\ndef conv_(ch):  #conversion to pandas\n    res = pred.strip(\'][\').split(\', \') \n    res = [s.strip(\'["]\') for s in res]\n    res_pd = pd.DataFrame(res)\n    res_pd.columns = [\'pred\']\n    res_pd[\'pred\'] = pd.to_numeric(res_pd[\'pred\'])\n    return res_pd\n\n\n#scoring function\ndef score_url(df): \n# URL for the web service.\n    scoring_uri = \'http://41698060-7459-44a9-97d6-6b4da799ad9a.southeastasia.azurecontainer.io/score\'\n    # Convert to JSON string.\n    input_data_json = df.to_json()\n    # Set the content type.\n    headers = {\'Content-Type\': \'application/json\'}\n    # Make the request and display the response.\n    resp = requests.post(scoring_uri, input_data_json, headers=headers)\n    # print(resp.text)\n    pred = conv_(resp.text)\n    return pred\n\ndef score_batch(df):\n    pd_all = pd.DataFrame()\n    if df.shape[0]>1000:\n        for i in range(1,len(df)//1000+1,1):\n            a = 1000*(i-1); b = i*1000\n            temp = df.iloc[a:b,:]\n            pred_temp = score_url(temp)\n            pd_all = pd.concat([pd_all, pred_temp], axis = 0)\n    else:\n        pd_all= score_url(df)\n        \n    return pd_all     ')


# In[64]:


pd_all = pd.DataFrame()
if sample.shape[0]>1000:
    for i in range(1,len(sample)//1000+1,1):
        a = 1000*(i-1); b = i*1000
        temp = sample.iloc[a:b,:]
        pred_temp = score_url(temp)
        pd_all = pd.concat([pd_all, temp], axis = 0)
else:
    pd_all= sample.copy()


# # Delete workspace

# In[ ]:


ws.delete()

