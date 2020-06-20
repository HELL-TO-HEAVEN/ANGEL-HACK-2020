# ANGELHACK-2020 (GRAB) 

Route ETA Prediction in Singapore

# The repository consists of: 
1. Scoring file (using Model Deployed in Azure)  
   - 202020_FINAL_TEST_SCORING.ipynb
   - FINAL_ENDPOINTS_SCORING.py (same as above)
   
2. Scoring file (local) - in case  Azure Failed 
   - LOCAL_PREDICTIONS.ipynb (import the sample format file as per required and all inputs are available at ANGEL-HACK-2020
/Additional Data & Models)
   
3. External Data & Model Files 

4. Other Experiments and Initial Works (Codes are slightly messy).


A brief outline of our final model deployment: 

![Model Deployment](https://user-images.githubusercontent.com/7208012/85190155-10aab080-b2e8-11ea-8b35-902b0dffb3d5.png)


# Quick Demo: 

Using 202020_FINAL_TEST_SCORING.ipynb:

(a) If we one to predict the ETA of single route:
![1](https://user-images.githubusercontent.com/7208012/85190250-076e1380-b2e9-11ea-8f8f-734fe8ee617e.PNG)


(b) If we one to predict the ETA of multiple routes in a txt file, we can import the file into Pandas dataframe and run as below:
![2](https://user-images.githubusercontent.com/7208012/85190268-397f7580-b2e9-11ea-92d8-5d0280fa3b20.PNG)

