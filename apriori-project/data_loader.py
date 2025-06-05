
############################################
#Import panda libraries, pollars, and ...
#
############################################


############################################
#data_loader.py FILE for Online Retake
# 
# EDUARDO JUNQUEIRA 
# TOPIC 1 ASSOCIATION RULES WITH APRIORI ALGORITHM PROJECT EDAMI
# 
#  ALL FILES .PY ARE EXECUTED HERE IN main.py

############################################



#######################################
## Import from Machine Learning Repository DataSet Online Retake
#######################################
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset Online Retake Import 
online_retail = fetch_ucirepo(id=352) 
  
# data (as pandas dataframes) 
X = online_retail.data.features 
y = online_retail.data.targets 
  
# metadata 
print(online_retail.metadata) 
  
# variable information 
print(online_retail.variables) 

############################################
#
#
############################################