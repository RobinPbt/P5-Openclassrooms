# P5-Openclassrooms

This repository has been created in the context of the 5th project of my Data Scientist training with Openclassrooms. 
The objective was to realize a clustering based on a Kaggle dataset (https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) containing customers information of a Brazilian marketplace.

After preprocessing steps, the project has been conducted in the following order:
-	Use a KMeans with few variables (measuring last activity, frequency of orders and amount spent)
-	Add more variables (satisfaction, location of customers) to assess if it still gives relevant results both in term of business interpretation and scoring (silhouette score)
-	With the number of variables giving the best results, run a DBSCAN and a hierarchical clustering
-	Evaluate all three models and retain the most performing one (business interpretation, score, and training time)
-	Give a final business interpretation for each cluster and advise some possible actions
-	Estimate the required maintenance time to keep an accurate model

You can also find the final presentation of the project in the file " P5 - Support de présentation - 2022 07 14" (French)
