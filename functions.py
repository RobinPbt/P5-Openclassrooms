import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import timeit

from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from matplotlib.collections import LineCollection

from scipy.cluster.hierarchy import dendrogram

from datetime import datetime, timezone

# ---------------------------------Cleaning functions--------------------------------------------------

def select_one_mode_value(df, variable:str):
    """ After an aggregation of mode values in a dataframe for a given variable, this function selects only one mode value if several are returned during aggregation. Return a list"""

    val_list = []
    
    for val in df[variable]:
        
        if isinstance(val, str): # If we have only one value, the type is a str. We keep this value
            val_list.append(val)
        
        elif isinstance(val, np.ndarray):
            if val.size == 0: # If the value is a NaN we have an empty array
                val_list.append(np.nan)
            else: # If we have several value, it's stored in a nparray, we take only the first value of this array
                val_list.append(val[0])

    return val_list
    
    
# ---------------------------------Modeling functions-------------------------------------------------

def compute_time_diff(df, var:str):
    """Function which transforms a time variable in a dataframe in a number of days differential with a reference date (2016-01-01 00:00:01)"""
    
    df[var] = df[var].apply(lambda p: (p - pd._libs.tslibs.timestamps.Timestamp('2016-01-01 00:00:01')).days)

    
def test_KMeans(k_clusters:range, X):
    """Function which tests values in k_clusters for a KMeans (KMeans++ with one iteration) and displays inertia, 
    davies bouldin and silhouettes scores. X must be processed"""
    
    inertia_scores = []
    silhouette_scores = []
    davies_bouldin_scores = []
    
    for k in k_clusters:
    
        model = KMeans(n_clusters=k, random_state=0)
        model.fit(X)

        inertia_scores.append(-model.score(X))
        silhouette_scores.append(metrics.silhouette_score(X, model.labels_))
        davies_bouldin_scores.append(metrics.davies_bouldin_score(X, model.labels_))
        
    
    plt.figure(figsize=(19,5))
    
    plt.subplot(1,3,1)
    plt.plot(k_clusters, inertia_scores)
    plt.xlabel('number of clusters')
    plt.title('inertia score')
    
    plt.subplot(1,3,2)
    plt.plot(k_clusters, silhouette_scores)
    plt.xlabel('number of clusters')
    plt.title('silhouette score')
    
    plt.subplot(1,3,3)
    plt.plot(k_clusters, davies_bouldin_scores)
    plt.xlabel('number of clusters')
    plt.title('davies bouldin score')
    
    return pd.DataFrame({'K' : [i for i in k_clusters], 'inertia_scores' : inertia_scores, 
                         'silhouette_scores' : silhouette_scores, 'davies_bouldin_scores' : davies_bouldin_scores})


def plot_silhouette(model, X):
    """Function which plot the silhouette score of each cluster of a model. X must be preprocessed"""
    
    visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
    visualizer.fit(X)
    visualizer.show()


def display_clustering_2D(model, X):
    """Function which performs a PCA to display the result of a clustering in 2D. Model must be fitted and X preprocessed"""

    # Perform PCA with 2 components and displays variance explained by the first axis and the sum of the 2 first axis
    pca = decomposition.PCA(n_components=2)
    pca.fit(X)
    print("Total explained variance by 2 components : {:.2%}".format(pca.explained_variance_ratio_.cumsum()[1]))
    X_trans = pca.transform(X)

    # Plot the dataset in the 2 first components of the pca
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X_trans[:,0], X_trans[:,1], c=model.labels_)
    plt.xlabel("First inertia axis ({:.2%} of variance)".format(pca.explained_variance_ratio_[0]))
    plt.ylabel("Second inertia axis ({:.2%} of variance)".format(pca.explained_variance_ratio_[1]))
    plt.show()

    
def display_clustering_2D_TSNE(model, X, perplexity=40):
    """Function which performs a TSNE to display the result of a clustering in 2D. Model must be fitted and X preprocessed"""

    # Perform TSNE with 2 components
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=3000, init='pca')
    X_trans = tsne.fit_transform(X)

    # Plot the dataset in the 2 first components of the pca
    fig = plt.figure(figsize=(8,6))
    plt.scatter(X_trans[:,0], X_trans[:,1], c=model.labels_)
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.show()


def display_clusters_characteristics(model, X):
    """Function which returns a dataframe with main informations for each cluster after a clustering. 
    Arguments : model must be fitted and X is the initial matrix not processed"""
    
    nb_clusters = max(model.labels_) + 1

    df_clusters = pd.DataFrame(index=['Cluster {}'.format(i+1) for i in range(0, nb_clusters)])


    # Computing nb_customers and percentage of total customers per cluster
    nb_customers = []
    prop_customers = []

    for i in range(0, nb_clusters):
        nb_customers.append(sum(model.labels_ == i))
        prop_customers.append(sum(model.labels_ == i) / len(model.labels_))

    df_clusters['nb_customers'] = nb_customers
    df_clusters['prop_customers'] = prop_customers
    
    # Compute variables characteristics, with a distinction between categorical and numeric variables
    variables = X.columns
    
    for var in variables:
        
        # If the variable is an object (categorical variable) compute the mode and count the percentage represented by it
        if X[var].dtypes == 'object':
            df_clusters['mode_{}'.format(var)] = [0 for i in range(0, nb_clusters)]
            df_clusters['mode_%_{}'.format(var)] = [0 for i in range(0, nb_clusters)]
    
            for i in range(0, nb_clusters):
                temp_df = X[model.labels_ == i]
                df_clusters.loc['Cluster {}'.format(i+1), 'mode_{}'.format(var)] = temp_df[var].mode()[0]
                df_clusters.loc['Cluster {}'.format(i+1), 'mode_%_{}'.format(var)] = len(temp_df[temp_df[var] == temp_df[var].mode()[0]]) / len(temp_df)  
         
        # Else, the variable is numeric and we compute mean, max and min for each cluster
        else:
            df_clusters['mean_{}'.format(var)] = [0 for i in range(0, nb_clusters)]
            df_clusters['max_{}'.format(var)] = [0 for i in range(0, nb_clusters)]
            df_clusters['min_{}'.format(var)] = [0 for i in range(0, nb_clusters)]

            for i in range(0, nb_clusters):
                df_clusters.loc['Cluster {}'.format(i+1), 'mean_{}'.format(var)] = X[model.labels_ == i][var].mean()
                df_clusters.loc['Cluster {}'.format(i+1), 'max_{}'.format(var)] = X[model.labels_ == i][var].max()
                df_clusters.loc['Cluster {}'.format(i+1), 'min_{}'.format(var)] = X[model.labels_ == i][var].min()
    
    return df_clusters


def plot_variables_distributions_clusters(model, X, figsize=(20,20)):
    """Visualisation of distribution of variables for each cluster.
    Arguments : model must be fitted and X is the initial matrix not processed"""
    
    nb_clusters = max(model.labels_) + 1
    variables = X.columns
    nb_var = X.shape[1]

    plt.figure(figsize=figsize)

    for i in range(0,nb_clusters):

        for j, var in enumerate(variables):
            plt.subplot(nb_clusters, nb_var, (nb_var*i) + (j+1))
            plt.hist(X[model.labels_ == i][var], bins=50)
            plt.title('Cluster {} : {}'.format(i+1, var))

            
def plot_variables_aggregated_distributions_clusters(model, X, figsize=(18,6)):
    """Visualisation of distribution of variables for each cluster.
    Arguments : model must be fitted and X is the initial matrix not processed"""
    
    nb_clusters = max(model.labels_) + 1
    variables = X.columns
    nb_var = X.shape[1]

    plt.figure(figsize=figsize)
    
    for i, var in enumerate(variables):
        plt.subplot(1,nb_var,i+1)
        
        for j in range(0, nb_clusters):
            plt.hist(X[model.labels_ == j][var], bins=50, label='Cluster {}'.format(j+1))

        plt.legend()
        plt.title(var)


def relation_variable_clusters(model, X, figsize=(20,7)):
    """Visualisation of relation between clusters and variables with boxplots.
    Arguments : model must be fitted and X is the initial matrix not processed"""
    
    nb_clusters = max(model.labels_) + 1
    variables = X.columns
    nb_var = X.shape[1]
    
    plt.figure(figsize=figsize)

    for i, var in enumerate(variables):
        plt.subplot(1, nb_var, i+1)
        sns.boxplot(x=model.labels_, y=var, data=X)
        plt.title(var)
        plt.xlabel('Clusters')
        plt.ylabel(None)
        plt.xticks(ticks=[i for i in range(0, nb_clusters)], labels=[i+1 for i in range(0, nb_clusters)])


def test_DBSCAN(X, eps_range=np.linspace(0.5,2,4), min_samples=50):
    """Function which test a DBSCAN model with the given parameters eps and min_sample (all other parameters being the ones by default) and gives for each test the silhouette score, the davies bouldin score and the number of clusters. It then plots the results. X must be processed"""

    silhouette_scores = []
    davies_bouldin_scores = []
    nb_clusters = []

    # Computing the scores for each configuration of eps_range and min_sample
    for eps_param in eps_range:
        model = DBSCAN(eps=eps_param, min_samples=min_samples, metric='euclidean') 
        model.fit(X)

        silhouette_scores.append(metrics.silhouette_score(X, model.labels_))
        davies_bouldin_scores.append(metrics.davies_bouldin_score(X, model.labels_))
        nb_clusters.append(max(model.labels_) + 1)

    
    # Plotting the results
    plt.figure(figsize=(19,5))

    plt.subplot(1,3,1)
    plt.plot(eps_range, nb_clusters)
    plt.xlabel('eps')
    plt.title('number of clusters')

    plt.subplot(1,3,2)
    plt.plot(eps_range, silhouette_scores)
    plt.xlabel('eps')
    plt.title('silhouette score')

    plt.subplot(1,3,3)
    plt.plot(eps_range, davies_bouldin_scores)
    plt.xlabel('eps')
    plt.title('davies bouldin score')
    
    return pd.DataFrame({'eps' : [i for i in eps_range], 'nb_clusters' : nb_clusters, 
                     'silhouette_scores' : silhouette_scores, 'davies_bouldin_scores' : davies_bouldin_scores})


def plot_dendrogram(model, **kwargs):
    """Create linkage matrix and then plot the dendrogram"""

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
    
    
# ---------------------------------Maintenance functions-------------------------------------------------

def filter_dataset(reference_date, full_data):
    """Function which filters the dataset by taking all orders before a reference date 
    and giving synthetic variables that can be used in the machine learning algorithm. 
    Specific function built for this project only"""
    
    # Selecting orders before a reference date
    selected_data = full_data[full_data['order_purchase_timestamp'] < reference_date]
     
    # Create df to store our variables
    data = pd.DataFrame()
    data['customer_unique_id'] = selected_data['customer_unique_id'].unique()
    
    # Adding number of orders per customer
    temp_df = pd.DataFrame()
    temp_df['customer_unique_id'] = selected_data['customer_unique_id'].value_counts().index
    temp_df['nb_orders'] = selected_data['customer_unique_id'].value_counts().values
    data = pd.merge(data, temp_df, how='left', on='customer_unique_id')

    # Adding and transforming last order date by customer
    temp_df = pd.DataFrame()
    temp_df['last_order_time'] = selected_data.groupby(by='customer_unique_id')['order_purchase_timestamp'].max()
    temp_df['last_order_time'] = pd.to_datetime(temp_df['last_order_time'], infer_datetime_format=True, errors='raise')
    temp_df['last_order_time'] = temp_df['last_order_time'].apply(datetime.timestamp)
    data = pd.merge(data, temp_df, how='left', on='customer_unique_id')

    # Adding variable mean_review_score
    temp_df = pd.DataFrame()
    temp_df['mean_review_score'] = selected_data.groupby(by='customer_unique_id')['review_score'].mean()
    data = pd.merge(data, temp_df, how='left', on='customer_unique_id')
    
    # Adding variable total_payment_value
    temp_df = pd.DataFrame()
    temp_df['total_payment_value'] = selected_data.groupby(by='customer_unique_id')['total_payment_value'].sum()
    data = pd.merge(data, temp_df, how='left', on='customer_unique_id')
    
    # Withdrawing customer_unique_id
    data.drop(columns='customer_unique_id', inplace=True)
    
    return data


def fitting_reference_model(reference_date, full_data):
    """Fitting the reference_model with at reference_date"""

    # Creating a dataset with all orders from begining to reference date
    X = filter_dataset(reference_date, full_data)

    # Creating reference preprocessor and fitting
    reference_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
        ('stdscaler', StandardScaler())])

    processed_X = reference_preprocessor.fit_transform(X)

    # Creating and fitting reference model
    reference_model = KMeans(n_clusters=5, random_state=0)
    reference_model.fit(processed_X)
    
    return reference_preprocessor, reference_model


def compute_ARI_scores(reference_date, full_data, delta_days=7, nb_delta_days=16):
    """Computing ARI at different dates intervals to see the evolution"""

    # Fitting reference_preprocessor and reference_model at reference_date
    reference_preprocessor, reference_model = fitting_reference_model(reference_date, full_data)
    
    time_delta = pd._libs.tslibs.timedeltas.Timedelta(delta_days, unit='D')

    ARI_scores = []
    dates = []

    for i in range(nb_delta_days): # Compute over delta_days * nb_delta_days

        # Computing new date adding the time_delta
        date = reference_date + ((i+1) * time_delta)
        dates.append(date)

        # Creating the X matrix with orders until this new dates (thus adding all orders during time_delta)
        X_new = filter_dataset(date, full_data)

        # New preprocessor
        preprocessor_new = Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')),
        ('stdscaler', StandardScaler())])

        # Creating to preprocessed matrix, one with the reference_preprocessor (used in reference_model), and one with a new one
        processed_X_new = preprocessor_new.fit_transform(X_new)
        processed_X_reference = reference_preprocessor.transform(X_new)

        # Creating and fitting new model
        new_model = KMeans(n_clusters=5, random_state=0)
        new_model.fit(processed_X_new)

        # Compute ARI score to see similarity with prediction of old model vs. a new model fitted on new datas
        ARI_scores.append(adjusted_rand_score(reference_model.predict(processed_X_reference), new_model.labels_))
        
    return dates, ARI_scores


def compute_maintenance_time(reference_date, full_data, delta_days=7, nb_delta_days=16, ARI_treshold=0.8):
    """Function which computes and plots ARI scores of the dataset at different dates to see evolution. 
    Tells when we reach ARI_treshold which mean inital model is obsolete.
    Specific function built for this project only"""
    
    dates, ARI_scores = compute_ARI_scores(reference_date, full_data, delta_days=delta_days, nb_delta_days=nb_delta_days)
    
    # Compute obsolescence time
    for ARI, date in zip(ARI_scores, dates):
        if ARI < ARI_treshold:
            print("Reference date : {}-{}-{}".format(reference_date.year, reference_date.month, reference_date.day))
            print("Date from which reference model isn't working well enough : {}-{}-{}".format(date.year, date.month, date.day))
            print("In days from reference date : {}".format(-(reference_date - date).days))
            print("In weeks from reference date : {:.0f}".format(int((-(reference_date - date).days)) / 7))
            break
    
    # Plotting results
    plt.figure(figsize=(12,8))
    plt.plot(dates, ARI_scores)
    plt.xlabel('dates')
    plt.ylabel('ARI score')
    plt.show()


def compute_times(model, X):
    """Function which computes training and predict time of a model"""
    
    # Train time
    start_time = timeit.default_timer()
    model.fit(X)
    fit_time = timeit.default_timer() - start_time

    # Predict time
    start_time = timeit.default_timer()
    prediction = model.predict(X)
    predict_time = timeit.default_timer() - start_time
    
    times = [fit_time, predict_time]
    
    print("Fit time : {:.2f}s".format(fit_time))
    print("Predict time : {:.2f}s".format(predict_time))
    
    return times