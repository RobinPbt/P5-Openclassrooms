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

from matplotlib.collections import LineCollection

# ---------------------------------Cleaning functions--------------------------------------------------

def var_del(df, liste_var_del: list):
    """Function which delete in df the variables included in liste_var_del"""
    
    for var in liste_var_del:
        del df[var]


def remove_outlier_IQR(df, variable: str):
    """Function which applies a IQR to detect outliers of a given variable and deletes these outliers in the df"""
    
    Q1 = df[variable].quantile(0.25)
    Q3 = df[variable].quantile(0.75)
    IQR = Q3-Q1
    df_final = df[~((df[variable]<(Q1-1.5*IQR)) | (df[variable]>(Q3+1.5*IQR)))]
    return df_final


def drop_zeros(df, variable: str):
    """Function which drop rows with values equals 0 for a given variable on a given df"""
    
    idx_list = list(df[df[variable] == 0].index)

    df.drop(index=idx_list, inplace=True)


def val_correction(df, variable : str, correction_dict : dict):
    """Function which takes a dictionary with keys as values to replace and values for replacment and operates the replacment 
    on a given variable for a given df"""

    for inc_val, cor_val in correction_dict.items():
        df.loc[df[variable] == inc_val, variable] = cor_val


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

# ---------------------------------Analysis functions--------------------------------------------------

def eta_squared(x,y):
    """Function which computes eta squared between x and y"""
    
    moyenne_y = y.mean()
    classes = []
    for classe in x.unique():
        yi_classe = y[x==classe]
        classes.append({'ni': len(yi_classe),
                        'moyenne_classe': yi_classe.mean()})
    SCT = sum([(yj-moyenne_y)**2 for yj in y])
    SCE = sum([c['ni']*(c['moyenne_classe']-moyenne_y)**2 for c in classes])
    return SCE/SCT


def anova_plot(df, qualitative_var, quantitative_var, figsize=(8,6)):
    """Function which creates a boxplot between a qualitative_var and a quantitative_var in a df and displays eta squared"""
    
    figure = plt.figure(figsize=figsize)
    ax = plt.axes()

    modalites = df[qualitative_var].sort_values(ascending=False).unique()
    groupes = []

    for m in modalites:
        groupes.append(df[df[qualitative_var]==m][quantitative_var])

    medianprops = {'color':"black"}
    meanprops = {'marker':'o', 'markeredgecolor':'black',
                'markerfacecolor':'firebrick'}

    h = plt.boxplot(groupes, labels=modalites, showfliers=False, medianprops=medianprops, 
                vert=False, patch_artist=True, showmeans=True, meanprops=meanprops)

    ax = ax.set(xlabel=quantitative_var, 
                ylabel=qualitative_var, 
                title='eta squared = {:.3f}'.format(eta_squared(df[qualitative_var],df[quantitative_var])))

    
# ---------------------------------PCA functions-------------------------------------------------

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    """Function which displays correlation circles of a pca"""
    
    for d1, d2 in axis_ranks:
        if d2 < n_comp:

            # Initialization
            fig, ax = plt.subplots(figsize=(7,6))

            # Determining chart limits
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # displays arrows
            # if there are more than 30 arrows, we don't display the triangle
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (see : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # displays variables names  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # displays circle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # defines charts limits
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # displays horizontal and vertical lines
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # axis names with explained variance ratio
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_components(pca, nb_comp, features):
    """Displays the decomposition of a pca component computation"""

    pca_results_df = pd.DataFrame()
    
    for i, component in enumerate(pca.components_):
        pca_results_df['F{}'.format(i+1)] = component
    
    pca_results_df.index = features
    
    print("Component F{} :".format(nb_comp))
    print("Explained variance : {} %".format(round(100*pca.explained_variance_ratio_[nb_comp-1],1)))
    print('{}'.format('-'*80))
    print('')
    print('Top 10 positively correlated features to component :')
    print('')
    print(pca_results_df.sort_values(by='F{}'.format(nb_comp), ascending=False).iloc[:10]['F{}'.format(nb_comp)])
    print('{}'.format('-'*80))
    print('')
    print('Top 10 negatively correlated features to component :')
    print('')
    print(pca_results_df.sort_values(by='F{}'.format(nb_comp), ascending=True).iloc[:10]['F{}'.format(nb_comp)])

    
def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)
    
    
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