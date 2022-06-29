import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import timeit

from sklearn import metrics
from sklearn import model_selection

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