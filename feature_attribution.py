import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from model_pipelines import ModelConstructor

def feature_attribution_summary(mcat, pipelines):
    ## Fetch the splits
    mc = ModelConstructor(mcat)
    X_train, X_test, y_train, y_test = mc.fetch_splits()
    ohe = OneHotEncoder(sparse_output=False)

    # one hot to get loadings on each target class
    y_train = ohe.fit_transform(y_train.reshape(-1, 1))
    y_test = ohe.fit_transform(y_test.reshape(-1, 1))

    # Fetch the necessary pipeline steps
    column_transformer = pipelines[mcat].named_steps['preprocessor']
    smote = pipelines[mcat].named_steps['smote']
    pls = pipelines[mcat].named_steps['pls'].pls

    # Fetch the feature names   
    feature_names = column_transformer.get_feature_names_out() #*

    # Apply the column transformer
    X_train = column_transformer.fit_transform(X_train)

    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = column_transformer.fit_transform(X_test)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    # Apply SMOTE
    X_train, y_train = smote.fit_resample(X_train, y_train)

    ## First, narrow down the 3 most important latent PLS features based
    ## on how important they are in the XGBClassifier 
    fi = pipelines[mcat].named_steps['xgb'].feature_importances_
    fi_i = np.argsort(fi)[::-1]

    ## Fetch the PLS model
    pls = pipelines[mcat].named_steps['pls'].pls
    pls.fit(X_train, y_train)

    # features x components
    x_rotations = pls.x_rotations_ # How much each feature loads onto each component
    y_loadings = pls.y_loadings_ # How much each component loads onto each target variables

    # sorted from importance based on the XGB critereon
    sorted_xrot = x_rotations[:,fi_i] 
    sorted_yload = y_loadings[:,fi_i]

    return {
        'feature_names': feature_names,
        'feature_importances': fi,
        'fi_index': fi_i,
        'sorted_feature_importances': fi[fi_i],
        'x_rotations': x_rotations,
        'y_loadings': y_loadings,
        'sorted_xrot': sorted_xrot,
        'sorted_yload': sorted_yload
    }

vdesc = pd.read_csv('variable_descriptions.csv')
def get_feature_labels(feature_names):
    # Get clean feature labels
    
    feature_labels = []
    for i, name in enumerate(feature_names):
        splt = name.split('_')
        vartype = splt[0]

        description = vdesc.loc[vdesc.name == 'Marital Status']['description'].values[0]
        if vartype == 'onehot':
            identifier = splt[-1]
            name = splt[-2]
            description = vdesc.loc[vdesc.name == name]['description'].values[0]
            
            id_i = description.index(f"{identifier} ")+2
            search = re.search(r'\s\d{1,4}\s[-–]', description[id_i:])
            if search == None:
                subname = description[id_i:]
                nxt = ''
            else:
                nxt = id_i+search.span()[0]
                subname = description[id_i:nxt]
            
            feature_labels.append(f"{name} {subname}")
            
        if vartype == 'scale':
            name = splt[-1]
            description = vdesc.loc[vdesc.name == name]['description'].values[0]

            feature_labels.append(name)

    return np.array(feature_labels)
