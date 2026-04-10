import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo 

## fetch dataset 
try: 
    predict_students_dropout_and_academic_success = fetch_ucirepo(id=31) 
    # # data (as pandas dataframes) 
    X = predict_students_dropout_and_academic_success.data.features 
    Y = predict_students_dropout_and_academic_success.data.targets 

except ConnectionError:

    ## In case of down connection to ucimlrepo 
    data = pd.read_csv('data.csv')
    col = ['Marital Status'] + list(data.columns[0].split(';'))[1:]
    col[4] = col[4].replace('\t', '').replace('"', '')
    d = [str(row[0]).split(';') for row in data.values]
    data = pd.DataFrame(np.array(d[1:]), columns=col)
    X = data[list(data.columns)[:-1]]
    Y = pd.DataFrame(data[list(data.columns)[-1]])
    dfvar = pd.read_csv('variable_descriptions.csv')
    typed_data = []
    for c in col[:-1]:
        if dfvar.loc[dfvar.name == c].type.values[0] == 'Integer':
            typed_data.append([int(float(x)) for x in X[c]])
        elif dfvar.loc[dfvar.name == c].type.values[0] == 'Continuous':
            typed_data.append(X[c].astype(float))

    X = pd.DataFrame(np.array(typed_data).T, columns=col[:-1])

## Split variables based on type. Each variable type gets a different scaling treatment.

# Nominal categorical and binary variables (Apply one-hot encoding)
onehot_cols = [
    'Marital Status',
    'Application mode',
    'Course',
    'Nacionality',
    "Mother's occupation",
    "Father's occupation",
    'Daytime/evening attendance',
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Gender',
    'Scholarship holder',
    'International'
]

# Ordinal categorical and continuous variables (Apply Z-score normalization)
scale_cols = [
    'Previous qualification',
    "Mother's qualification",
    "Father's qualification",
    'Application order',
    'Age at enrollment',
    'Previous qualification (grade)',
    'Admission grade',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (without evaluations)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# For the early detection model
scale_cols_pre = [
    'Previous qualification',
    "Mother's qualification",
    "Father's qualification",
    'Application order',
    'Age at enrollment',
    'Previous qualification (grade)',
    'Admission grade',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# For the post enrollment detection model
scale_cols_post = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (without evaluations)',
    'Curricular units 1st sem (grade)',
    'Curricular units 2nd sem (grade)',
]

def fetch_anomalous_samples(X, onehot_cols, scale_cols):
    # Initialize the column transformer
    column_transformer = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False), onehot_cols),
        ('scale', StandardScaler(), scale_cols)
    ])

    # Transform the data for the purpose of fetching the anomalous samples
    recoded_data = column_transformer.fit_transform(X)
    feature_names = column_transformer.get_feature_names_out()
    X_transformed = pd.DataFrame(recoded_data, columns=feature_names)

    # Fetch the anomalous samples
    pca = PCA()
    pca_transformed_x = pca.fit_transform(recoded_data)
    anomalous_samples = np.where(pca_transformed_x[:,0]<-5)[0]

    return anomalous_samples

def fetch_pre_and_post_vars(X):
    # Fetch the pre and post enrollment variables
    post_enrollment_var = list(X.columns[-15:-3])
    pre_enrollment_var = [x for x in list(X.columns) if x not in list(X.columns[-15:-3])]

    return pre_enrollment_var, post_enrollment_var

def fetch_data():
    anomalous_samples = fetch_anomalous_samples(X, onehot_cols, scale_cols)
    pre_enrollment_var, post_enrollment_var = fetch_pre_and_post_vars(X)

    # Build the datasets
    X_pre = X[pre_enrollment_var]
    X_post = X[post_enrollment_var].iloc[~X.index.isin(anomalous_samples)]
    X_combo = X[~X.index.isin(anomalous_samples)]

    Y_exclude_anomalous = Y[~Y.index.isin(anomalous_samples)]

    return X, X_pre, X_post, X_combo, Y, Y_exclude_anomalous, onehot_cols, scale_cols, scale_cols_pre, scale_cols_post
