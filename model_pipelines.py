import warnings
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from fetch_separate_datasets import fetch_data
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from plsda import PLSDA

# Silence warning due to imblearn calling BaseEstimator._validate_data instead 
# of sklearn.utils.validation.validate_data. The two are equivalent but 
# BaseEstimator._validate_data is depricated in the sklearn version I'm using.
warnings.filterwarnings(action='ignore', category=FutureWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)


class PLSTransformer(BaseEstimator, TransformerMixin):
    """
    A custom transformer class to get the transform from PLS instead of prediction.
    """
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.pls.fit(X, y)
        return self

    def transform(self, X):
        X_scores = self.pls.transform(X)
        return X_scores
    
class PipelineConstructor:
    """
    For constructing the different pipelines used. There are 3 pipeline categories
    and three different pipelines within each category. The categories are:

        - 'xgb': Uses XGBClassifier as the final estimator
        - 'lgbm: Uses LGBMClassifier as the final estimator
        - 'stack': Uses a stacked model XGB>LGBM>LogisticRegression

    Within each category, there are three different pipelines:
        1) Base model onely
        2) PCA dimensionality + base model
        3) PLS dimensionality reduction + base model

    """
    def __init__(self, column_transformer):
        self.column_transformer = column_transformer

    def fetch_pipelines(self, pipeline = str, *n_components):

        if pipeline == 'plsda':
            model = PLSDA(n_components = n_components[0])
            plsda =  ImbPipeline([
                        ('preprocessor', self.column_transformer),
                        ('smote', SMOTE(random_state=42)),
                        (pipeline, model)
                    ])

        if pipeline == 'xgb':
            objective='multi:softmax'
            num_class=3
            eval_metric='mlogloss'
            random_state=42  

            model = XGBClassifier(
                objective=objective,
                num_class=num_class,
                eval_metric=eval_metric,
                random_state=random_state
            )

        elif pipeline == 'lgbm':
            objective='multiclass'
            num_class=3
            random_state=42
            n_estimators=500
            learning_rate=0.05

            model = LGBMClassifier(
                objective=objective,
                num_class=num_class,
                random_state=random_state,
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                verbosity = -1
            )

        elif pipeline == 'stack':
                objective_xgb='multi:softmax'
                num_class_xgb=3
                eval_metric_xgb='mlogloss'
                random_state_xgb=42
                objective_lgbm='multiclass'
                num_class_lgbm=3
                random_state_lgbm=42
                n_estimators_lgbm=500
                learning_rate_lgbm=0.05
                max_iter_lr=1000
                multi_class_lr='multinomial'

                model = StackingClassifier(
                    estimators=[
                        ('xgb', XGBClassifier(
                            objective=objective_xgb,
                            num_class=num_class_xgb,
                            random_state=random_state_xgb,
                            eval_metric=eval_metric_xgb,
                        )),

                        ('lgbm', LGBMClassifier(
                            objective=objective_lgbm,
                            num_class=num_class_lgbm,
                            random_state=random_state_lgbm,
                            n_estimators=n_estimators_lgbm,
                            learning_rate=learning_rate_lgbm,
                            verbosity = -1
                        ))
                    ],
                
                    final_estimator=LogisticRegression(
                        max_iter=max_iter_lr,
                        multi_class=multi_class_lr
                    ),

                    stack_method='predict_proba',
                )
        
        

        model_only = ImbPipeline([
            ('preprocessor', self.column_transformer),
            ('smote', SMOTE(random_state=42)),
            (pipeline, model)
        ])

        pca_model = ImbPipeline([
            ('preprocessor', self.column_transformer),
            ('smote', SMOTE(random_state=42)),
            ('pca', PCA(n_components=self.PCA_components)),
            (pipeline, model)
        ])

        pls_model = ImbPipeline([
            ('preprocessor', self.column_transformer),
            ('smote', SMOTE(random_state=42)),
            ('pls', PLSTransformer(n_components=self.PLS_components)),
            (pipeline, model), 
        ])

        pipelines = {
            'model_only': model_only,
            'pca_model': pca_model,
            'pls_model': pls_model
        }

        if pipeline == 'plsda':
            return plsda
        else:
            return pipelines

class ModelConstructor(PipelineConstructor):
    """
    For constructing the three different models based on the training data.
        - Model A: Early detection model using only pre-enrollment indicators.
        - Model B: Enrolled detection model using only post-enrollment indicator.
        - Model C: Combined model using both pre and post-enrollment indicators.
    """
    def __init__(
            self, 
            model = str,
    ):
        self.data = fetch_data()

        if model == 'A':
            self.fetch_A()
        elif model == 'B':
            self.fetch_B()
        elif model == 'C':
            self.fetch_C()

        PipelineConstructor.__init__(self, self.column_transformer)
        ## Train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=42
        )
        
    def fetch_A(self):
        # fetch the corresponding data
        self.X = self.data[1]
        self.Y = self.data[4]
        self.onehot_cols = self.data[6]
        self.scale_cols = self.data[8]

        ## Initialize the corresponding column transformer
        self.column_transformer = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(
                # pass categories to handle rare categories that might be missing from training folds 
                categories = [np.sort(self.X[col].unique()) for col in self.onehot_cols],
                drop='if_binary', handle_unknown='ignore', sparse_output=False
            ), self.onehot_cols),
            ('scale', StandardScaler(), self.scale_cols)
        ])
        self.PCA_components = 40
        self.PLS_components = 3

    def fetch_B(self):
        # fetch the corresponding data
        self.X = self.data[2]
        self.Y = self.data[5]
        self.scale_cols = self.data[9]

        ## Initialize the corresponding column transformer
        self.column_transformer = ColumnTransformer(transformers=[
            ('scale', StandardScaler(), self.scale_cols)
        ])
        self.PCA_components = 4
        self.PLS_components = 7

    def fetch_C(self):
        # fetch the corresponding data
        self.X = self.data[3]
        self.Y = self.data[5]
        self.onehot_cols = self.data[6]
        self.scale_cols = self.data[7]

        ## Initialize the corresponding column transformer
        self.column_transformer = ColumnTransformer(transformers=[
            ('onehot', OneHotEncoder(
                categories = [np.sort(self.X[col].unique()) for col in self.onehot_cols],
                drop='if_binary', handle_unknown='ignore', sparse_output=False
            ), self.onehot_cols),
            ('scale', StandardScaler(), self.scale_cols)
        ])
        self.PCA_components = 40
        self.PLS_components = 13
            
    def fetch_splits(self, label_encode = True):
        if label_encode:
            le = LabelEncoder()
            # Ravel applied to handle DataConversionWarning
            # because LabelEncoder expects flat arrays
            y_train = le.fit_transform(self.y_train.values.ravel())
            y_test = le.transform(self.y_test.values.ravel())
            return self.X_train, self.X_test, y_train, y_test
        else:
            return self.X_train, self.X_test, self.y_train, self.y_test
    
    def fetch_inputs(self):
        return self.X, self.Y
