from Logger.logger_class import Logger
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pickle
from sklearn.preprocessing import StandardScaler


class Encoding:
    """
    Class for encoding the features and labels
    """
    def __int__(self):
        self.log_writer = Logger()
        self.file_object = open('logs/feature_engineering_logs.txt', 'a+')

    def encoding_columns(self, data):
        """
        Method Name: encoding_columns
        Description: This method converts categorical features into numerical features by using one hot encoding.
        Output: Returns a Data Frame with encoded columns.
        On Failure: Raise Exception
        Written By: Aditya Agrawal

        """
        try:
            categorical_ix = ['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
            t = [('cat', OneHotEncoder(handle_unknown='ignore'), categorical_ix)]
            col_transform = ColumnTransformer(transformers=t, remainder='passthrough')
            new_data = pd.DataFrame(col_transform.fit_transform(data))
            file_name = 'models/transformer'
            file = open(file_name, 'wb')
            pickle.dump(col_transform, file)
            return new_data
        except Exception as e:
            raise Exception()

    def target_encoding(self, data):
        """
        Method Name: target_encoding
        Description: This method converts target column values into numerical values by using label encoder.
        Output: Returns a Data Frame with encoded column.
        On Failure: Raise Exception
        Written By: Aditya Agrawal

                """
        try:
            encoder = LabelEncoder()
            new = pd.DataFrame(encoder.fit_transform(data), columns=['Target'])
            file_name = 'models/label_model'
            file = open(file_name, 'wb')
            pickle.dump(encoder, file)
            return new
        except Exception as e:
            raise Exception()

    def scaling(self, data):
        """
        Method Name: scaling
        Description: Method for Applying standard scaler on features.
        Output: Scaled features
        On Failure : Raise Exception
        Written by : Aditya Agrawal

        """
        try:
            scaler = StandardScaler()
            new_data = scaler.fit_transform(data)
            return new_data
        except Exception as e:
            raise e
