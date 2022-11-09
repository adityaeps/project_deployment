import pandas as pd
import numpy as np
from Logger.logger_class import Logger


class Transformation:
    """
     This class shall be used for transforming the data before preprocessing
     Written By: Aditya Agrawal

    """
    def __int__(self):
        self.logging_writer = Logger()
        self.file = open('logs/data_transformation_logs.txt', 'a+')

    def is_null_present(self, data):
        """
        Method Name: is_null_present
        Description: This method checks whether there are null values present in the pandas Dataframe or not.
        Output: Returns a Boolean Value.True if null values are present in the DataFrame, False if they are not present.
        On Failure: Raise Exception
        Written By: Aditya Agrawal

        """
        null_present = False
        try:
            null_counts = data.isna().sum()  # check for the count of null values per column
            for i in null_counts:
                if i > 0:
                    null_present = True
                    break
            if null_present is True:  # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing values count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv(
                    'preprocessing_data/null_values.csv')  # storing the null column information to file
            self.logging_writer.log(self.file, 'Finding missing values is a success.Data written in null values file')
            return null_present
        except Exception as e:
            self.logging_writer.log(self.file, 'Exception occurs in is_null_present method. Exception message:  ' + str(
                                    e))
            self.logging_writer.log(self.file, 'Finding missing values failed.')
            raise Exception()

    def imputing_missing_values(self, data):
        """
        Method Name: impute_missing_values
        Description: This method replaces all the missing values in the Dataframe.
        Output: A Dataframe which has all the missing values imputed.
        On Failure: Raise Exception
        Written By: Aditya Agrawal
        """

        data = data
        categorical_columns = []
        try:
            for i in data.columns:
                if data[i].dtype == 'O':
                    categorical_columns.append(i)
            # here we are replacing null values for categorical columns with mode value
            for i in categorical_columns:
                data[i] = data[i].fillna(data[i].mode())
            # For all numerical columns we are replacing null values with mean value
            for i in data.columns:
                if i not in categorical_columns:
                    data[i] = data[i].fillna(data[i].mean())
            self.logging_writer.log(self.file, 'Missing values are successfully imputed')
            return data
        except Exception as e:
            self.logging_writer.log(self.file, 'Error occurs while imputing  missing values')
            raise e
