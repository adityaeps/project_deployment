import pandas as pd
from Logger.logger_class import Logger


class LoadingRaw:
    """
    Class for loading raw data from local source
    Written by = Aditya Agrawal
    """

    def __init__(self):
        self.log_writer = Logger()
        self.file_object = open('logs/data_loading_logs.txt', 'a+')

    def load_train(self):
        """
        Method: load_train
        input: None
        output: return train_data as pandas DataFrame
        on fail: return None and log error
        Written by: Aditya Agrawal
        """
        try:
            # read data from csv
            train_data = pd.read_csv('Dataset/train.csv')
            self.log_writer.log(self.file_object, 'train data load successfully')
            return train_data
        except Exception as e:
            self.log_writer.log(self.file_object, 'failed to load train data')
            raise e

    def load_test(self):
        """
        Method: load_test
        input: None
        output : return test_data as pandas DataFrame
        on fail: return None and log error
        Written by: Aditya Agrawal

        """
        try:
            # read data from csv
            test_data = pd.read_csv('Dataset/Test_data.csv')
            self.log_writer.log(self.file_object, 'test data load successfully')
            return test_data
        except Exception as e:
            self.log_writer.log(self.file_object, 'Failed to load test data ')
            raise e
