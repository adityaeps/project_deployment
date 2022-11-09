
"""
This is the Entry point for Training the Machine Learning Model.
Written By: Aditya Agrawal
Version: 1.0
Revisions: None
"""

# Doing the necessary imports
import pickle
import pandas as pd
from data_preprocessing.Data_preprocessing import preprocessing
from Logger.logger_class import Logger
from feature_engineering.Feature_engineering import Encoding
from data_ingestion.loading import LoadingRaw
from sklearn.model_selection import train_test_split
from best_model_finder.best_model_finder import BestModel


class TrainModel:

    def __int__(self):
        self.logging_writer = Logger()
        self.file_object = open("logs/Training_model_logs.txt", 'a+')

    def training_model(self):
        try:
            # Getting the data from the source
            get_data = LoadingRaw()
            train_data = get_data.load_train()
            test_data = get_data.load_test()

            """doing the data preprocessing"""
            preprocessor = preprocessing()
            # remove unwanted columns
            train_data = preprocessor.remove_columns(train_data, ['sl_no', 'salary'])
            test_data = preprocessor.remove_columns(test_data, ['salary'])
            # separating feature and label
            train_x, train_y = preprocessor.separate_label_feature(train_data, label_column_name='status')
            test_x, test_y = preprocessor.separate_label_feature(test_data, label_column_name='status')
            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # And they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop

            """doing the Feature engineering"""
            # Encoding features
            feature_engineering = Encoding()
            train_x = feature_engineering.encoding_columns(train_x)
            column_transformer = pickle.load(open('models/transformer', 'rb'))
            test_x = pd.DataFrame(column_transformer.transform(test_x))
            # Encoding Label column
            train_y = feature_engineering.target_encoding(train_y)
            label_transformer = pickle.load(open('models/label_model', 'rb'))
            test_y = pd.DataFrame(label_transformer.transform(test_y), columns=['Target'])
            # Scaling features columns
            train_x = feature_engineering.scaling(train_x)
            test_x = feature_engineering.scaling(test_x)

            """Finding best model"""
            # train test split
            x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=.25, random_state=300)
            best_model = BestModel()
            model_object, best_estimator = best_model.model_finder(x_train, y_train)
            """Getting evaluation metrics for trained model on train and test data"""
            # On train data
            score_accuracy, score_balance_accuracy, score_roc_auc, report = best_model.scores(x_test, y_test,
                                                                                              model_object)
            # On test data
            test_score_accuracy, test_score_balance_accuracy, test_score_roc_auc, test_report = best_model.scores(
                test_x, test_y, model_object)
            f = open('logs/model_scores.txt', 'a+')
            lines = [str(score_accuracy) + str(score_roc_auc) + str(score_balance_accuracy) + str(report)
                     + str(test_score_accuracy) + str(test_score_balance_accuracy) + str(test_score_roc_auc)
                     + str(test_report)]
            f.writelines(lines)
            # Saving the model
            model_file = 'models/best_model'
            pickle.dump(model_object, open(model_file, 'wb'))
        except Exception as e:
            raise e


train_model = TrainModel()
train_model.training_model()
