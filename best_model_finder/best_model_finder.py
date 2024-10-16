from Logger.logger_class import Logger
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold


class BestModel:

    def __init__(self):
        self.log_writer = Logger()
        self.file_object = open('logs/model_finding_logs.txt', 'a+')

    def model_finder(self, x_train, y_train):
        """
        Method Name: model_finder
        Description: Method for finding best model using pipeline and grid search cv.
        Output: Grid cv object and saved model file name
        On Failure : Raise Exception
        Written by : Aditya Agrawal

        """
        try:
            kfold = StratifiedKFold()
            pipe = Pipeline(steps=[('classifier', RandomForestClassifier())])
            param = [{'classifier': [RandomForestClassifier()],
                      "classifier__n_estimators": [10, 50, 100, 130],
                      "classifier__criterion": ['gini', 'entropy'],
                      "classifier__max_depth": range(2, 4, 1),
                      "classifier__max_features": ['auto', 'log2']},
                     {'classifier': [XGBClassifier()],
                      'classifier__learning_rate': [0.5, 0.1, 0.01, 0.001],
                      'classifier__max_depth': [3, 5, 10, 20],
                      'classifier__n_estimators': [10, 50, 100, 200]},
                     {'classifier': [SVC()],
                      'classifier__C': [1, 10, 100, 1000, 10000],
                      'classifier__gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'classifier__kernel': ['rbf', 'linear']}]
            grid_cv = GridSearchCV(estimator=pipe, param_grid=param, scoring='f1', cv=kfold)
            grid_cv.fit(x_train, y_train)
            best_model = grid_cv.best_estimator_['classifier']
            return grid_cv, best_model
        except Exception as e:
            raise Exception()

    def scores(self, x_test, y_test, model_object):
        """
        Method Name: scoring
        Description: Method for finding scoring parameters on test data.
        Output: scoring metrics.
        On Failure : Raise Exception
        Written by : Aditya Agrawal

        """
        try:
            model = model_object
            y_prediction = model.predict(x_test)
            score_accuracy = accuracy_score(y_test, y_prediction)
            score_roc_auc = roc_auc_score(y_test, y_prediction)
            score_balance_accuracy = balanced_accuracy_score(y_test, y_prediction)
            report = classification_report(y_test, y_prediction)
            return score_accuracy, score_balance_accuracy, score_roc_auc, report
        except Exception as e:
            raise Exception()
