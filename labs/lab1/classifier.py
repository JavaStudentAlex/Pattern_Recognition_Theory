import pandas as pd
import numpy as np


class Classifier:
    __classes_centers = None
    __features = None

    def fit(self, df: pd.DataFrame, features: list, target: str):
        self.__features = features
        self.__classes_centers = df.groupby(by=target).mean()[features]

    def predict(self, df: pd.DataFrame):
        predicted_classes = []
        for current_measure_index in df.index:
            current_measure = df.loc[current_measure_index, self.__features].values
            current_predicted_class = self.__get_nearest_class(current_measure)
            predicted_classes.append(current_predicted_class)
        return predicted_classes

    def __get_nearest_class(self, measure):
        best_class_index = self.__get_best_class_index(measure)
        return self.__classes_centers.index[best_class_index]

    def __get_best_class_index(self, measure):
        tilled_measure = np.tile(measure, (len(self.__classes_centers.index), 1))
        difference = self.__classes_centers.values - tilled_measure
        squares = np.square(difference)
        sums = np.sum(squares, axis=1)
        return np.argmin(sums)
