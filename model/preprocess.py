from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale, StandardScaler


class Preprocess:
    def scale_and_split(self, x_train, file):
        x_trained = file[x_train]
        scaled = scale(self.__to_float(x_trained))
        return scaled

    def __to_float(self, input):
        return input.astype(float)


    def prepare_input(self, x, file):
        X = file[x]
        scaled = StandardScaler()
        scaled = scaled.fit_transform(X.astype(float))
        return scaled
