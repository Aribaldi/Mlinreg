from matrix import *
import numpy as np

class Analytic_sol():
    def __init__(self, xdata, ydata, params=None):
        self.coeffs = []
        self.xdata = xdata
        self.ydata = ydata

    def fit(self):
        first_multiplyer = np.linalg.inv(matrix_multiply(transpose(self.xdata), self.xdata)) #на дефолтном питоне ОЧЕНЬ долго
        second_multiplyer = transpose(self.xdata)

        F_cross = matrix_multiply(first_multiplyer, second_multiplyer)
        self.coeffs = matrix_multiply(F_cross, self.ydata)


    def predict(self, x_data):
        predict = []
        for idx, data in enumerate(x_data):
            data = add_one_for_bias(data)
            result = matrix_multiply([data], self.coeffs)
            predict.append(result)

        return predict