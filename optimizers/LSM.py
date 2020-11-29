from matrix import *

class LSM:

    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        self.w = [[0.0], [0.0], [0.0]]

    def fit(self):
        x = self.x_data[:][:]
        for i in range(len(x)):
            x[i] = add_one_for_bias(x[i])

        x_t = transpose(x)
        inv = invert_matrix(matrix_multiply(x_t, x))

        inv_prod_x = matrix_multiply(inv, x_t)
        self.w = matrix_multiply(inv_prod_x, self.y_data)
        print("W:", self.w)

        sum_loss = 0.0
        for idx, data in enumerate(self.x_data):
            data = add_one_for_bias(data)
            result = matrix_multiply([data], self.w)
            sum_loss += (result[0][0] - self.y_data[idx][0]) ** 2
        loss = sum_loss / len(self.x_data)
        print("LOSS:", loss)

    def predict(self, x_data):
        predict = []
        for idx, data in enumerate(x_data):
            data = add_one_for_bias(data)
            result = matrix_multiply([data], self.w)
            predict.append(result[0][0])
        return predict


if __name__ == "__main__":
    x = [[1, 9], [5, 8], [4, 5], [1, 9], [1, 3], [1, 8], [0, 3], [1, 3], ]
    y = [[10], [13], [9], [10], [4], [9], [3], [4]]

    optim = LSM(x, y)

    optim.fit()

    # predict = optim.predict([[2, 5], [7, 3]])
    # print(predict)
