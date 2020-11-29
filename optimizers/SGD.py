from optimizers.Optimizer import Optimizer
from matrix import *


class SGD(Optimizer):

    def __init__(self, x_data, y_data, params):
        super(SGD, self).__init__(x_data, y_data, params)

    def _Optimizer__calc_grad(self, losses, batch):
        grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))

        for num, example in enumerate(batch):
            loss = losses[num]
            print("LOSS: ", loss)

            tmp_grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))
            tmp_grad[0][0] = 2 * loss

            for z in range(1, len(self.x_data[0]) + 1):
                tmp_grad[z][0] = 2 * loss * example[z - 1]

            grad = matrix_sum(grad, tmp_grad)

        grad = matrix_by_scalar(grad, 1 / self.params["batch_size"])

        return grad


if __name__ == "__main__":
    x_data = [[1, 2], [2, 3], [4, 5], [1, 9], [1, 3], [1, 8], [0, 3], [1, 3]]
    y_data = [[3], [5], [9], [10], [4], [9], [3], [4]]

    optim = SGD(x_data=x_data, y_data=y_data,
                    params={"epochs": 1000,
                            "learning_rate": 0.01,
                            "batch_size": 4,
                            'epsilon': 1e-8})
    optim.fit()
