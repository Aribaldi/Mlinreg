from optimizers.Optimizer import Optimizer
from matrix import *


class AdaGrad(Optimizer):

    def __init__(self, x_data, y_data, params):
        super(AdaGrad, self).__init__(x_data, y_data, params)

        self.statistics["sum_squared"] = create_zeros_same_shape(self.w)

    def _Optimizer__calc_grad(self, losses, batch):
        grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))

        for num, example in enumerate(batch):
            loss = losses[num]
            print("LOSS: ", loss)
            tmp_grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))
            tmp_grad[0][0] = 2 * loss


            for z in range(1, len(self.x_data[0]) + 1):
                tmp_grad[z][0] = 2 * loss * example[z - 1]
                self.statistics["sum_squared"][z][0] += (2 * loss * example[z - 1]) ** 2

            grad = matrix_sum(grad, tmp_grad)

        grad = matrix_by_scalar(grad, 1 / self.params["batch_size"])

        self.statistics["sum_squared"][0][0] += (grad[0][0] ** 2)
        for z in range(1, len(self.x_data[0]) + 1):
            self.statistics["sum_squared"][z][0] += grad[z][0] ** 2

        coefficient = elementwise_power(matrix_add_scalar(self.statistics["sum_squared"], self.params["epsilon"]), -1/2)
        grad = elementwise_product(grad, coefficient)

        return grad

if __name__ == "__main__":
    x_data = [[1, 2], [2, 3], [4, 5], [1, 9], [1, 3], [1, 8], [0, 3], [1, 3]]
    y_data = [[3], [5], [9], [10], [4], [9], [3], [4]]

    optim = AdaGrad(x_data=x_data, y_data=y_data,
                    params={"epochs": 1000,
                            "learning_rate": 0.1,
                            "batch_size": 4,
                            'epsilon': 1e-8})
    optim.fit()
    print(optim.w)