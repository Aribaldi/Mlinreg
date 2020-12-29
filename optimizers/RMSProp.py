from optimizers.Optimizer import Optimizer
from matrix import *


class RMSProp(Optimizer):

    def __init__(self, x_data, y_data, params):
        super(RMSProp, self).__init__(x_data, y_data, params)

        self.statistics["ema_grad"] = create_zeros_same_shape(self.w)

    def _Optimizer__calc_grad(self, losses, batch):
        grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))

        for num, example in enumerate(batch):
            loss = losses[num]
            #print("LOSS: ", loss)

            tmp_grad = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))
            tmp_grad[0][0] = 2 * loss

            for z in range(1, len(self.x_data[0]) + 1):
                tmp_grad[z][0] = 2 * loss * example[z - 1]

            grad = matrix_sum(grad, tmp_grad)

        grad = matrix_by_scalar(grad, 1 / self.params["batch_size"])
        #print("GRAD inside: ", grad)
        self.statistics["ema_grad"][0][0] = self.statistics["ema_grad"][0][0] * self.params["gamma"] + \
                                            (1 - self.params["gamma"]) * (grad[0][0] ** 2)

        for z in range(1, len(self.x_data[0]) + 1):
            self.statistics["ema_grad"][z][0] += self.statistics["ema_grad"][z][0] * self.params["gamma"] + \
                                                 (1 - self.params["gamma"]) * (grad[z][0] ** 2)

        #print("STATISTIC: ", self.statistics["ema_grad"])
        coefficient = elementwise_power(matrix_add_scalar(self.statistics["ema_grad"], self.params["epsilon"]), -1/2)
        grad = elementwise_product(grad, coefficient)

        return grad

if __name__ == "__main__":
    x_data = [[1, 2], [2, 3], [4, 5], [1, 9], [1, 3], [1, 8], [0, 3], [1, 3]]
    y_data = [[3], [5], [9], [10], [4], [9], [3], [4]]

    # mean_1 = 11/8
    # std_1 = 0.0
    # mean_2 = 36/8
    # std_2 = 0.0
    # for i in range(len(x_data)):
    # 	std_1 += abs(x_data[i][0] - mean_1) ** 2
    # 	std_2 += abs(x_data[i][1] - mean_2) ** 2
    #
    # std_1 = (std_1/len(x_data)) ** (1/2)
    # std_2 = (std_2 / len(x_data)) ** (1/2)
    #
    # for i in range(len(x_data)):
    # 	x_data[i][0] = (x_data[i][0] - mean_1)/std_1
    # 	x_data[i][1] = (x_data[i][1] - mean_2) / std_2

    optim = RMSProp(x_data=x_data, y_data=y_data,
                    params={"epochs": 100,
                            "learning_rate": 0.01,
                            "batch_size": 4,
                            'epsilon': 1e-8,
                            'gamma': 0.975})
    optim.fit()
    print(optim.w)