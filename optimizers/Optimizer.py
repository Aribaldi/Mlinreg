from abc import abstractmethod, ABC, ABCMeta
from matrix import *


class Optimizer(ABC):

    def __init__(self, x_data, y_data, params):
        self.x_data = x_data
        self.y_data = y_data
        self.params = params
        # self.w = zeros_matrix(len(self.x_data[0]) + 1, len(self.y_data[0]))
        #self.w = [[0.0], [0.888], [0.777]]
        self.w = init_norm(len(self.x_data[0]) + 1, 1)
        self.statistics = {}
        self.params = params
        self.steps = 0

    @abstractmethod
    def __calc_grad(self, losses, batch):
        pass

    def fit(self):
        for i in range(self.params["epochs"]):
            for j in range(len(self.x_data) // self.params["batch_size"]):
                batch = self.x_data[self.params["batch_size"] * j: self.params["batch_size"] * (j + 1)]
                batch_ans = []
                losses = []
                curr_loss = 0.0
                for num, example in enumerate(batch):
                    example = add_one_for_bias(example)
                    print(example)
                    result = matrix_multiply([example], self.w)
                    batch_ans.append(result)

                    print(result)
                    print(self.y_data[j * self.params["batch_size"] + num][0])
                    loss_ex = result[0][0] - self.y_data[j * self.params["batch_size"] + num][0]
                    losses.append(loss_ex)

                    mse_loss = loss_ex ** 2
                    curr_loss += mse_loss

                grad = self.__calc_grad(losses, batch)
                self.steps += 1
                print("GRAD: ", grad)
                print("W BEFORE: ", self.w)
                print(grad)

                if "regularization" in self.params:
                    alpha = self.params["weight_decay"]
                    if self.params["regularization"] == "l1":
                        self.w = matrix_add_scalar(self.w, -alpha)
                    elif self.params["regularization"] == "l2":
                        self.w = matrix_sum(self.w, matrix_by_scalar(self.w, -2*alpha))
                    else:
                        raise ValueError("No such type of regularization")

                self.w = matrix_sum(self.w, matrix_by_scalar(grad, -self.params["learning_rate"]))
                print("W: ", self.w)

                print("Curr epoch: {}, Num of example: {}, Curr loss: {}".format(i + 1, j + 1, curr_loss / self.params["batch_size"]))

    def predict(self, x_data):
        predict = []
        for idx, data in enumerate(x_data):
            data = add_one_for_bias(data)
            result = matrix_multiply([data], self.w)
            predict.append(result)

        return predict