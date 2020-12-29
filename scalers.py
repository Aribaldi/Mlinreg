import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import abstractmethod


class Scaler():
    @abstractmethod
    def fit(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass


class StScaler(Scaler):
    def __init__(self):
        self.means = []
        self.stds = []

    def fit(self, data):
        rows, cols = len(data), len(data[0])
        for j in range(cols):
            s = 0
            ss = 0
            for i in range(rows):
                s += data[i][j]
                ss += data[i][j] ** 2
            self.means.append(s / rows)
            self.stds.append((ss / rows - (s / rows) ** 2) ** (1 / 2))

    def transform(self, data):
        rows, cols = len(data), len(data[0])
        res = copy.deepcopy(data)
        for j in range(cols):
            for i in range(rows):
                res[i][j] = (res[i][j] - self.means[j]) / self.stds[j]
        return res


class MmScaler(Scaler):
    def __init__(self):
        self.maxs = []
        self.mins = []

    def fit(self, data):
        rows, cols = len(data), len(data[0])
        for j in range(cols):
            temp = []
            for i in range(rows):
                temp.append(data[i][j])
            self.maxs.append(max(temp))
            self.mins.append(min(temp))

    def transform(self, data, max_parameter=1, min_parameter=0):
        rows, cols = len(data), len(data[0])
        res = copy.deepcopy(data)
        for j in range(cols):
            for i in range(rows):
                std = (res[i][j] - self.mins[j]) / (self.maxs[j] - self.mins[j])
                res[i][j] = std * (max_parameter - min_parameter) + min_parameter
        return res


if __name__ == '__main__':
    test = [[5 , 3 , 2], [40, 12, 5], [33, 22, 1]]
    skl_s_scaler = StandardScaler()
    skl_s_scaler.fit(test)
    print(skl_s_scaler.transform(test)[0])
    custom_s_scaler = StScaler()
    custom_s_scaler.fit(test)
    print(custom_s_scaler.transform(test)[0])
    skl_m_scaler = MinMaxScaler()
    print(skl_m_scaler.fit_transform(test)[0])
    custom_m_scaler = MmScaler()
    custom_m_scaler.fit(test)
    print(custom_m_scaler.transform(test)[0])