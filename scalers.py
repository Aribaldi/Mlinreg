import copy
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def standardscaler(data):
    rows, cols = len(data), len(data[0])
    means = []
    stds = []
    for j in range(cols):
        s = 0
        ss = 0
        for i in range(rows):
            s += data[i][j]
            ss += data[i][j]**2
        means.append(s / rows)
        stds.append((ss / rows - (s/rows)**2)**(1/2))
    res = copy.deepcopy(data)
    for j in range(cols):
        for i in range(rows):
            res[i][j] = (res[i][j] - means[j]) / stds[j]
    return res


def minmaxscaler(data, max_parameter=1, min_parameter=0):
    rows, cols = len(data), len(data[0])
    maxs = []
    mins = []
    for j in range(cols):
        temp = []
        for i in range(rows):
            temp.append(data[i][j])
        maxs.append(max(temp))
        mins.append(min(temp))
    res = copy.deepcopy(data)
    for j in range(cols):
        for i in range(rows):
            std = (res[i][j] - mins[j]) / (maxs[j]-mins[j])
            res[i][j] = std * (max_parameter - min_parameter) + min_parameter
    return res

if __name__ == '__main__':
    test = [[5 , 3 , 2], [40, 12, 5], [33, 22, 1]]
    scaler = StandardScaler()
    scaler.fit(test)
    print(scaler.transform(test)[0])
    print(standardscaler(test)[0])
    mmscaler = MinMaxScaler()
    print(mmscaler.fit_transform(test)[0])
    print(minmaxscaler(test)[0])