import pandas as pd

def train(params, methods_list, metrics_list, df, scaler, reg=None):
    if reg:
        params["regularization"] = reg

    temp = df.sample(1000)

    print('#' * 32)
    print('1K TRAIN + 200 VAL:')
    print('#' * 32)
    Y_flat = temp['DepDelay'].to_list()
    Y_train = []
    for i in range(len(Y_flat)):
        t = []
        t.append(Y_flat[i])
        Y_train.append(t)

    temp_test = pd.concat([df, temp]).drop_duplicates(keep=False).sample(200)
    Y_test = temp_test['DepDelay'].to_list()

    temp_train = temp.drop(['DepDelay'], axis=1)
    X_train = temp_train.values.tolist()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    temp_test = temp_test.drop(['DepDelay'], axis=1)
    X_test = temp_test.values.tolist()
    X_test = scaler.transform(X_test)

    for method in methods_list:
        obj = method(X_train, Y_train, params)
        obj.fit()
        predicts_test = obj.predict(X_test)
        predicts_train = obj.predict(X_train)

        res_test = [m(predicts_test, Y_test) for m in metrics_list]
        res_train = [m(predicts_train, Y_flat) for m in metrics_list]
        print(f'method: {obj.__class__.__name__}; \n train MSE:{res_train[0]}; \t test R2:{res_train[1]} \n test MSE:{res_test[0]}; \t test R2:{res_test[1]}')

    print('\n')
    print('#'*32)
    print('5-FOLD CV:')
    print('#' * 32)
    n_folds = 5
    n_test = len(temp) // n_folds

    for fold_idx in range(n_folds):
        temp_test = temp.iloc[fold_idx * n_test: (fold_idx + 1) * n_test]
        temp_train = pd.concat([temp, temp_test]).drop_duplicates(keep=False)
        Y_test = temp_test['DepDelay'].to_list()
        temp_test = temp_test.drop(['DepDelay'], axis=1)
        X_test = temp_test.values.tolist()


        Y_flat = temp_train['DepDelay'].to_list()
        Y_train = []
        for i in range(len(Y_flat)):
            t = []
            t.append(Y_flat[i])
            Y_train.append(t)
        temp_train = temp_train.drop(['DepDelay'], axis=1)


        X_train = temp_train.values.tolist()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        for method in methods_list:
            obj = method(X_train, Y_train, params)
            obj.fit()
            predicts_test = obj.predict(X_test)
            predicts_train = obj.predict(X_train)
            res_train = [m(predicts_train, Y_flat) for m in metrics_list]
            res_test = [m(predicts_test, Y_test) for m in metrics_list]

            print(f' method: {obj.__class__.__name__}; fold №{fold_idx} \n train MSE:{res_train[0]}; \t train R2:{res_train[1]} \n '
                  f'test MSE:{res_test[0]}; \t test R2:{res_test[1]} ')
