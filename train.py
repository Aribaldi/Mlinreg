import pandas as pd

def train(method, metrics_list, df, scaler, sampling_method, epochs, batch_size, reg=None):
    params = {"epochs": epochs,
              "learning_rate": 0.03,
              "batch_size": batch_size,
              'epsilon': 1e-8,
              'beta_1': 0.9,
              'beta_2': 0.999,
              "weight_decay": 1e-6,
              "gamma": 0.975}
    if reg:
        params["regularization"] = reg

    temp = df.sample(1000)
    if sampling_method == 'default':

        Y_flat = temp['DepDelay'].to_list()
        Y_train = []
        for i in range(len(Y_flat)):
            t = []
            t.append(Y_flat[i])
            Y_train.append(t)

        temp_test = pd.concat([df, temp]).drop_duplicates(keep=False).sample(200)
        Y_test = temp_test['DepDelay'].to_list()

        temp = temp.drop(['DepDelay'], axis=1)
        X_train = temp.values.tolist()
        X_train = scaler(X_train)
        temp_test = temp_test.drop(['DepDelay'], axis=1)
        X_test = temp_test.values.tolist()
        X_test = scaler(X_test)

        obj = method(X_train, Y_train, params)
        obj.fit()
        predicts = obj.predict(X_test)

        res = [m(predicts, Y_test) for m in metrics_list]
        print(f'MSE:{res[0]} \t R2: {res[1]}')

    if sampling_method == 'k-fold':
        n_folds = 5
        n_test = len(temp) // n_folds
        res_t = []
        
        for fold_idx in range(n_folds):
            temp_test = temp.iloc[fold_idx * n_test : (fold_idx + 1) * n_test]
            Y_test = temp_test['DepDelay'].to_list()
            temp_test = temp_test.drop(['DepDelay'], axis=1)
            X_test = temp_test.values.tolist()
            X_test = scaler(X_test)
            
            temp_train = pd.concat([temp, temp_test]).drop_duplicates(keep=False)
            Y_flat = temp_train['DepDelay'].to_list()
            Y_train = []
            for i in range(len(Y_flat)):
                t = []
                t.append(Y_flat[i])
                Y_train.append(t)
            temp_train = temp_train.drop(['DepDelay'], axis=1)
            X_train = temp_train.values.tolist()
            X_train = scaler(X_train)
            
            obj = method(X_train, Y_train, params)
            obj.fit()
            predicts = obj.predict(X_test)

            res_t.append([m(predicts, Y_test) for m in metrics_list])
        
        res = [sum(a[i] for a in res_t) / len(res_t) for i in range(len(res_t[0]))]
        print(f'MSE:{res[0]} \t R2: {res[1]}')
