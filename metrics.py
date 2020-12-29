def MSE_metric(pred, gt):
    pred = [el[0][0] for el in pred]
    l = len(gt)
    return round((1/l) * sum(list(map(lambda x, y: (x-y)**2, pred, gt))), 2)

def R2_metric(pred, gt):
    pred = [el[0][0] for el in pred]
    SS_res = sum(list(map(lambda x, y: (x-y)**2, pred, gt)))
    mean_y = sum(gt) / len(gt)
    SS_tot = sum(list(map(lambda x: (x-mean_y)**2, gt)))
    return round(1 - SS_res/SS_tot, 2)