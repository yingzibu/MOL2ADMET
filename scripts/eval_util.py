from mycolorpy import colorlist as mcp
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def reg_evaluate(label_clean, preds_clean):
    mae = metrics.mean_absolute_error(label_clean, preds_clean)
    mse = metrics.mean_squared_error(label_clean, preds_clean)
    rmse = np.sqrt(mse) #mse**(0.5)
    r2 = metrics.r2_score(label_clean, preds_clean)

    # print("Overall results of sklearn.metrics:")
    # print("MAE:",mae)
    # print("MSE:", mse)
    # print("RMSE:", rmse)
    # print("R-Squared:", r2)
    print('MAE     RMSE     R2')
    print("& %5.3f" % (mae), " & %3.3f" % (rmse), " & %3.3f" % (r2))

    eval_result_r2 =   f'R2:     {r2:.3f}'
    eval_result_mae =  f'MAE:   {mae:.3f}'
    eval_result_rmse = f'RMSE: {rmse:.3f}'

    return eval_result_r2, eval_result_mae, eval_result_rmse
