import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def weighted_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Custom Loss function that combines MAPE and Reward.
    alpha and beta params can be used to tune.

    params:
        - y_true (numpy array) : numpy array of actual revenues 
        - y_pred (numpy array) : numpy array of predicted revenues
        - alpha (float): tuning parameter for MAPE 
        - beta (float) : tuning parameter for Rewards
    returns:
        - loss (float) : final combined loss
    """

    def mean_absolute_percentage_error(true, pred):
        """ Computes MAPE """
        return np.mean(np.abs((true - pred) / true)) * 100

    def threshold_score(mape):
        """ Computes reward for one mape value """
        if mape <= 5:
            return 2
        elif mape > 5 and mape <=10:
            return 1
        elif mape > 10 and mape <=15:
            return -1
        else:
            return -2
    
    # compute mape for all predictions
    mape_loss = mean_absolute_percentage_error(y_true, y_pred)

    # calculate mape and reward for each practice
    mape_list = []
    reward_list = []

    for true, pred in zip(np.array(y_test), preds):
        mape_list.append(mean_absolute_percentage_error(true, pred))
        reward_list.append(threshold_score(mean_absolute_percentage_error(true, pred)))
    
    scaler = MinMaxScaler()

    #  normalize reward from -2,2 to 0,1
    normalized_reward_list  = scaler.fit_transform(reward_list)
    # compute reward score
    reward_score = np.mean(normalized_reward_list) * 100

    return (alpha *  mape_loss) + (beta * reward_score)
    
