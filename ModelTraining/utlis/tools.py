import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_labelled_data(PATH, isLeft=False):
    df = pd.read_csv(PATH)
    remove_flags = np.zeros(df.shape[0])
    ## Check Labels
    for i, row in df.iterrows():
        label = row.iloc[0] 
        if isLeft:
            if (label not in [0.0,1.0,2.0,3.0,8.0]):
                remove_flags[i] = 1
        # if right hand label
        elif (label not in [4.0,5.0,6.0,7.0,9.0]):
            remove_flags[i] = 1

    remove_indice =  np.where(remove_flags==1)[0]
    print("Found {} rows to remove for selecting one hand.".format(len(remove_indice)))
    temp_df = df.drop(remove_indice, axis=0)

    if isLeft:
        one_hand_df = temp_df[['Label', 'Time','Counter',
                # Left Hand Information
                'IndexDistalJoint.1_0','IndexDistalJoint.1_1','IndexDistalJoint.1_2',
                'IndexKnuckle.1_0','IndexKnuckle.1_1','IndexKnuckle.1_2',
                'IndexMetacarpal.1_0','IndexMetacarpal.1_1','IndexMetacarpal.1_2',
                'IndexMiddleJoint.1_0','IndexMiddleJoint.1_1','IndexMiddleJoint.1_2',
                'IndexTip.1_0','IndexTip.1_1','IndexTip.1_2',
                'MiddleDistalJoint.1_0','MiddleDistalJoint.1_1','MiddleDistalJoint.1_2',
                'MiddleKnuckle.1_0','MiddleKnuckle.1_1','MiddleKnuckle.1_2',
                'MiddleMetacarpal.1_0','MiddleMetacarpal.1_1','MiddleMetacarpal.1_2',
                'MiddleMiddleJoint.1_0','MiddleMiddleJoint.1_1','MiddleMiddleJoint.1_2',
                'MiddleTip.1_0','MiddleTip.1_1','MiddleTip.1_2',
                'Palm.1_0','Palm.1_1','Palm.1_2',
                'PinkyDistalJoint.1_0','PinkyDistalJoint.1_1','PinkyDistalJoint.1_2',
                'PinkyKnuckle.1_0','PinkyKnuckle.1_1','PinkyKnuckle.1_2',
                'PinkyMetacarpal.1_0','PinkyMetacarpal.1_1','PinkyMetacarpal.1_2',
                'PinkyMiddleJoint.1_0','PinkyMiddleJoint.1_1','PinkyMiddleJoint.1_2',
                'PinkyTip.1_0','PinkyTip.1_1','PinkyTip.1_2',
                'RingDistalJoint.1_0','RingDistalJoint.1_1','RingDistalJoint.1_2',
                'RingKnuckle.1_0','RingKnuckle.1_1','RingKnuckle.1_2',
                'RingMetacarpal.1_0','RingMetacarpal.1_1','RingMetacarpal.1_2',
                'RingMiddleJoint.1_0','RingMiddleJoint.1_1','RingMiddleJoint.1_2',
                'RingTip.1_0','RingTip.1_1','RingTip.1_2',
                'ThumbDistalJoint.1_0','ThumbDistalJoint.1_1','ThumbDistalJoint.1_2',
                'ThumbMetacarpalJoint.1_0','ThumbMetacarpalJoint.1_1','ThumbMetacarpalJoint.1_2',
                'ThumbProximalJoint.1_0','ThumbProximalJoint.1_1','ThumbProximalJoint.1_2',
                'ThumbTip.1_0','ThumbTip.1_1','ThumbTip.1_2',
                'Wrist.1_0','Wrist.1_1','Wrist.1_2']]
    else:
        one_hand_df = temp_df[['Label', 'Time','Counter',
                # Right Hand Information
                'IndexDistalJoint_0','IndexDistalJoint_1','IndexDistalJoint_2',
                'IndexKnuckle_0','IndexKnuckle_1','IndexKnuckle_2',
                'IndexMetacarpal_0','IndexMetacarpal_1','IndexMetacarpal_2',
                'IndexMiddleJoint_0','IndexMiddleJoint_1','IndexMiddleJoint_2',
                'IndexTip_0','IndexTip_1','IndexTip_2',
                'MiddleDistalJoint_0','MiddleDistalJoint_1','MiddleDistalJoint_2',
                'MiddleKnuckle_0','MiddleKnuckle_1','MiddleKnuckle_2',
                'MiddleMetacarpal_0','MiddleMetacarpal_1','MiddleMetacarpal_2',
                'MiddleMiddleJoint_0','MiddleMiddleJoint_1','MiddleMiddleJoint_2',
                'MiddleTip_0','MiddleTip_1','MiddleTip_2',
                'Palm_0','Palm_1','Palm_2',
                'PinkyDistalJoint_0','PinkyDistalJoint_1','PinkyDistalJoint_2',
                'PinkyKnuckle_0','PinkyKnuckle_1','PinkyKnuckle_2',
                'PinkyMetacarpal_0','PinkyMetacarpal_1','PinkyMetacarpal_2',
                'PinkyMiddleJoint_0','PinkyMiddleJoint_1','PinkyMiddleJoint_2',
                'PinkyTip_0','PinkyTip_1','PinkyTip_2',
                'RingDistalJoint_0','RingDistalJoint_1','RingDistalJoint_2',
                'RingKnuckle_0','RingKnuckle_1','RingKnuckle_2',
                'RingMetacarpal_0','RingMetacarpal_1','RingMetacarpal_2',
                'RingMiddleJoint_0','RingMiddleJoint_1','RingMiddleJoint_2',
                'RingTip_0','RingTip_1','RingTip_2',
                'ThumbDistalJoint_0','ThumbDistalJoint_1','ThumbDistalJoint_2',
                'ThumbMetacarpalJoint_0','ThumbMetacarpalJoint_1','ThumbMetacarpalJoint_2',
                'ThumbProximalJoint_0','ThumbProximalJoint_1','ThumbProximalJoint_2',
                'ThumbTip_0','ThumbTip_1','ThumbTip_2',
                'Wrist_0','Wrist_1','Wrist_2']]

    if isLeft:
        one_hand_df.replace({'Label': {8.0: 4.0}}, inplace=True)
    else:
        one_hand_df.replace({'Label': {4.0: 0.0,
                                        5.0: 1.0,
                                        6.0: 2.0,
                                        7.0: 3.0,
                                        9.0: 4.0,}}, 
                                        inplace=True)


    y = one_hand_df[['Label']]

    X = one_hand_df.drop(['Label', 'Time', 'Counter'], axis=1)
    return X, y


def load_labelled_data_combine(PATH, combineLeftRight=True):
    df_left = pd.read_csv(PATH)
    df_right = pd.read_csv(PATH)

    remove_flags_left = np.zeros(df_left.shape[0])
    remove_flags_right = np.zeros(df_right.shape[0])

    ## Check Labels
    for i, row in df_left.iterrows():
        label = row.iloc[0] 
        if (label not in [0.0,1.0,2.0,3.0,8.0]):
            remove_flags_left[i] = 1

    for i, row in df_right.iterrows():
        label = row.iloc[0] 
        if (label not in [4.0,5.0,6.0,7.0,9.0]):
            remove_flags_right[i] = 1

    remove_indice_left =  np.where(remove_flags_left==1)[0]
    temp_df_left = df_left.drop(remove_indice_left, axis=0)

    remove_indice_right =  np.where(remove_flags_right==1)[0]
    temp_df_right = df_right.drop(remove_indice_right, axis=0)

    left_hand_df = temp_df_left[['Label', 'Time','Counter',
            # Left Hand Information
            'IndexDistalJoint.1_0','IndexDistalJoint.1_1','IndexDistalJoint.1_2',
            'IndexKnuckle.1_0','IndexKnuckle.1_1','IndexKnuckle.1_2',
            'IndexMetacarpal.1_0','IndexMetacarpal.1_1','IndexMetacarpal.1_2',
            'IndexMiddleJoint.1_0','IndexMiddleJoint.1_1','IndexMiddleJoint.1_2',
            'IndexTip.1_0','IndexTip.1_1','IndexTip.1_2',
            'MiddleDistalJoint.1_0','MiddleDistalJoint.1_1','MiddleDistalJoint.1_2',
            'MiddleKnuckle.1_0','MiddleKnuckle.1_1','MiddleKnuckle.1_2',
            'MiddleMetacarpal.1_0','MiddleMetacarpal.1_1','MiddleMetacarpal.1_2',
            'MiddleMiddleJoint.1_0','MiddleMiddleJoint.1_1','MiddleMiddleJoint.1_2',
            'MiddleTip.1_0','MiddleTip.1_1','MiddleTip.1_2',
            'Palm.1_0','Palm.1_1','Palm.1_2',
            'PinkyDistalJoint.1_0','PinkyDistalJoint.1_1','PinkyDistalJoint.1_2',
            'PinkyKnuckle.1_0','PinkyKnuckle.1_1','PinkyKnuckle.1_2',
            'PinkyMetacarpal.1_0','PinkyMetacarpal.1_1','PinkyMetacarpal.1_2',
            'PinkyMiddleJoint.1_0','PinkyMiddleJoint.1_1','PinkyMiddleJoint.1_2',
            'PinkyTip.1_0','PinkyTip.1_1','PinkyTip.1_2',
            'RingDistalJoint.1_0','RingDistalJoint.1_1','RingDistalJoint.1_2',
            'RingKnuckle.1_0','RingKnuckle.1_1','RingKnuckle.1_2',
            'RingMetacarpal.1_0','RingMetacarpal.1_1','RingMetacarpal.1_2',
            'RingMiddleJoint.1_0','RingMiddleJoint.1_1','RingMiddleJoint.1_2',
            'RingTip.1_0','RingTip.1_1','RingTip.1_2',
            'ThumbDistalJoint.1_0','ThumbDistalJoint.1_1','ThumbDistalJoint.1_2',
            'ThumbMetacarpalJoint.1_0','ThumbMetacarpalJoint.1_1','ThumbMetacarpalJoint.1_2',
            'ThumbProximalJoint.1_0','ThumbProximalJoint.1_1','ThumbProximalJoint.1_2',
            'ThumbTip.1_0','ThumbTip.1_1','ThumbTip.1_2',
            'Wrist.1_0','Wrist.1_1','Wrist.1_2']]
    left_hand_df.replace({'Label': {8.0: 4.0}}, inplace=True)

    right_hand_df = temp_df_right[['Label', 'Time','Counter',
            # Right Hand Information
            'IndexDistalJoint_0','IndexDistalJoint_1','IndexDistalJoint_2',
            'IndexKnuckle_0','IndexKnuckle_1','IndexKnuckle_2',
            'IndexMetacarpal_0','IndexMetacarpal_1','IndexMetacarpal_2',
            'IndexMiddleJoint_0','IndexMiddleJoint_1','IndexMiddleJoint_2',
            'IndexTip_0','IndexTip_1','IndexTip_2',
            'MiddleDistalJoint_0','MiddleDistalJoint_1','MiddleDistalJoint_2',
            'MiddleKnuckle_0','MiddleKnuckle_1','MiddleKnuckle_2',
            'MiddleMetacarpal_0','MiddleMetacarpal_1','MiddleMetacarpal_2',
            'MiddleMiddleJoint_0','MiddleMiddleJoint_1','MiddleMiddleJoint_2',
            'MiddleTip_0','MiddleTip_1','MiddleTip_2',
            'Palm_0','Palm_1','Palm_2',
            'PinkyDistalJoint_0','PinkyDistalJoint_1','PinkyDistalJoint_2',
            'PinkyKnuckle_0','PinkyKnuckle_1','PinkyKnuckle_2',
            'PinkyMetacarpal_0','PinkyMetacarpal_1','PinkyMetacarpal_2',
            'PinkyMiddleJoint_0','PinkyMiddleJoint_1','PinkyMiddleJoint_2',
            'PinkyTip_0','PinkyTip_1','PinkyTip_2',
            'RingDistalJoint_0','RingDistalJoint_1','RingDistalJoint_2',
            'RingKnuckle_0','RingKnuckle_1','RingKnuckle_2',
            'RingMetacarpal_0','RingMetacarpal_1','RingMetacarpal_2',
            'RingMiddleJoint_0','RingMiddleJoint_1','RingMiddleJoint_2',
            'RingTip_0','RingTip_1','RingTip_2',
            'ThumbDistalJoint_0','ThumbDistalJoint_1','ThumbDistalJoint_2',
            'ThumbMetacarpalJoint_0','ThumbMetacarpalJoint_1','ThumbMetacarpalJoint_2',
            'ThumbProximalJoint_0','ThumbProximalJoint_1','ThumbProximalJoint_2',
            'ThumbTip_0','ThumbTip_1','ThumbTip_2',
            'Wrist_0','Wrist_1','Wrist_2']]
    right_hand_df.replace({'Label': {4.0: 0.0,
                                    5.0: 1.0,
                                    6.0: 2.0,
                                    7.0: 3.0,
                                    9.0: 4.0,}}, 
                                    inplace=True)
    
    new_cols = {x: y for x, y in zip(left_hand_df.columns, right_hand_df.columns)}
    combine_df = pd.concat([right_hand_df, 
                            left_hand_df.rename(columns=new_cols)], 
                            axis=0, 
                            ignore_index=True)

    y = combine_df[['Label']]
    X = combine_df.drop(['Label', 'Time', 'Counter'], axis=1)
    return X, y



def generate_time_lags(X_df, y_df, lagWindowSize):
    features = X_df.copy()
    for n in range(1, lagWindowSize):
        for idx, featureName in enumerate(list(X_df.columns.values)):        
            features[f"{featureName}_lag{n}"] = features[featureName].shift(n)
    # drop the first #lagWindowSize rows
    lag_features = features.iloc[lagWindowSize:]
    lag_labels = y_df.iloc[lagWindowSize:]
    lag_features.reset_index(inplace = True, drop = True)
    lag_labels.reset_index(inplace = True, drop = True)
    return lag_features, lag_labels

def convert_df_2_np_3D(X_lag, lagWindowSize):
    feature2D = X_lag.to_numpy()
    featureLength = int(X_lag.shape[1]/lagWindowSize)
    feature3D = None
    for i in range(feature2D.shape[0]):
        featureMap = feature2D[i].reshape([lagWindowSize, featureLength])
        featureMap = np.expand_dims(featureMap, axis=0)
        if feature3D is None:
            feature3D = featureMap
        else:
            feature3D = np.concatenate((feature3D, featureMap),
                                    axis=0)
    return feature3D
