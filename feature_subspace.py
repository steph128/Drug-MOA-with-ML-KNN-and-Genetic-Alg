from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import multiprocessing as mp

X = pd.read_csv('train_features.csv', index_col=0)
y = pd.read_csv('train_targets_scored.csv', index_col=0)

X = X.replace({
    'cp_type': {'trt_cp': -1.0, 'ctl_vehicle': 1.0},
    'cp_time': {24: -1.0, 48: 0.0, 72: 1.0},
    'cp_dose': {'D1': -1.0, 'D2': 1.0}
})

X_scaled = StandardScaler().fit_transform(X.iloc(axis=1)[3:].values)
y_indices = list(range(len(y)))

X_scaled = np.hstack([X.iloc(axis=1)[:3].values, X_scaled])


def select_features(y_col_idx, threshold=0.95):
    y_col = y.iloc(axis=1)[y_col_idx].values
    column_name = y.iloc(axis=1)[y_col_idx].name
    mi = mutual_info_classif(X_scaled, y_col)  # calculates the mutual information of all x for a single y_col
    v = StandardScaler().fit_transform(mi.reshape((-1, 1))).reshape((-1, )) > threshold  # returns which x_cols to use
    with open('feature_subspaces/{}.npy'.format(column_name), 'wb') as f:
        np.save(f, v)


if __name__ == '__main__':
    pool = mp.Pool(mp.cpu_count())
    pool.map(select_features, y_indices)
