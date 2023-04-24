import pandas as pd
from sklearn.model_selection import GroupShuffleSplit 

data = pd.read_csv('data/cah_2023_small.csv')
data = data.drop(columns=[
    'Unnamed: 0',
    'Unnamed: 0.1',
    'round_completion_seconds',
    'round_skipped',
    ])

splitter = GroupShuffleSplit(test_size=.20, n_splits=2, random_state=7)
test_split = splitter.split(data, groups=data['fake_round_id'])
train_inds, test_inds = next(test_split)
train = data.iloc[train_inds]
test: pd.DataFrame = data.iloc[test_inds]

val_split = splitter.split(train, groups=train['fake_round_id'])
train_inds, val_inds = next(val_split)
val: pd.DataFrame = train.iloc[val_inds]
train: pd.DataFrame = train.iloc[train_inds]
train.to_csv('data/train.csv')
val.to_csv('data/val.csv')
test.to_csv('data/test.csv')