import pandas as pd
import numpy as np
import torch


def filter_items(df, user_col, item_col, item_min_count=20, user_min_count=10):

    print('Filtering items..')

    item_count = df.groupby(item_col)[user_col].nunique()
    item_ids = item_count[item_count >= item_min_count].index
    print(f'Number of items before: {len(item_count)}')
    print(f'Number of items after: {len(item_ids)}')

    print(f'Interactions length before: {len(df)}')
    df = df[df[item_col].isin(item_ids)]
    df.reset_index(inplace=True, drop=True)
    print(f'Interactions length after: {len(df)}')

    # check that user_count hasn't fallen below the user_min_count threshold after filtering items
    user_count = df.groupby(user_col)[item_col].nunique()
    print(f'Does each user still have enough ratings?')
    print('Yes') if len(user_count[user_count < user_min_count]) == 0 else print(f'No, users with fewer than {user_min_count} ratings detected')

    return df.reset_index(drop=True)


def tr_tst_split(df, test_quantile=0.8):
    """
    Split clickstream by date.
    """
    df = df.sort_values(['user', 'timestamp'])
    test_timepoint = df['timestamp'].quantile(q=test_quantile, interpolation='nearest')
    test = df.query('timestamp >= @test_timepoint')
    train = df.drop(test.index)

    # make sure there are no cold items or users in the test set
    test = test[test['user'].isin(train['user'])]
    test = test[test['item'].isin(train['item'])]

    test.reset_index(drop=True, inplace=True)
    train.reset_index(drop=True, inplace=True)
    return train, test


def negative_sampling(df, global_df, num_items, num_negative=1):
    """
    For each positive pair (u, i), sample num_negative items the user never interacted with
    Returns a df with columns: user, item, label
    """
    users, items, labels = [], [], []
    user_item_set = set(zip(global_df['user'], global_df['item']))

    for u, i, l in df.itertuples(index=False):

        # positive item
        users.append(u)
        items.append(i)
        labels.append(l)

        # negative items
        for _ in range(num_negative):
            j = np.random.randint(num_items)
            while (u, j) in user_item_set:
                j = np.random.randint(num_items)
            users.append(u)
            items.append(j)
            labels.append(0)
    return pd.DataFrame({'user': users, 'item': items, 'label': labels})


def count_parameters(model):
    """
    Show total and trainable parameter counts of the given model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")


def recall(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=20) -> float:
    recall_values = []
    for _, row in df.iterrows():
      num_relevant = len(set(row[true_col]) & set(row[pred_col][:k]))
      num_true = len(row[true_col])
      recall_values.append(num_relevant / num_true)
    return round(np.mean(recall_values), 4)


def precision(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=20) -> float:
    precision_values = []
    for _, row in df.iterrows():
      num_relevant = len(set(row[true_col]) & set(row[pred_col][:k]))
      num_true = k
      precision_values.append(num_relevant / num_true)
    return round(np.mean(precision_values), 4)


def mrr(df: pd.DataFrame, pred_col='preds', true_col='item_id', k=20) -> float:
    mrr_values = []
    for _, row in df.iterrows():
      intersection = set(row[true_col]) & set(row[pred_col][:k])
      user_mrr = 0
      if len(intersection) > 0:
          for item in intersection:
              user_mrr = max(user_mrr, 1 / (row[pred_col].index(item) + 1))
      mrr_values.append(user_mrr)
    return round(np.mean(mrr_values), 4)