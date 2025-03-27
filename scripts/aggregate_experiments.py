import os

import pandas as pd
from tqdm import tqdm

dir_name = 'experiments/wine_test_imputers_scores'
results = []
for file_name in tqdm(os.listdir(dir_name)):
    try:
        if file_name.endswith('.csv'):
                results.append(pd.read_csv(os.path.join(dir_name, file_name)))
    except Exception as e:
        continue
results = pd.concat(results)
path = os.path.join(f'{dir_name}.csv')
results.to_csv(path, index=False)
results
