import pandas as pd

from utils.config import Config
conf = Config('config.json')

from env_factors import gen_dataset
from utils import draw_pic


pred_target = conf.get_config('predict-target')
pred_target_filename = conf.get_data_loc(pred_target)
frame = pd.read_csv(pred_target_filename, parse_dates=True, index_col='date')
cols = gen_dataset.get_valid_cols(frame.columns)

for col in cols:
    print(col)
    x, y = gen_dataset.load_one_col_not_split(pred_target_filename, col)
    draw_pic.draw_by_label('corr_pics', col, temperature=x[:, 0], pressure=x[:, 1], stress=y)
