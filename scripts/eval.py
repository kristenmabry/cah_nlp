import pandas as pd
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice
from cah_funcs import *

train = pd.read_csv('../data/train.csv')
val = pd.read_csv('../data/val.csv')
test = pd.read_csv('../data/train.csv')
double = pd.read_csv('../data/double_card_rounds.csv')
rare = pd.read_csv('../data/cah_rare_cards.csv')

tokenizer = AutoTokenizer.from_pretrained('../cah_model')
model = AutoModelForMultipleChoice.from_pretrained('../cah_model')

card_wins = train.groupby(by='white_card_text') \
                .agg(picks=('won','sum'),
                    pick_opportunities=('won', 'count'),
                    pick_ratio=('won', 'mean')) \
                .sort_values('pick_ratio', ascending=False)

evaluateDataset(model, tokenizer, 'validation', val, card_wins)
evaluateDataset(model, tokenizer, 'test', test, card_wins)
evaluateDataset(model, tokenizer, 'double', double, None, True)
evaluateDataset(model, tokenizer, 'rare', rare)