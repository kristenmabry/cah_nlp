import pandas as pd
from transformers import AutoTokenizer
import datasets
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer
import evaluate
import numpy as np
from cah_funcs import *

bert_model = 'bert-base-uncased'
model_dir = '../cah_model'
print('Bert:', bert_model)
print('CAH Model:', model_dir)

train = pd.read_csv('../data/train.csv')
val = pd.read_csv('../data/val.csv')

tokenizer = AutoTokenizer.from_pretrained(bert_model)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

extra = (tokenizer,)
tokenized_train = train.groupby('fake_round_id').apply(preprocess_function, extra).reset_index()
print('finish tokenizing train')
tokenized_val = val.groupby('fake_round_id').apply(preprocess_function, extra).reset_index()
print('finish tokenizing val')
train_set = datasets.Dataset.from_list(tokenized_train[0].to_list())
val_set = datasets.Dataset.from_list(tokenized_val[0].to_list())

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        labels = [feature.pop('label') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )

        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch


model = AutoModelForMultipleChoice.from_pretrained(bert_model)

accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

weight_decay = 0.01
epochs = 3
batch_size = 16
learning_rate = 5e-5

training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=weight_decay,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_set,
    eval_dataset=val_set,
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
print('finish train')
trainer.save_model(model_dir)
print('finish saving model')

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForMultipleChoice.from_pretrained(model_dir)

print('Bert:', bert_model)
print('CAH Model:', model_dir)
print(weight_decay, epochs, batch_size, learning_rate)
evaluateDataset(model, tokenizer, 'validation', val)