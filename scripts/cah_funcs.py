import random
import pandas as pd
from tqdm import tqdm
import torch


def getDoubleCards(round):
    answers = []
    cards = round['white_card_text'].to_list()
    winners_index = round[round['won']].sort_values('winning_index', ascending=False)['white_card_text'].index % 10
    sub = 1 if winners_index[1] > winners_index[0] else 0
    winner = cards.pop(winners_index[0]) + ' <sep> ' + cards.pop(winners_index[1] - sub)
    random.shuffle(cards)
    for i in range((int)(len(cards)/2)):
        answers.append(cards[2*i] + ' <sep> ' + cards[2*i+1])
    label = random.randint(0, len(answers)-1)
    answers.insert(label, winner)
    return answers, label

def preprocess_function(round: pd.DataFrame, args):
    tokenizer = args[0]
    questions = []
    answers = []
    label = 0
    if round.iloc[0]['black_card_pick_num'] == 2:
        questions = [round.iloc[0]['black_card_text']] * 5
        answers, label = getDoubleCards(round) 
    else:
        questions = round['black_card_text'].to_list()
        answers = round['white_card_text'].to_list()
        label = round[round['won']].index[0] % 10

    tokenized_examples = tokenizer(questions, answers, truncation=True, padding=True)
    tokenized_examples['label'] = label
    tokenized_examples['round'] = round['fake_round_id']
    tokenized_examples['black_card'] = questions[0]
    tokenized_examples['white_cards'] = answers
    return tokenized_examples

def evaluateDataset(model, tokenizer, name: str, data, card_wins=None, isDouble=False):
    rounds = (int)(len(data) / 10)
    correct = 0
    popular = 0
    for round_num in tqdm(range(rounds)):
        i = round_num * 10
        prompt = data[i:i+1]['black_card_text'].iloc[0]
        candidates = data[i:i+10]['white_card_text'].to_list()
        index = data[i:i+10][data[i:i+10]['won']].index
        if data.iloc[i]['black_card_pick_num'] == 2:
            candidates, index = getDoubleCards(data[i:i+10])
            index = [index]
        label = torch.tensor(index[0] % 10).unsqueeze(0)
        inputs = tokenizer([[prompt, candidate] for candidate in candidates], return_tensors="pt", padding=True)
        outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=label)
        logits = outputs.logits
        predicted_class = logits.argmax().item()
        if predicted_class == label[0]:
            correct += 1
        if card_wins is not None:
            popular_card = card_wins.reindex(candidates).sort_values('pick_ratio', ascending=False).index[0]
            popular_label = candidates.index(popular_card)
            if popular_label == predicted_class:
                popular += 1
    print(name + ': ')
    print('total:', correct, '/', rounds, '=', correct/rounds)
    if card_wins is not None:
        print('popular:', popular, '/', rounds, '=', popular/rounds)
    print()
