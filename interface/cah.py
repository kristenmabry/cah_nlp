import json
from transformers import AutoTokenizer
from transformers import AutoModelForMultipleChoice

tokenizer = AutoTokenizer.from_pretrained("cah_model")
model = AutoModelForMultipleChoice.from_pretrained("cah_model")

cards = {}
with open('data/data_appropriate.json', 'r') as f:
    cards = json.load(f)
    f.close()

def getWinner(prompt, candidates, num_cards):
  tokens = []
  if num_cards == 2:
    for i in range(10):
      answer = candidates[i] + ' <sep> ' + candidates[i+10]
      tokens.append([prompt, answer])
  else:
    tokens = [[prompt, candidate] for candidate in candidates]
  inputs = tokenizer(tokens, return_tensors="pt", padding=True)
  outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()})
  logits = outputs.logits
  return logits.argmax().item()
