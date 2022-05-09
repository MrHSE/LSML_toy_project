from flask import Flask, request, render_template, jsonify
from matplotlib.cbook import report_memory
import config.development as config

import numpy as np
from gensim import models
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("updated_bert")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

BATCH_SIZE = 12
LR = 1e-5
EPOCHS = 10
MAX_SEQ_LEN = 256

def encode_data(tokenizer, questions, passages, max_length):
    input_ids = []
    attention_masks = []

    for question, passage in zip(questions, passages):
        encoded_data = tokenizer.encode_plus(question, passage, max_length=max_length, pad_to_max_length=True, truncation_strategy="longest_first", add_special_tokens=True)
        encoded_pair = encoded_data["input_ids"]
        attention_mask = encoded_data["attention_mask"]

        input_ids.append(encoded_pair)
        attention_masks.append(attention_mask)

    return np.array(input_ids), np.array(attention_masks)

def answer_quest(ques, passage):
    ques = np.array([ques])
    passage = np.array([passage])
    input_ids, attention_masks = encode_data(tokenizer, ques, passage, MAX_SEQ_LEN)
    features = (input_ids, attention_masks)
    features_tensors = [torch.tensor(feature, dtype=torch.long) for feature in features]
    dataset = TensorDataset(*features_tensors)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    model.eval()
    for step, batch in enumerate(dataloader):

        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        # labels = batch[2].to(device)

        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_masks)

        logits = outputs[0]
        logits = logits.detach().cpu().numpy()

        predictions = np.argmax(logits, axis=1).flatten()
        return predictions

app = Flask(__name__, template_folder='static')
app.config.from_object(config)

@app.route('/', methods=['GET', 'POST'])
def home():

    response = ""
    passage = ""
    ques = ""
    if request.method == 'GET':
        return render_template('main.html', response=response, ques=ques, passage=passage)
    elif request.method == 'POST':
        ques = request.form["ques"]
        passage = request.form["passage"]
        response = int(answer_quest(ques, passage)[0])
        print(response)
        if response == 1:
            response = "Yes"
        else:
            response = "No"
        return render_template('main.html', response=response, ques=ques, passage=passage)

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
