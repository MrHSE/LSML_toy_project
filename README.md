# LSML_toy_project

This project is devoted to the question-answering task. The aim is to work with the **BoolQ** dataset from SuperGLUE.

BoolQ is a question answering dataset for yes/no. 

Each example is a triplet of (question, passage, answer), with the title of the page as optional additional context. The dataset release consists of three `.jsonl` files (`train, val, test`), where each line is a JSON dictionary with the following format:

    Example:
    
    {
      "question": "is france the same timezone as the uk",
      "passage": "At the Liberation of France in the summer of 1944, Metropolitan France kept GMT+2 as it was the time then used by the Allies (British Double Summer Time). In the winter of 1944--1945, Metropolitan France switched to GMT+1, same as in the United Kingdom, and switched again to GMT+2 in April 1945 like its British ally. In September 1945, Metropolitan France returned to GMT+1 (pre-war summer time), which the British had already done in July 1945. Metropolitan France was officially scheduled to return to GMT+0 on November 18, 1945 (the British returned to GMT+0 in on October 7, 1945), but the French government canceled the decision on November 5, 1945, and GMT+1 has since then remained the official time of Metropolitan France."
      "answer": false,
      "title": "Time in France",
    }

Dataset is available here: https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip

### Now let's describe the model architecture
Project has two steps:
1) using pretrained BERT model without any additional learning as a word2vec model and training a classifier afterwards;
2) updating BERT to return a classification and learning it to solve our problem.

### The way first model built:
1) tokenizing words of each passage using BERT pretrained model (I used "bert-base-uncased");
2) getting passages embeddings by four ways: "mean", "min", "max" and "sum" of vectors got on stage 1;
3) tokenizing words of each question using BERT;
4) getting questions embeddings by four ways: "mean", "min", "max" and "sum"
5) getting cosine similarity between question and passage vectors;
6) training RandomForestClassifier with cosine similarities and labels with:
    - 300 estimators
    - unbounded maximum depth of a tree
    - bootstrap True
7) gini coef is used as a **loss metric**
8) used such metrics as precision, recall and f1-score

### Second model
"bert-base-uncased" BERT pretrained model is used
Global params for BERT learning are:
BATCH_SIZE = 12
LR = 1e-5
EPOCHS = 10
MAX_SEQ_LEN = 256
optimizer = AdamW

## Model
**Trained model** is available by this link from yandex disk: https://disk.yandex.ru/d/sHw_6SW8IuZYZw

For evaluating model efficiency accuracy is used

## HTML frontend
To deliver model to consumers a simple flask+html system was created.
Flask with Jinja returns an HTML from by response with several fields:
- Question;
- Passage;
- Answer.
To get the answer you need to write your question, passage and submit it. After a second flask will return rendered HTML page with the answer.
