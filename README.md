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
The way model built:
* tokenizing words of each passage using BERT
* getting passages embeddings by three ways: "mean", "min", "max" and "sum"
* tokenizing words of each question using BERT
* getting questions embeddings by three ways: "mean", "min", "max" and "sum"
* getting cos similarity between question and passage
* training RandomForestClassifier with cos similarities and labels
