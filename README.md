# Sentiment Analysis on Extracted Entities 
Various ops for handling several entities in a document, performs anaphora resolution, clustering, etc..

## Requirements
- Python 2.7
- [CharLSTM model](https://github.com/charlesashby/CharLSTM/)

## Directory Structure
This code assume you have the following directory structure, where CharLSTM is taken from this [page](https://github.com/charlesashby/CharLSTM/).
```
- Main
-- CharLSTM
-- news-sentiment-analysis
```

## Exemple Usage

You can use this script has follows to perform anaphora resolution & sentiment analysis on the sentence "Jean is really sad, but Adam is the happiest guy ever".

```
python news-sentiment-analysis/parse_doc.py

>>> Sentence: 0,Jean is really sad , yielded results (pos/neg): 0.35512/0.64488, prediction: neg
>>> Sentence: 0,Adam is the happiest guy ever , yielded results (pos/neg): 0.97269/0.02731, prediction: pos
>>> Entity:  Jean -- sentiment: -0.2897535
>>> Entity:  Adam -- sentiment: 0.94538456
```
