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
--- stanford-corenlp-full-2016-10-31
```

## Exemple Usage

First, download the Stanford-CoreNLP server files at this [page](http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip). And type the following commands in a terminal to start it.

```
# Start the Stanford-CoreNLP server
cd news-sentiment-analysis/stanford-corenlp-*
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
```

Then, you can use this script to perform anaphora resolution, entity extraction & sentiment analysis on any sentence by modifying `parse_doc.py`. As an example, here's the result by running this script on the sentence: "Jean is really sad, but Adam is the happiest guy ever".

```
python news-sentiment-analysis/parse_doc.py

>>> Sentence: 0,Jean is really sad , yielded results (pos/neg): 0.35512/0.64488, prediction: neg
>>> Sentence: 0,Adam is the happiest guy ever , yielded results (pos/neg): 0.97269/0.02731, prediction: pos
>>> Entity:  Jean -- sentiment: -0.2897535
>>> Entity:  Adam -- sentiment: 0.94538456
```

You can read the full blog post [here](https://charlesashby.github.io/2017/06/05/sentiment-analysis-with-char-lstm/).
