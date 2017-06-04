# This code assumes you are using CharLSTM from https://github.com/charlesashby/CharLSTM
# and the Stanford CoreNLP server. 
#
# Results for sentence: Jean is really sad, but Adam is the happiest guy ever
# Entity:  Jean -- sentiment: -0.197092 (neg)
# Entity:  Adam -- sentiment: 0.885632  (pos)

from pycorenlp import StanfordCoreNLP
from nltk.tokenize import sent_tokenize
from nltk import Tree

# The following imports are for CharLSTM -- change your path accordingly
CharLSTM_PATH = '/home/ashbylepoc/PycharmProjects/CharLSTM/'
import sys; sys.path.append(CharLSTM_PATH)
from lib_model.char_lstm import *

try:
    nlp = StanfordCoreNLP('http://localhost:9000')
    SERVER_RUNNING = True
except:
    print('Did you forget to start the StanfordCoreNLP Server?')
    print('$ cd stanford-corenlp-*')
    print('$ java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000')
    SERVER_RUNNING = False

assert SERVER_RUNNING == True, "Download the server files at: http://nlp.stanford.edu/software/stanford-" \
                               "corenlp-full-2016-10-31.zip"

def print_tree(output):
    print(output['sentences'][0]['parse'])
    return True

def get_rep_mention(coreference):
    for reference in coreference:
        if reference['isRepresentativeMention'] == True:
            pos = (reference['startIndex'], reference['headIndex'])
            text = reference['text']
            return text, pos

def parse_sentence(sentence):
    """ sentence --> named-entity chunked tree """
    try:
        output = nlp.annotate(sentence, properties={'annotators':   'tokenize, ssplit, pos,'
                                                                    ' lemma, ner, parse',
                                                    'outputFormat': 'json'})
        # print_tree(output)
        return Tree.fromstring(output['sentences'][0]['parse'])
    except TypeError as e:
        import pdb; pdb.set_trace()

def coreference_resolution(sentence):
    # coreference resolution
    output = nlp.annotate(sentence, properties={'annotators':   'coref',
                                                'outputFormat': 'json'})
    tokens = word_tokenize(sentence)
    coreferences = output['corefs']
    entity_keys = coreferences.keys()
    for k in entity_keys:
        # skip non PERSON NP
        if coreferences[k][0]['gender'] == 'MALE' or coreferences[k][0]['gender'] == 'FEMALE':
            rep_mention, pos = get_rep_mention(coreferences[k])
            for reference in coreferences[k]:
                if not reference['isRepresentativeMention']:
                    start, end = reference['startIndex'] - 1, reference['headIndex'] - 1
                    if start == end:
                        tokens[start] = rep_mention
                    else:
                        tokens[start] = rep_mention
                        del tokens[start + 1: end]

    sentence = ' '.join(tokens)
    return sentence.encode('utf-8')

def tree_to_str(tree):
    return ' '.join([w for w in tree.leaves()])

def get_subtrees(tree):
    """ Return chunked sentences """
    
    subtrees = []
    queue = Queue.Queue()
    queue.put(tree)
    
    while not queue.empty():
        node = queue.get()
        
        for child in node:
            if isinstance(child, Tree):
                queue.put(child)
                
        if node.label() == "S":
            # if childs are (respectively) 'NP' and 'VP'
            # convert subtree to string, else keep looking

            # TODO: MAKE SURE NP IS A PERSON
            child_labels = [child.label() for child in node]

            if "NP" in child_labels and "VP" in child_labels:
                sentence = tree_to_str(node)
                for child in node:
                    if child.label() == "NP":
                        # look for NNP
                        subchild_labels = [subchild.label() for subchild in child]
                        if "NNP" in subchild_labels:
                            noun = ""
                            for subchild in child:
                                if subchild.label() == "NNP":
                                    noun = ' '.join([noun, subchild.leaves()[0]])

                            subtrees.append((noun, sentence))
    return subtrees

def flatten(list):
    return [val for sublist in list for val in sublist]

def parse_doc(document):
    """ Extract relevant entities in a document """
    print('Tokenizing sentences...')
    sentences = sent_tokenize(document)
    print('Done!')
    # Context of all named entities
    ne_context = []
    for sentence in sentences:
        # change pronouns to their respective nouns
        print('Anaphora resolution for sentence: %s' % sentence)
        tree = parse_sentence(coreference_resolution(sentence))
        print('Done!')

        # get context for each noun
        print('Named Entity Clustering:')
        context = get_subtrees(tree)
        for n, s in context:
            print('%s' % s)
        ne_context.append(context)
    return flatten(ne_context)

def init_dict(contexts):
    dict = {}
    for k, _ in contexts:
        if not k in dict:
            dict[k] = None
    return dict

def load_model():
    assert os.path.exists(CharLSTM_PATH), 'Did you forget to change the path to CharLSTM?'
    network = LSTM()
    network.build(training=False)
    return network

def get_sentiment(document, network):
    """ Create a dict of every entities with their associated sentiment """
    print('Parsing Document...')
    contexts = parse_doc(document)
    print('Done!')
    entities = init_dict(contexts)
    sentences = [sentence.encode('utf-8') for _, sentence in contexts]
    predictions = network.categorize_document(sentences)

    for i, c in enumerate(contexts):
        key = c[0]
        if entities[key] != None:
            entities[key] += (predictions[0][i][0] - predictions[0][i][1])
            entities[key] /= 2
        else:
            entities[key] = (predictions[0][i][0] - predictions[0][i][1])

    for e in entities.keys():
        print('Entity: %s -- sentiment: %s' % (e, entities[e]))

if __name__ == '__main__':
    text = 'Jean is really sad, but Adam is the happiest guy ever'
    network = LSTM()
    network.build()
    get_sentiment(text, network)
