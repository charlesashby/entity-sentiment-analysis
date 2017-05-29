import CharLSTM
from pycorenlp import StanfordCoreNLP
from hobbs import *

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

test = 'Mr. BLodwij is interested in knowing more about this company, but he is lost.'

def print_tree(output):
    print(output['sentences'][0]['parse'])
    return True

def parse_sentence(sentence):
    """ sentence --> named-entity chunked tree """
    output = nlp.annotate(sentence, properties={'annotators': 'tokenize, ssplit, pos, depparse, parse,',
                                            'outputFormat': 'json'})
    print_tree(output)
    return Tree.fromstring(output['sentences'][0]['parse'])

def anaphora_resolution(tree):
    """ Convert pronouns to their respective noun """
    # Get a list of all the pronouns to be resolved
    # TODO: Discard non-relevant entities
    pronouns = find_pronouns(tree)
    entity_positions = []

    for prp in pronouns:
        # Change the pronoun for the noun found with Hobbs' algo
        # e.g. Mr. blah blah is here, but he is mad.
        # ---> Mr. blah blah is here but Mr. blah blah is mad
        _, noun = hobbs([tree], prp)
        tree[prp] = tree[noun]
        entity_positions.append(prp)
        if not noun in entity_positions:
            entity_positions.append(noun)

    return tree, entity_positions

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
            if isinstance(child, nltk.Tree):
                queue.put(child)
        if "S" in node.label():
            # if childs are (respectively) 'NP' and 'VP'
            # convert subtree to string, else keep looking
            child_labels = [child.label() for child in node]

            if "NP" in child_labels and "VP" in child_labels:
                sentence = tree_to_str(node)
                for child in node:
                    if child.label() == "NP":
                        noun = child
                subtrees.append((tree_to_str(noun), sentence))

    return subtrees


"""




from CharLSTM.lib import data_utils
from CharLSTM.lib_model import char_lstm
network = char_lstm.LSTM()
network.build()
from parse_doc import *
tree = anaphora_resolution(parse_sentence(test))
st = get_subtrees(tree[0])
sentences = [s[1] for s in st]
tt = network.categorize_sentences(sentences)
"""














