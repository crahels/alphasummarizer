from anytree import Node, RenderTree
import pickle, re, stanfordnlp
from ..models.sentence import Sentence
from ..models.token import Token
import networkx as nx
import os
from pathlib import Path

exception_pos_tags = ["PUNCT", "SYM", "X"]

def initialize_stanford_nlp():
    nlp = stanfordnlp.Pipeline(lang="id", treebank="id_gsd")
    return nlp

def print_tree(root):
    for pre, fill, node in RenderTree(root):
        print("%s%s" % (pre, node.name))

def tokenize(doc, idx_topic, idx_news):
    sentences = []
    for idx_sentence, sentence in enumerate(doc.sentences):
        tokens = []
        token_words = []
        for word in sentence.words:
            tok = Node(Token(int(word.index), word.text, word.upos, word.dependency_relation, word.governor))
            tokens.append(tok)
            token_words.append(word.text)

        root = None
        for token in tokens:
            governor = token.name.governor
            if (governor > 0):
                token.parent = tokens[governor - 1]
            else:
                root = token

        sentence_obj = Sentence(idx_topic, idx_news, idx_sentence, sentence, tokens, root, len(doc.sentences))
        sentences.append(sentence_obj)

    return sentences

def save_object(file_dir, obj):
    file_dest = file_dir + ".dat" 
    with open(file_dest, "wb") as f:
        pickle.dump(obj, f)

def load_object(file_dir):
    file_src = file_dir + ".dat"
    with open(file_src, "rb") as f:
        obj = pickle.load(f)
        return obj

def find_whole_word(w):
    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search

def save_graph(file_dir, obj):
    nx.write_gpickle(obj, file_dir + ".gpickle")

def load_graph(file_dir):
    return nx.read_gpickle(file_dir + ".gpickle")

def load_corpus(corpus_name, summary_type, corpus_topics, corpus_pas, sentence_similarity_table, corpus_summaries, graph_docs):
    pwd = Path(os.path.dirname(__file__)).parent
    pwd = str(pwd)
    postfix = str(summary_type) 

    corpus_pas[corpus_name] = load_object(pwd + "/temporary_data_" + postfix + "/corpus_pas/" + corpus_name)
    sentence_similarity_table[corpus_name] = load_object(pwd + "/temporary_data_" + postfix + "/sentence_similarity_table/" + corpus_name)
    corpus_summaries[corpus_name] = load_object(pwd + "/temporary_data_" + postfix + "/corpus_summaries_" + postfix + "/" + corpus_name)
    graph_docs[corpus_name] = load_graph(pwd + "/temporary_data_" + postfix + "/graph_docs/" + corpus_name)

# MMR
def get_argument_tokens(arguments):
    tokens = []
    for argument in arguments:
        tokens.extend(argument)
    return tokens

def get_argument_tokens_without_punctuation(pas_tokens, arguments):
    tokens = []
    for argument in arguments:
    	for word in argument:
            if (pas_tokens[word].name.pos_tag not in exception_pos_tags):
                tokens.append(word)
    return tokens

def get_tokens(extracted_pas):
    tokens = []
    tokens.extend(get_argument_tokens(extracted_pas.pas.subjects))
    tokens.extend(get_argument_tokens(extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens(extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens(extracted_pas.pas.times))
    tokens.extend(get_argument_tokens(extracted_pas.pas.places))
    tokens.extend(get_argument_tokens(extracted_pas.pas.explanations))
    tokens.sort()
    return tokens

def get_tokens_without_punctuation(extracted_pas):
    tokens = []
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.subjects))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.times))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.places))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.explanations))
    tokens.sort()
    return tokens
  
def get_subjects_tokens(extracted_pas):
    tokens = get_argument_tokens(extracted_pas.pas.subjects)
    tokens.sort()
    return tokens

def get_subjects(extracted_pas):
    tokens = get_subjects_tokens(extracted_pas)

    subjects = [extracted_pas.tokens[token].name.text for token in tokens]
    return " ".join(subjects)

def get_first_subject_tokens(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 0):
        for subj in extracted_pas.pas.subjects[0]:
            if extracted_pas.tokens[subj].name.pos_tag not in exception_pos_tags:
                tokens.append(subj)

    return tokens

def get_first_subject(extracted_pas):
    tokens = get_first_subject_tokens(extracted_pas)

    subjects = [extracted_pas.tokens[token].name.text for token in tokens]
    return " ".join(subjects)

def get_tokens_without_first_subject(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 1):
        tokens.extend(get_argument_tokens(extracted_pas.pas.subjects[1:len(extracted_pas.pas.subjects)]))

    tokens.extend(get_argument_tokens(extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens(extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens(extracted_pas.pas.times))
    tokens.extend(get_argument_tokens(extracted_pas.pas.places))
    tokens.extend(get_argument_tokens(extracted_pas.pas.explanations))
    tokens.sort()
    return tokens

def get_tokens_without_first_subject_without_punctuation(extracted_pas):
    tokens = []
    if (len(extracted_pas.pas.subjects) > 1):
        tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.subjects[1:len(extracted_pas.pas.subjects)]))

    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.predicates))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.objects))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.times))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.places))
    tokens.extend(get_argument_tokens_without_punctuation(extracted_pas.tokens, extracted_pas.pas.explanations))
    tokens.sort()
    return tokens

def print_argument(title, subtitle, tokens, arguments):
    print(title)
    for idx_argument, argument in enumerate(arguments):
        print(subtitle + str(idx_argument))
        for word in argument:
            print(tokens[word].name)
    print()

def print_result(extracted_pas):
    print_argument("Subjek", "Subjek ke: ", extracted_pas.tokens, extracted_pas.pas.subjects)
    print_argument("Predikat", "Predikat ke: ", extracted_pas.tokens, extracted_pas.pas.predicates)
    print_argument("Objek", "Objek ke: ", extracted_pas.tokens, extracted_pas.pas.objects)
    print_argument("Keterangan", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.pas.explanations)
    print_argument("K. Waktu", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.pas.times)
    print_argument("K. Tempat", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.pas.places)

def print_clean_result(extracted_pas):
    # print_argument("Subjek", "Subjek ke: ", extracted_pas.tokens, extracted_pas.pas.subjects)
    print_argument("Subjek (clean)", "Subjek ke: ", extracted_pas.tokens, extracted_pas.clean_pas.subjects)
    
    # print_argument("Predikat", "Predikat ke: ", extracted_pas.tokens, extracted_pas.pas.predicates)
    print_argument("Predikat (clean)", "Predikat ke: ", extracted_pas.tokens, extracted_pas.clean_pas.predicates)
    
    # print_argument("Objek", "Objek ke: ", extracted_pas.tokens, extracted_pas.pas.objects)
    print_argument("Objek (clean)", "Objek ke: ", extracted_pas.tokens, extracted_pas.clean_pas.objects)
    
    # print_argument("Keterangan", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.pas.explanations)
    print_argument("Keterangan (clean)", "Keterangan ke: ", extracted_pas.tokens, extracted_pas.clean_pas.explanations)
    
    # print_argument("K. Waktu", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.pas.times)
    print_argument("K. Waktu (clean)", "K. Waktu ke: ", extracted_pas.tokens, extracted_pas.clean_pas.times)
    
    # print_argument("K. Tempat", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.pas.places)
    print_argument("K. Tempat (clean)", "K. Tempat ke: ", extracted_pas.tokens, extracted_pas.clean_pas.places)
