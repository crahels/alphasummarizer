from copy import deepcopy
from utils.main_utils import find_whole_word
from utils.pas_utils import write_result
from models.extracted_pas import ExtractedPAS
from models.pas import PAS
import re

subject_relations = ["nsubj"]
predicate_relations = ["root", "advmod", "amod"]
object_relations = ["obj", "nummod"]
explanation_relations = ["case", "mark", "det"]
new_explanation_relations = ["obl"]
further_relations = ["advcl", "xcomp", "dep", "ccomp", "conj", "parataxis"]
place_keywords = ["di", "ke", "dari"]
time_keywords = ["setiap", "setelah", "sesudah", "sebelum", "hingga", "semenjak", "sejak", "sampai", "selama", "saat", "kali", "ketika", "kemarin", "sesaat"]
special_keywords = ["sekitar", "pada"]
noise_keywords = ["yakni", "yaitu"]
time_general_keywords = ['senin', 'selasa', 'rabu', 'kamis', 'jumat', 'sabtu', 'minggu', 'januari', 'februari', 'maret', 'april', 'mei', 'juni', 'juli', 'agustus', 'september'
                        'oktober', 'november', 'desember', 'pukul', 'wib', 'wita', 'wit', 'jam', 'menit', 'hari', 'waktu', 'kemarin']

def get_children(words, tree):
    words.append(tree.name.position - 1)
    for child in tree.children:
        get_children(words, child)

def get_arguments_and_explanations(idx, words, explanations, tree):
    words.append(tree.name.position - 1)
    for child in tree.children:
        get_arguments_and_explanations(1, words, explanations, child)

def get_explanations_helper(explanations, tree):
    explanation_words = []
    get_explanations(explanation_words, explanations, tree)
    explanation_words.sort()
    explanations.append(explanation_words)

def get_explanations(words, explanations, tree):
    words.append(tree.name.position - 1)
    for child in tree.children:
        get_explanations(words, explanations, child)

def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2))

def extract_pas_from_sentence(tree, idx_sentence_in_corpus, sentence, parent, corpus_pas):
    subjects = []
    predicates = []
    objects = []
    explanations = []
    other_explanations = []
    times = []
    places = []

    # predicate
    if (tree.name.pos_tag.lower() == 'verb'):
        predicate_words = []
        predicate_words.append(tree.name.position - 1)
        for child in tree.children:
            if child.name.relation.lower() in predicate_relations:
                predicate_words.append(child.name.position - 1)
        predicate_words.sort()
        predicates.append(predicate_words)
    
    has_child = False
    have_children = {}
    for idx_child, child in enumerate(tree.children):
        # initialize
        have_children[idx_child] = False
        # subject
        if (child.name.relation.lower().startswith(tuple(subject_relations))):
            subject_words = []
            get_arguments_and_explanations(0, subject_words, explanations, child)
            subject_words.sort()
            subjects.append(subject_words)
        # object
        elif (child.name.relation.lower().startswith(tuple(object_relations))):
            object_words = []
            get_arguments_and_explanations(0, object_words, explanations, child)
            object_words.sort()
            objects.append(object_words)
        # explanation
        elif (any(next_child.name.relation.lower() in explanation_relations for next_child in child.children[:1])
            or (child.name.relation.lower() in explanation_relations + new_explanation_relations)):
            get_explanations_helper(explanations, child)
        # complex sentence
        elif (child.name.pos_tag.lower() == 'verb' and child.name.relation.lower() in further_relations):
            has_child = True
            have_children[idx_child] = True

    for explanation in explanations:
        idx_attention = 0
        if (sentence.tokens[explanation[idx_attention]].name.text.lower() in noise_keywords):
            idx_exp = idx_attention + 1
            while (idx_exp < len(explanation)) and (sentence.tokens[explanation[idx_exp]].name.relation.lower() not in explanation_relations):
                idx_exp += 1

            if idx_exp < len(explanation):
                idx_attention = idx_exp

        if (sentence.tokens[explanation[idx_attention]].name.text.lower() in time_keywords):
            times.append(explanation)
        elif (sentence.tokens[explanation[idx_attention]].name.text.lower() in place_keywords):
            places.append(explanation)
        else:
            # match time keywords, regex for date, or regex for time
            idx_boundary = len(explanation)
            if (sentence.tokens[explanation[idx_attention]].name.text.lower() not in special_keywords):
                counter = 0
                for idx_exp, exp in enumerate(explanation[idx_attention:len(explanation)]):
                    if (sentence.tokens[exp].name.relation.lower() in explanation_relations): 
                        counter += 1
                        idx_boundary = idx_exp + idx_attention
                        if (counter >= 2):
                            break

                if (idx_boundary == 0):
                    idx_boundary = len(explanation)

            text = " ".join([sentence.tokens[idx_exp].name.text.lower() for idx_exp in explanation[:idx_boundary]])
            if (any(bool(find_whole_word(word)(text)) for word in time_general_keywords) or bool(re.search(r"([^\d]|^)\d{1,2}(\/\d{1,2})(\/\d{2,4})?([^\d]|$)", text)) or bool(re.search(r"([^\d]|^)\d{1,2}((\.|:)\d{1,2}){1,2}([^\d]|$)", text))):
                times.append(explanation)
            else:
                other_explanations.append(explanation)

    # check if the pas is deserved to be called pas
    pas = None
    extract_pas = True
    analyse_pas = True
    if parent is not None:
        if len(predicates) > 0:
            # handle xcomp by merging with parent pas
            if (any(intersection([sentence.tokens[predicates[0][idx_pred]].name.relation.lower() for idx_pred in range(len(predicates[0]))], ['xcomp', 'parataxis', 'ccomp']))):
            # if (sentence.tokens[predicates[0][-1]].name.relation.lower() in ['xcomp', 'parataxis', 'ccomp'] or sentence.tokens[predicates[0][0]].name.relation.lower() in ['xcomp', 'parataxis', 'ccomp']):
                extract_pas = False
                parent.pas.subjects = parent.pas.subjects + subjects
                parent.pas.predicates = parent.pas.predicates + predicates
                parent.pas.objects = parent.pas.objects + objects
                parent.pas.explanations = parent.pas.explanations + other_explanations
                parent.pas.times = parent.pas.times + times
                parent.pas.places = parent.pas.places + places
            # handle advcl, parataxis, and ccomp by making the pas as parent's explanation
            elif ('advcl' in [sentence.tokens[predicates[0][idx_pred]].name.relation.lower() for idx_pred in range(len(predicates[0]))]):
            # elif (sentence.tokens[predicates[0][-1]].name.relation.lower() == 'advcl' or sentence.tokens[predicates[0][0]].name.relation.lower() == 'advcl'):
                extract_pas = False

                explanation = []
                for row in (subjects + predicates + objects + times + places + other_explanations):
                    for j in row:
                        explanation.append(j)
                explanation.sort()
                first_word = sentence.tokens[explanation[0]]

                if (first_word.name.relation.lower() in explanation_relations):
                    if (first_word.name.text.lower() in time_keywords):
                        parent.pas.times.append(explanation)
                    elif (first_word.name.text.lower() in place_keywords):
                        parent.pas.places.append(explanation)
                    else:
                        parent.pas.explanations.append(explanation)
                else:
                    parent.pas.explanations.append(explanation)
            # handle conj by add parent's subject
            elif ('conj' in [sentence.tokens[predicates[0][idx_pred]].name.relation.lower() for idx_pred in range(len(predicates[0]))]):
            # elif (sentence.tokens[predicates[0][-1]].name.relation.lower() == 'conj' or sentence.tokens[predicates[0][0]].name.relation.lower() == 'conj'):
                if len(subjects) == 0:
                    subjects = deepcopy(parent.pas.subjects)

        if (len(parent.pas.predicates) > 0 and (len(parent.pas.subjects) + len(parent.pas.objects) +
            len(parent.pas.explanations) + len(parent.pas.times) + len(parent.pas.places) >= 1)):
            pas = parent
            parent.delete = False
        else:
            parent.delete = True

    if extract_pas:
        if not has_child:
            if not (len(predicates) > 0 and (len(subjects) + len(objects) + len(other_explanations) + len(times) + len(places) >= 1)):
                analyse_pas = False
        if analyse_pas:
            pas = ExtractedPAS(parent, idx_sentence_in_corpus, sentence, PAS(subjects, predicates, objects, other_explanations, times, places))
            corpus_pas.append(pas)

    # complex sentence
    for idx_child, child in enumerate(tree.children):
        if have_children[idx_child]:
            extract_pas_from_sentence(child, idx_sentence_in_corpus, sentence, pas, corpus_pas)

def extract_pas_from_sentence_caller(corpus_sentences, corpus_pas):
    for idx, sentence in enumerate(corpus_sentences):
        extract_pas_from_sentence(sentence.root, idx, sentence, None, corpus_pas)

def predict_pas(corpus_test, corpus_pas, corpus_docs, pas_prediction):
    for corpus_name in corpus_test:
        corpus_sentences = corpus_docs[corpus_name]["sentences"]
        corpus_pas[corpus_name] = []
        extract_pas_from_sentence_caller(corpus_sentences, corpus_pas[corpus_name])
        corpus_pas[corpus_name] = [extracted_pas for extracted_pas in corpus_pas[corpus_name] if extracted_pas.delete == False]

        f = open("corpus_pas/" + corpus_name + "-pred" + ".txt","w+")

        pas_prediction[corpus_name] = []
        for extracted_pas in corpus_pas[corpus_name]:
#             print(" ".join([o.name.text for o in extracted_pas.tokens]))
#             print_tree(extracted_pas.root)
            f.write(" ".join([o.name.text for o in extracted_pas.tokens]) + "\r\n")
            write_result(f, extracted_pas, pas_prediction[corpus_name])
        f.close()
