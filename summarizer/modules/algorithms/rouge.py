from pythonrouge.pythonrouge import Pythonrouge
# from pprint import pprint

rouge = Pythonrouge(summary_file_exist=True,
    summary=None, reference=None,
    recall_only=True,
    peer_path="", model_path="",
    n_gram=2, ROUGE_SU4=False, ROUGE_L=False,
    word_level=True,
    scoring_formula='average')

def transform_candidate_summary(summary_paragraph):
    candidate_summary = []
    for sent in summary_paragraph:
        candidate_summary.append(" ".join(sent))
    return candidate_summary

def transform_reference_summaries(corpus_summaries):
    reference_summaries = []
    for summ in corpus_summaries:
        reference_summary = []
        for sent in summ:
            reference_summary.append(" ".join(sent))
        reference_summaries.append(reference_summary)
    return reference_summaries

def calculate_rouge(folder, postfix):
    rouge.peer_path = folder + "/candidate_summaries/" + postfix + "/"
    rouge.model_path = folder + "/reference_summaries/" + postfix + "/"
    score = rouge.calc_score()
    return score["ROUGE-2"]


def calculate_direct_rouge(candidate_summary, reference_summaries):
    direct_rouge = Pythonrouge(summary_file_exist=False,
                        summary=candidate_summary, reference=reference_summaries,
                        recall_only=True,
                        n_gram=2, ROUGE_SU4=False, ROUGE_L=False,
                        word_level=True,
                        scoring_formula='best')

    score = direct_rouge.calc_score()
    return score["ROUGE-2"]
