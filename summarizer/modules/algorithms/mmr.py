from ..utils.main_utils import get_tokens_without_punctuation
from .graph import get_idx_j_val

def get_max_similarity(cand_elmt, summary, sentence_similarity_table):
    max_sim = 0.0
    for sum_elmt in summary:
        i = -1
        j = -1
        if (cand_elmt < sum_elmt):
            i = cand_elmt
            j = get_idx_j_val(i, sum_elmt)
        else:
            i = sum_elmt
            j = get_idx_j_val(i, cand_elmt)
        if sentence_similarity_table[i][j] > max_sim:
            max_sim = sentence_similarity_table[i][j]
    return max_sim

def maximal_marginal_relevance(max_sum_length, corpus, graph_sentences, num_iter, sentence_similarity_table):
    not_chosen = [i for i in range(len(corpus))]
    summary = []

    sum_length = 0
    get_summary = True
    max_score = max(graph_sentences.node[i][num_iter] for i in not_chosen)
    for i in not_chosen:
        if graph_sentences.node[i][num_iter] == max_score:
            length_new_summary = len(get_tokens_without_punctuation(corpus[i]))
            if (length_new_summary + sum_length > max_sum_length):
                get_summary = False
            else:
                sum_length += length_new_summary
                summary.append(i)
                not_chosen.remove(i)
                if sum_length == max_sum_length:
                    get_summary = False
            break

    while (get_summary and len(not_chosen) > 0):
        max_mmr = 0.0
        idx_max_mmr = not_chosen[0]
        for i in not_chosen:
            max_sim = get_max_similarity(i, summary, sentence_similarity_table)
            mmr = (0.5 * graph_sentences.node[i][num_iter]) - (0.5 * max_sim)
            if mmr > max_mmr:
                max_mmr = mmr
                idx_max_mmr = i

        length_new_summary = len(get_tokens_without_punctuation(corpus[idx_max_mmr]))
        if (length_new_summary + sum_length > max_sum_length):
            get_summary = False
        else:
            sum_length += length_new_summary # len(corpus_sentences[idx_max_mmr].sentence.words)
            summary.append(idx_max_mmr)
            not_chosen.remove(idx_max_mmr)
            if sum_length == max_sum_length:
                get_summary = False

    return summary
