from ..utils.main_utils import get_tokens
from ..utils.main_utils import get_first_subject_tokens
from ..utils.main_utils import get_first_subject
from ..utils.main_utils import get_subjects
from ..utils.main_utils import get_tokens_without_first_subject
from .levenshtein import levenshtein_distance

def natural_language_generation(summary, corpus_pas):
    summary_pas = []
    for idx in summary:
        summary_pas.append(corpus_pas[idx])

    summary_pas.sort(key = lambda x: (x.idx_sentence, x.idx_news))

    # grouped with subjects
    grouped_summary_pas = []
    picked_pas = []
    for idx_1, pas_1 in enumerate(summary_pas):
        if (idx_1 not in picked_pas):
            a_group = [pas_1]
            picked_pas.append(idx_1)
            for idx_2, pas_2 in enumerate(summary_pas):
                if idx_1 != idx_2:
                    tokens_1 = get_tokens(pas_1)
                    tokens_2 = get_tokens(pas_2)
                    
                    subject_tokens_1 = get_first_subject_tokens(pas_1)
                    subject_tokens_2 = get_first_subject_tokens(pas_2)

                    subject_1 = get_first_subject(pas_1).lower()
                    subject_2 = get_first_subject(pas_2).lower()
                    
                    distance = levenshtein_distance(subject_1, subject_2)
                    if ((tokens_1[0] in subject_tokens_1) and (tokens_2[0] in subject_tokens_2)) and (distance >= 0 and distance <= 0.3):
                        a_group.append(pas_2)
                        picked_pas.append(idx_2)

            grouped_summary_pas.append(a_group)

    # for pases in grouped_summary_pas:
    #     print("group:")
    #     for pas in pases:
    #         print(get_subjects(pas))

    summary_paragraph = []
    idx_grouped_summary_pas = 0
    while idx_grouped_summary_pas < len(grouped_summary_pas):
        pases = grouped_summary_pas[idx_grouped_summary_pas]

        if (len(pases) > 1):
            pases.sort(key = lambda x: len(get_first_subject_tokens(x)), reverse = True)

            summary_sentence = []
            for idx, pas in enumerate(pases):
                if (idx == 0):
                    subject_tokens = get_first_subject_tokens(pas)
                    other_tokens = get_tokens_without_first_subject(pas)
                    summary_sentence.extend([pas.tokens[token].name.text for token in subject_tokens])
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
                elif (idx == len(pases) - 1):
                    other_tokens = get_tokens_without_first_subject(pas)
                    if len(pases) == 2:
                        summary_sentence.append("dan")
                    else:
                        summary_sentence.extend([",", "dan"])
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
                else:
                    other_tokens = get_tokens_without_first_subject(pas)
                    summary_sentence.append(",")
                    summary_sentence.extend([pas.tokens[token].name.text for token in other_tokens])
            summary_sentence.append(".")
            summary_paragraph.append(summary_sentence)
        else:
            pas = pases[0]
            tokens = get_tokens(pas)
            summary_sentence = [pas.tokens[token].name.text for token in tokens]
            summary_sentence.append(".")
            summary_paragraph.append(summary_sentence)

        idx_grouped_summary_pas += 1

    return summary_paragraph
