from .algorithms.graph import semantic_graph_modification
from .algorithms.graph import GraphAlgorithm
from .algorithms.mmr import maximal_marginal_relevance
from .algorithms.nlg import natural_language_generation
from .algorithms.rouge import transform_candidate_summary
from .algorithms.rouge import transform_reference_summaries
from .algorithms.rouge import calculate_rouge, calculate_direct_rouge
from .utils.main_utils import save_object

import os

def evaluate(summary_type, corpus_name, corpus_type, individual, corpus_topics, corpus_pas, sentence_similarity_table, graph_docs, reference_summaries):
    # graph modification
    semantic_graph_modification(individual, graph_docs[corpus_name], corpus_pas[corpus_name])
    
    # graph-based ranking algorithm
    graph_algorithm = GraphAlgorithm(graph_docs[corpus_name], threshold=0.0001, dp=0.85, init=1.0, max_iter=100)
    graph_algorithm.run_algorithm()
    num_iter = graph_algorithm.get_num_iter()
    
    # maximal marginal relevance
    summary = maximal_marginal_relevance(summary_type, corpus_pas[corpus_name], graph_docs[corpus_name], num_iter, sentence_similarity_table[corpus_name])
    
    # natural language generation
    summary_paragraph = natural_language_generation(summary, corpus_pas[corpus_name])
    
    # transform for evaluation
    cand_summary = transform_candidate_summary(summary_paragraph)
    
    pwd = os.path.dirname(__file__)
    f = open(pwd + "/summary_performance_results/candidate_summaries/" + str(summary_type) + "/" + corpus_type + "/" + corpus_name + ".txt","w+")
    for sent in cand_summary:
        f.write(sent + "\r\n")
    f.close()

    ref_summaries = reference_summaries[corpus_type][corpus_name]
    

    res = {
        "rouge_2": calculate_direct_rouge([cand_summary], [ref_summaries]),
        "cand_summary": "",
        "ref_summary": ""
    }

    for sentence in cand_summary:
        res["cand_summary"] += sentence + " "
    res["cand_summary"].strip()

    for sentence in ref_summaries[0]:
        res["ref_summary"] += sentence + " "
    res["ref_summary"].strip()

    return res
