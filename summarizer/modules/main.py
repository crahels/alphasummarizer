from os import listdir
from .utils.main_utils import load_corpus
from .utils.main_utils import load_object
from .utils.main_utils import save_object
from .evaluation import evaluate
from .models.extracted_pas import ExtractedPAS
from .models.token import Token
from .models.pas import PAS
from .algorithms.rouge import transform_reference_summaries

import numpy as np
import os
from copy import deepcopy

def main_summarizer(summary_type, title, model):	
	corpus_topics = {}
	graph_docs = {}
	corpus_pas = {}
	sentence_similarity_table = {}
	corpus_summaries = {}
	reference_summaries_corpus = {}

	pwd = os.path.dirname(__file__)

	path = pwd + "/corpus/"

	root_dirs = [f for f in listdir(path)] # train, test, validation

	path = path + root_dirs[0] # docs, summaries_100, summaries_200
	types = [f for f in listdir(path)]

	for root_dir in [root_dirs[1]]:
		reference_summaries_corpus[root_dir] = []

		path = pwd + "/corpus/" + root_dir + "/" + types[0]
		topics = [f for f in listdir(path)]
		topics.sort()
		corpus_topics[root_dir] = topics

	load_corpus(title, summary_type, corpus_topics, corpus_pas, sentence_similarity_table, corpus_summaries, graph_docs)

	for root_dir in [root_dirs[1]]:
		reference_summaries_corpus[root_dir] = {}
		corpus_name = title
		reference_summaries = transform_reference_summaries(corpus_summaries[corpus_name])
		for idx, summary in enumerate(reference_summaries):
			f = open(pwd + "/summary_performance_results/reference_summaries/" + str(summary_type) + "/" + root_dir + "/" + corpus_name + "." + str(idx + 1) + ".txt","w+")
			for sent in summary:
				f.write(sent + "\r\n")
			f.close()

		reference_summaries_corpus[root_dir][corpus_name] = reference_summaries

	individual = []
	if (model == "Khan-10"):
		individual = [0.21308642, 0.7118985, 0.41313752, 0.475472, 0.134772, 0.351493, 0.24141418, 0.614672, 0.18693995, 0.0]
	elif (model == "Khan-4"):
		individual = [0.21308642, 0.7118985, 0.41313752, 0.475472, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	elif (model == "best-model-10"):
		if (summary_type == 100):
			individual = [0.059797193827282845, 0.7445171231828352, 0.1057383469923665, 0.005837304227753614, 0.25836691986474397, 0.6773102114241532, 0.9570207563640649, 0.01374495963389121, 0.47316705249985086, 0.569996650187908]
		elif (summary_type == 200):
			individual = [0.5319917067830644, 0.3409998607382585, 0.16722992857812402, 0.0016241363144431364, 0.0977777840511711, 0.13225443521982905, 0.9844373500577952, 0.036347091719314384, 0.6996785982237446, 0.13538144706184485]
	elif (model == "best-model-4"):
		if (summary_type == 100):
			individual = [0.059797193827282845, 0.7445171231828352, 0.1057383469923665, 0.005837304227753614, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		elif (summary_type == 200):
			individual = [0.5319917067830644, 0.3409998607382585, 0.16722992857812402, 0.0016241363144431364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
	elif (model == "default-10"):
		individual = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
	elif (model == "default-4"):
		individual = [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

	result = evaluate(summary_type, title, "test", individual, corpus_topics, corpus_pas, sentence_similarity_table, graph_docs, reference_summaries_corpus)

	return result
