import stanfordnlp
from utils.pas_utils import load_pas_corpus
from utils.main_utils import initialize_stanford_nlp
from utils.main_utils import save_object
from pas_extraction.prediction import predict_pas
from pas_extraction.performance_measure import calculate_pas_performance

def main():
	nlp = initialize_stanford_nlp()

	corpus_docs = {}
	corpus_pas = {}
	
	pas_prediction = {}
	pas_actual = {}

	precision = {}
	recall = {}
	fmeasure = {}

	corpus_test = ["f-bunuh-diri", "all-case"]
	for corpus_name in corpus_test:
		corpus_docs[corpus_name] = {
			"sentences": []
		}
		corpus_pas[corpus_name] = []
		pas_prediction[corpus_name] = []
		pas_actual[corpus_name] = []
	
	load_pas_corpus(nlp, corpus_test, pas_actual, corpus_docs)
	predict_pas(corpus_test, corpus_pas, corpus_docs, pas_prediction)
	for corpus_name in corpus_test:
		print(corpus_name)
		precision[corpus_name], recall[corpus_name] = calculate_pas_performance(pas_actual[corpus_name], pas_prediction[corpus_name])
		fmeasure[corpus_name] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		for idx in range(len(precision[corpus_name])):
			fmeasure[corpus_name][idx] = float(2 * precision[corpus_name][idx] * recall[corpus_name][idx])/float(precision[corpus_name][idx] + recall[corpus_name][idx])

		save_object("pas_performance_results/precision_" + corpus_name, precision[corpus_name])
		save_object("pas_performance_results/recall_" + corpus_name, recall[corpus_name])
		save_object("pas_performance_results/fmeasure_" + corpus_name, fmeasure[corpus_name])

		print("precision (SUBJ, VERB, OBJ, TMP, LOC, ADJ):", precision[corpus_name])
		print("average precision:", (precision[corpus_name][0] + precision[corpus_name][1] + precision[corpus_name][2] + precision[corpus_name][3] + precision[corpus_name][4] + precision[corpus_name][5])/6)
		print("recall (SUBJ, VERB, OBJ, TMP, LOC, ADJ):", recall[corpus_name])
		print("average recall:", (recall[corpus_name][0] + recall[corpus_name][1] + recall[corpus_name][2] + recall[corpus_name][3] + recall[corpus_name][4] + recall[corpus_name][5])/6)
		print("f-measure (SUBJ, VERB, OBJ, TMP, LOC, ADJ):", fmeasure[corpus_name])
		print("average f-measure:", (fmeasure[corpus_name][0] + fmeasure[corpus_name][1] + fmeasure[corpus_name][2] + fmeasure[corpus_name][3] + fmeasure[corpus_name][4] + fmeasure[corpus_name][5])/6)

if __name__ == '__main__':
	main()
