def get_real_j_val(i, j):
    return 1 + i + j

def get_idx_j_val(i, real_j):
    return real_j - i - 1

def fulfill_terms(value):
    return (value > 0 and value <= 0.5)

def semantic_graph_modification(individual, graph_docs, corpus_pas):
    for node1, node2, data in graph_docs.edges(data=True):
        weight_features_node1 = ((individual[0] * corpus_pas[node1].fst_feature) + (individual[1] * corpus_pas[node1].position_feature) +
            (individual[2] * corpus_pas[node1].p2p_feature) + (individual[3] * corpus_pas[node1].tfidf_feature)
            + (individual[4] * corpus_pas[node1].length_feature) + (individual[5] * corpus_pas[node1].pnoun_feature) + (individual[6] * corpus_pas[node1].num_feature) + 
            (individual[7] * corpus_pas[node1].noun_verb_feature) + (individual[8] * corpus_pas[node1].temporal_feature) + (individual[9] * corpus_pas[node2].location_feature))
        weight_features_node2 = ((individual[0] * corpus_pas[node2].fst_feature) + (individual[1] * corpus_pas[node2].position_feature) +
            (individual[2] * corpus_pas[node2].p2p_feature) + (individual[3] * corpus_pas[node2].tfidf_feature)
            + (individual[4] * corpus_pas[node2].length_feature) + (individual[5] * corpus_pas[node2].pnoun_feature) + (individual[6] * corpus_pas[node2].num_feature) + 
            (individual[7] * corpus_pas[node2].noun_verb_feature) + (individual[8] * corpus_pas[node2].temporal_feature) + (individual[9] * corpus_pas[node2].location_feature))
        data['weight'] = data['initial_weight'] * ((0.5 * weight_features_node1) + (0.5 * weight_features_node2))

    # fill sum weight
    for node in graph_docs.nodes:
        graph_docs.node[node]['sum_weight'] = sum(graph_docs[node][link]['weight'] for link in graph_docs[node])

class GraphAlgorithm:
    def __init__(self, graph, threshold=0.0001, dp=0.85, init=1.0, max_iter=100):
        self.__graph = graph
        self.__threshold = threshold
        self.__dp = dp
        self.__iteration = 0
        self.__init = init
        self.__max_iter = max_iter
        
    def init_graph(self):
        for node in self.__graph.nodes:
            self.__graph.node[node][self.__iteration] = self.__init

    def run_algorithm(self):
        keep_iteration = True
        self.__iteration = 0
        self.init_graph()
        
        for _ in range(self.__max_iter):
            self.__iteration += 1
            # print(self.__iteration)
            all_below_threshold = True
            for node in self.__graph.nodes:
                dp_multiplier = 0.0
                for neighbor in self.__graph[node]:
                    # neighbor's outgoing links
                    if (self.__graph.node[neighbor]['sum_weight'] > 0):
                        dp_multiplier += (self.__graph.node[neighbor][self.__iteration - 1] * self.__graph[node][neighbor]['weight'])/self.__graph.node[neighbor]['sum_weight']
                    else:
                        dp_multiplier += (self.__graph.node[neighbor][self.__iteration - 1] * self.__graph[node][neighbor]['weight'])
                self.__graph.node[node][self.__iteration] = (1 - self.__dp) + (self.__dp * dp_multiplier)
                # if (abs(self.__graph.node[node][self.__iteration] - self.__graph.node[node][self.__iteration - 1]) >= self.__threshold):
                #     all_below_threshold = False

            err = sum(abs(self.__graph.node[node][self.__iteration] - self.__graph.node[node][self.__iteration - 1]) for node in self.__graph.nodes) 
            if err < (len(self.__graph.nodes) * self.__threshold):
                break
                # keep_iteration = False
            # if (all_below_threshold):
            #    keep_iteration = False

    def get_num_iter(self):
        return self.__iteration
    
    def get_trained_graph(self):
        return self.__graph
