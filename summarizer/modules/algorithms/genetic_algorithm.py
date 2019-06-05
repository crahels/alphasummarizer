import random, math
from copy import deepcopy
from pyeasyga import pyeasyga_modified
from operator import attrgetter
from algorithms.graph import semantic_graph_modification
from algorithms.graph import GraphAlgorithm
from algorithms.mmr import maximal_marginal_relevance
from algorithms.nlg import natural_language_generation
from algorithms.rouge import transform_candidate_summary
from algorithms.rouge import transform_reference_summaries
from algorithms.rouge import calculate_rouge
from utils.main_utils import save_object

generations = 100
num_population = 50

TOURNAMENT = 0
ROULETTE_WHEEL = 1
SUS = 2
RANK_WHEEL = 3
RANK_SUS = 4

DEFAULT_CROSSOVER = 0
BLX = 1
LINEAR = 2

DEFAULT_MUTATION = 0
NONUNIFORM = 1

# define and set function to create a candidate solution representation
def create_individual(data):
    return [random.uniform(0, 1) for _ in range(data["num_features"])]

def normalize_element(elmt):
    if elmt > 1.0:
        return 1.0
    elif elmt < 0.0:
        return 0.0
    else:
        return elmt

# define and set the GA's crossover operation using BLX-0.5 algorithm
def blx_crossover(parent_1, parent_2, data):
    child = []
    for i in range(len(parent_1)):
        c_max = max(parent_1[i], parent_2[i])
        c_min = min(parent_1[i], parent_2[i])
        I = c_max - c_min

        lower_bound = c_min - (I * 0.5)
        lower_bound = normalize_element(lower_bound)

        upper_bound = c_max + (I * 0.5)
        upper_bound = normalize_element(upper_bound)

        child_element = random.uniform(lower_bound, upper_bound)

        child.append(child_element)

    return child

def linear_crossover(parent_1, parent_2, data):
    child_1 = []
    child_2 = []
    child_3 = []
    for i in range(len(parent_1)):
        elmt = (0.5 * parent_1[i]) + (0.5 * parent_2[i])
        child_1.append(normalize_element(elmt))
        elmt = (1.5 * parent_1[i]) - (0.5 * parent_2[i])
        child_2.append(normalize_element(elmt))
        elmt = (-0.5 * parent_1[i]) + (1.5 * parent_2[i])
        child_3.append(normalize_element(elmt))
    
    fitness_score = [
        {
            "individual": child_1,
            "fitness_score": fitness(child_1, data)
        },
        {
            "individual": child_2,
            "fitness_score": fitness(child_2, data)
        },
        {
            "individual": child_3,
            "fitness_score": fitness(child_3, data)
        }
    ]
    
    fitness_score.sort(key = lambda x: x["fitness_score"], reverse = True)
    return fitness_score[0]["individual"], fitness_score[1]["individual"] 

# define and set the GA's mutation operation using non-uniform mutation algorithm
def nonuniform_mutation(individual, generation_index, mutation_probability):
    for i in range(len(individual)):
        can_mutate = random.random() < mutation_probability
        if can_mutate:
            mutate_index = i
            tau = random.randint(0, 1) 
            b = 5 # research about this value
            upper_bound = 1.0
            lower_bound = 0.0

            y_multiplier = 1 - math.pow(random.uniform(0, 1), math.pow((1 - (float(generation_index + 1)/generations)), b))
            y = None
            if (tau == 0):
                individual[mutate_index] = individual[mutate_index] + ((upper_bound - individual[mutate_index]) * y_multiplier)
                individual[mutate_index] = normalize_element(individual[mutate_index])
            else:
                individual[mutate_index] = individual[mutate_index] - ((individual[mutate_index] - lower_bound) * y_multiplier)
                individual[mutate_index] = normalize_element(individual[mutate_index])

# define and set the GA's selection operation
def stochastic_universal_sampling_selection(population, crossover_method):
    # population.sort(key=attrgetter('fitness'), reverse=True)
    total_fitness = float(sum(individual.fitness for individual in population))
    temp_population = deepcopy(population)
    for idx in range(len(temp_population)):
        temp_population[idx].fitness /= total_fitness

    scaled_fitness = 1.0
    num_selection = num_population # replace this
    if crossover_method == BLX:
        num_selection = num_population * 2

    range_pointer = scaled_fitness/num_selection
    start = random.uniform(0, range_pointer)
    pointers = [start + (i * range_pointer) for i in range(num_selection)]
    chosen_population = []
    for pointer in pointers:
        i = 0
        while sum(individual.fitness for individual in temp_population[:(i + 1)]) < pointer:
            i += 1
        chosen_population.append(population[i])
    return chosen_population

def rank_stochastic_selection(population, crossover_method):
    population.sort(key=attrgetter('fitness'), reverse=True)
    temp_population = deepcopy(population)
    for idx in range(len(temp_population)):
        temp_population[idx].fitness = float(2 * (len(temp_population) - idx))/float(len(temp_population) * (len(temp_population) + 1))

    scaled_fitness = 1.0
    num_selection = num_population # replace this
    if crossover_method == BLX:
        num_selection = num_population * 2

    range_pointer = scaled_fitness/num_selection
    start = random.uniform(0, range_pointer)
    pointers = [start + (i * range_pointer) for i in range(num_selection)]
    chosen_population = []
    for pointer in pointers:
        i = 0
        while sum(individual.fitness for individual in temp_population[:(i + 1)]) < pointer:
            i += 1
        chosen_population.append(population[i])
    return chosen_population

def rank_wheel_selection(population, crossover_method):
    population.sort(key=attrgetter('fitness'), reverse=True)
    temp_population = deepcopy(population)
    for idx in range(len(temp_population)):
        temp_population[idx].fitness = float(2 * (len(temp_population) - idx))/float(len(temp_population) * (len(temp_population) + 1))

    num_selection = num_population # replace this
    if crossover_method == BLX:
        num_selection = num_population * 2

    random_numbers = [random.uniform(0, 1) for i in range(0, num_selection)]
    chosen_population = []
    for random_number in random_numbers:
        partial_sum = 0.0
        i = 0
        while random_number > partial_sum:
            partial_sum += temp_population[i].fitness
            i += 1
        chosen_population.append(population[i - 1])
    return chosen_population

def roulette_wheel_selection(population, crossover_method):
    total_fitness = float(sum(individual.fitness for individual in population))
    temp_population = deepcopy(population)
    for idx in range(len(temp_population)):
        temp_population[idx].fitness /= total_fitness

    num_selection = num_population # replace this
    if crossover_method == BLX:
        num_selection = num_population * 2

    random_numbers = [random.uniform(0, 1) for i in range(0, num_selection)]
    chosen_population = []
    for random_number in random_numbers:
        partial_sum = 0.0
        i = 0
        while random_number > partial_sum:
            partial_sum += temp_population[i].fitness
            i += 1
        chosen_population.append(population[i - 1])
    return chosen_population

# define a fitness function
def fitness(individual, data):
    # candidate_summary_100 = []
    # candidate_summary_200 = []
    for corpus_name in (data["corpus_topics"]["train"]):
        # print(corpus_name)
        semantic_graph_modification(individual, data["graph_docs"][corpus_name], data["corpus_pas"][corpus_name])
        
        # graph-based ranking algorithm
        graph_algorithm = GraphAlgorithm(data["graph_docs"][corpus_name], threshold=0.0001, dp=0.85, init=1.0, max_iter=100)
        graph_algorithm.run_algorithm()
        num_iter = graph_algorithm.get_num_iter()
        
        # maximal marginal relevance 100
        summary = maximal_marginal_relevance(100, data["corpus_pas"][corpus_name], data["graph_docs"][corpus_name], num_iter, data["sentence_similarity_table"][corpus_name])
        
        # natural language generation
        summary_paragraph = natural_language_generation(summary, data["corpus_pas"][corpus_name])
        
        # transform for evaluation
        cand_summary_100 = transform_candidate_summary(summary_paragraph)
        # candidate_summary_100.append(cand_summary_100)
        f = open(data["folder"] + "/candidate_summaries/100/train/" + corpus_name + ".txt","w+")
        for sent in cand_summary_100:
            f.write(sent + "\r\n")
        f.close()

        # # maximal marginal relevance 200
        # summary = maximal_marginal_relevance(200, data["corpus_pas"][corpus_name], data["graph_docs"][corpus_name], num_iter, data["sentence_similarity_table"][corpus_name])
        
        # # natural language generation
        # summary_paragraph = natural_language_generation(summary, data["corpus_pas"][corpus_name])
        
        # # transform for evaluation
        # cand_summary_200 = transform_candidate_summary(summary_paragraph)

        # f = open(data["folder"] + "/candidate_summaries/200/train/" + corpus_name + ".txt","w+")
        # for sent in cand_summary_200:
        #     f.write(sent + "\r\n")
        # f.close()

    rouge_100 = float(calculate_rouge(data["folder"], "100/train"))
    # rouge_200 = float(calculate_rouge(data["folder"], "200/train"))

    return rouge_100
    # return 0.5 * (rouge_100 + rouge_200)

class GeneticAlgorithm:
    def __init__(self, seed_data, population_size=50, generations=100, maximise_fitness=True, 
                 crossover_method=DEFAULT_CROSSOVER, selection_method=TOURNAMENT, mutation_method=DEFAULT_MUTATION,
                 crossover_probability=0.8, mutation_probability=0.1):
        self.__seed_data = seed_data
        self.__population_size = population_size
        self.__generations = generations
        self.__maximise_fitness = maximise_fitness
        self.__crossover_method = crossover_method
        self.__selection_method = selection_method
        self.__mutation_method = mutation_method
        self.__crossover_probability = crossover_probability
        self.__mutation_probability = mutation_probability

        self.__ga = pyeasyga_modified.GeneticAlgorithm(self.__seed_data, 
                                      population_size=self.__population_size, 
                                      generations=self.__generations, 
                                      crossover_probability=self.__crossover_probability,
                                      mutation_probability=self.__mutation_probability,
                                      maximise_fitness=self.__maximise_fitness, 
                                      crossover_method=self.__crossover_method,
                                      selection_method=self.__selection_method)

        self.__ga.create_individual = create_individual
        if (self.__crossover_method == BLX):
            self.__ga.crossover_function = blx_crossover
        elif (self.__crossover_method == LINEAR):
            self.__ga.crossover_function = linear_crossover

        if (self.__selection_method == ROULETTE_WHEEL):
            self.__ga.selection_function = roulette_wheel_selection
        elif (self.__selection_method == SUS):
            self.__ga.selection_function = stochastic_universal_sampling_selection
        elif (self.__selection_method == RANK_WHEEL):
            self.__ga.selection_function = rank_wheel_selection
        elif (self.__selection_method == RANK_SUS):
            self.__ga.selection_function = rank_stochastic_selection

        if (self.__mutation_method == NONUNIFORM):
            self.__ga.mutate_function = nonuniform_mutation

        self.__ga.fitness_function = fitness # set the GA's fitness function

    def best_individual_and_score(self):
        return self.__ga.best_individual()[1], self.__ga.best_individual()[0]

    def run_algorithm(self):
        self.__ga.run() # run the GA
        print("fitness score:", self.__ga.best_individual()[0])
        print("best individual:", self.__ga.best_individual()[1]) # print the GA's best solution

        # save_object("../../temporary_data/fitness_score", self.__ga.best_individual()[0])
        # save_object("../../temporary_data/best_individual", self.__ga.best_individual()[1])
