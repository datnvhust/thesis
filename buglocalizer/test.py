import GA
import numpy
from datasets import DATASET
import json
import pickle
import numpy as np
import operator
from datetime import datetime
# import pygad

with open(DATASET.root / 'preprocessed_src.pickle', 'rb') as file:
    src_files = pickle.load(file)
with open(DATASET.root / 'preprocessed_reports.pickle', 'rb') as file:
    bug_reports = pickle.load(file)

with open(DATASET.root / 'token_matching.json', 'r') as file:
    token_matching_score = json.load(file)  
with open(DATASET.root / 'vsm_similarity.json', 'r') as file:
    vsm_similarity_score = json.load(file)
with open(DATASET.root / 'stack_trace.json', 'r') as file:
    stack_trace_score = json.load(file)
with open(DATASET.root / 'semantic_similarity.json', 'r') as file:
    semantic_similarity_score = json.load(file)
with open(DATASET.root / 'fixed_bug_reports.json', 'r') as file:
    fixed_bug_reports_score = json.load(file)

def combine_rank_scores(coeffs, *rank_scores):
    """Combining the rank score of different algorithms"""
    
    final_score = []
    for scores in zip(*rank_scores):
        combined_score = (coeffs) @ (np.array(scores))
        final_score.append(combined_score)
        
    return final_score

def cost(coeffs, i):
    """The cost function to be minimized"""
    # coeffs = [max(x, 0) for x in coeffs]
    # s = sum(coeffs)
    # coeffs = [x/s for x in coeffs]
    print(coeffs)
    final_scores = combine_rank_scores(coeffs, vsm_similarity_score, token_matching_score,
                             fixed_bug_reports_score, semantic_similarity_score,
                             stack_trace_score)
    mrr = []
    mean_avgp = []
    # print(len(final_scores))
    # print(len(final_scores[0]))
    # print(sum(sum(final_scores)) /len(final_scores) /len(final_scores[0]))
    
    for i, report in enumerate(bug_reports.items()):
        
        # Finding source files from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))
        
        # Getting reported fixed files
        fixed_files = report[1].fixed_files
        
        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files)
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)
        
        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                   for j, rank in enumerate(relevant_ranks)]))
    # print(datetime.now() - now)
    # out = np.mean(mrr) + np.mean(mean_avgp)
    # print(coeffs, out)
    # return np.mean(mrr) + np.mean(mean_avgp) + sum(sum(final_scores)) /len(final_scores) /len(final_scores[0])
    return np.mean(mrr) + np.mean(mean_avgp)
    # return np.mean(mean_avgp)
def evaluate(coeffs):
    
    final_scores = combine_rank_scores(coeffs, vsm_similarity_score, token_matching_score,
                             fixed_bug_reports_score, semantic_similarity_score,
                             stack_trace_score)
    
    # Writer for the output file
    result_file = open('output.csv', 'w')

    top_n = (1, 5, 10)
    top_n_rank = [0] * len(top_n)
    mrr = []
    mean_avgp = []
    
    precision_at_n = [[] for _ in top_n]
    recall_at_n = [[] for _ in top_n]
    f_measure_at_n = [[] for _ in top_n]
    
    for i, (bug_id, report) in enumerate(bug_reports.items()):
        
        # Finding source codes from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))
        
        # Getting reported fixed files
        fixed_files = report.fixed_files
        
        # Iterating over top n
        for k, rank in enumerate(top_n):
            hit = set(src_ranks[:rank]) & set(fixed_files)
                
            # Computing top n rank
            if hit:
                top_n_rank[k] += 1
                
            # Computing precision and recall at n
            if not hit:
                precision_at_n[k].append(0)
            else:
                precision_at_n[k].append(len(hit) / len(src_ranks[:rank]))
            recall_at_n[k].append(len(hit) / len(fixed_files))
            if not (precision_at_n[k][i] + recall_at_n[k][i]):
                f_measure_at_n[k].append(0)
            else:
                f_measure_at_n[k].append(2 * (precision_at_n[k][i] * recall_at_n[k][i])
                                         / (precision_at_n[k][i] + recall_at_n[k][i]))
        
        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files)
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)
        
        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                   for j, rank in enumerate(relevant_ranks)]))
        
        result_file.write(bug_id + ',' + ','.join(src_ranks) + '\n')
        
    result_file.close()
    
    return (top_n_rank, [x / len(bug_reports) for x in top_n_rank],
            np.mean(mrr), np.mean(mean_avgp),
            np.mean(precision_at_n, axis=1).tolist(), np.mean(recall_at_n, axis=1).tolist(),
            np.mean(f_measure_at_n, axis=1).tolist())
# print(evaluate([0.14719488, 0.13018921, 0.08035653, 0.9556664,  6.26271194]))

num_generations = 100 # Number of generations.
num_parents_mating = 7 # Number of solutions to be selected as parents in the mating pool.

sol_per_pop = 75 # Number of solutions in the population.
num_genes = 5

init_range_low = 0
init_range_high = 1

parent_selection_type = "sss" # Type of parent selection.
keep_parents = 7 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.

crossover_type = "single_point" # Type of the crossover operator.

# Parameters of the mutation operation.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists or when mutation_type is None.

last_fitness = 0
def callback_generation(ga_instance):
    global last_fitness
    fitness = ga_instance.best_solution()[1]
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=fitness))
    print("Change     = {change}".format(change=fitness - last_fitness))
    last_fitness = fitness

# Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
time1 = datetime.now()
print(time1)
ga_instance = GA.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating, 
                       fitness_func=cost,
                       sol_per_pop=sol_per_pop, 
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       mutation_by_replacement=True,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_probability=0.1,
                       callback_generation=callback_generation)

# Running the GA to optimize the parameters of the function.
ga_instance.run()

# After the generations complete, some plots are showed that summarize the how the outputs/fitenss values evolve over generations.
ga_instance.plot_result()

# Returning the details of the best solution.
print(datetime.now() - time1)
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))
print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))

solution = solution / sum(solution)
prediction = evaluate(solution)
print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

if ga_instance.best_solution_generation != -1:
    print("Best fitness value reached after {best_solution_generation} generations.".format(best_solution_generation=ga_instance.best_solution_generation))

# Saving the GA instance.
filename = 'genetic' # The filename to which the instance is saved. The name is without extension.
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = GA.load(filename=filename)
loaded_ga_instance.plot_result()