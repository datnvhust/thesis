import pickle
import json
import operator
import numpy as np
from scipy import optimize
from datetime import datetime
import unidecode
from datasets import DATASET


def combine_rank_scores(coeffs, *rank_scores):
    """Combining the rank score of different algorithms"""
    
    final_score = []
    for scores in zip(*rank_scores):
        combined_score = coeffs @ np.array(scores)
        final_score.append(combined_score)
        
    return final_score
count = 0
def cost(coeffs, src_files, bug_reports, *rank_scores):
    """The cost function to be minimized"""
    global count
    count = count + 1
    final_scores = combine_rank_scores(coeffs, *rank_scores)
    mrr = []
    mean_avgp = []
    
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
    out = -1 * (np.mean(mrr) + np.mean(mean_avgp))
    # print(sum(sum(final_scores)) /len(final_scores) /len(final_scores[0]), np.mean(final_scores))
    # out = -1 * (np.mean(mrr) + np.mean(mean_avgp) + np.mean(final_scores)) 
    # print(coeffs, out)
    return out
gen_count = 0
def callback(a, convergence = 0):
    global gen_count
    gen_count = gen_count + 1
    print("out", a, gen_count)

def estiamte_params(src_files, bug_reports, *rank_scores):
    """Estimating linear combination parameters"""
    res = optimize.differential_evolution(
        cost, bounds=[(0, 1)] * len(rank_scores),
        args=(src_files, bug_reports, *rank_scores),
        callback=callback,
        popsize=15,
        mutation=(0, 1), 
        tol=0,
        disp=True,
        maxiter=300,
        # recombination=0.7,
        # init="random",
        # updating='deferred',
        # workers=1,
        strategy='randtobest1exp', polish=True, seed=458711526
    )

    print(count)
    return res.x.tolist()
    

def evaluate(src_files, bug_reports, coeffs, bug_has_stack_trace, *rank_scores):
    
    final_scores = combine_rank_scores(coeffs, *rank_scores)
    
    # Writer for the output file
    result_file = open('output.csv', 'w')

    top_n = (1, 5, 10)
    top_n_rank = [0] * len(top_n)
    mrr = []
    mean_avgp = []
    
    # precision_at_n = [[] for _ in top_n]
    # recall_at_n = [[] for _ in top_n]
    # f_measure_at_n = [[] for _ in top_n]
    
    for i, (bug_id, report) in enumerate(bug_reports.items()):
        # print(bug_id, report.description)
        # Finding source codes from the simis indices
        src_ranks, _ = zip(*sorted(zip(src_files.keys(), final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))
        # print(src_ranks)
        # Getting reported fixed files
        fixed_files = report.fixed_files
        # print(fixed_files)
        
        # Iterating over top n
        for k, rank in enumerate(top_n):
            # print(k, rank)
            hit = set(src_ranks[:rank]) & set(fixed_files)
            # print(hit)
            # Computing top n rank
            if hit:
                top_n_rank[k] += 1
                
            # # Computing precision and recall at n
            # if not hit:
            #     precision_at_n[k].append(0)
            # else:
            #     precision_at_n[k].append(len(hit) / len(src_ranks[:rank]))
            # recall_at_n[k].append(len(hit) / len(fixed_files))
            # if not (precision_at_n[k][i] + recall_at_n[k][i]):
            #     f_measure_at_n[k].append(0)
            # else:
            #     f_measure_at_n[k].append(2 * (precision_at_n[k][i] * recall_at_n[k][i])
            #                              / (precision_at_n[k][i] + recall_at_n[k][i]))
        
        # Getting the ranks of reported fixed files
        relevant_ranks = sorted(src_ranks.index(fixed) + 1
                                for fixed in fixed_files)
        # MRR
        min_rank = relevant_ranks[0]
        mrr.append(1 / min_rank)
        
        # MAP
        mean_avgp.append(np.mean([len(relevant_ranks[:j + 1]) / rank
                                   for j, rank in enumerate(relevant_ranks)]))
        # if bug_id in bug_has_stack_trace:
        print(i, bug_id, relevant_ranks, bug_id in bug_has_stack_trace)
        
        result_file.write(bug_id + ',' + ','.join(src_ranks) + '\n')
        
    result_file.close()
    
    return (top_n_rank, [x / len(bug_reports) for x in top_n_rank],
            np.mean(mrr), np.mean(mean_avgp),
            # np.mean(precision_at_n, axis=1).tolist(), np.mean(recall_at_n, axis=1).tolist(),
            # np.mean(f_measure_at_n, axis=1).tolist()
            )


def main():
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
    with open(DATASET.root / 'bug_has_stack_trace.json', 'r') as file:
        bug_has_stack_trace = json.load(file)
    with open(DATASET.root / 'semantic_similarity.json', 'r') as file:
        semantic_similarity_score = json.load(file)
    with open(DATASET.root / 'fixed_bug_reports.json', 'r') as file:
        fixed_bug_reports_score = json.load(file)
    time1 = datetime.now()
    sc = []
    for s_id, s in src_files.items():
        sc.append(s_id)
    with open(DATASET.root / 'src_id.json', 'w') as file:
        json.dump(sc, file)
    print(time1)
    params = [0.6078854662743608, 0.0019523903677798327, 0.22685018595673528, 0.5255269450784907, 0.9791843571062989] # aspectj
    # params = [0.07185275472918345, 0.039487968405805374, 0.04116174058460431, 0.4511423324034731, 0.7475546850987008] # zxing
    print('params', params)
    # with open(DATASET.root / 'params.json', 'w') as file:
    #     json.dump(params, file)
    # x = cost(params, src_files, bug_reports,
    #                    vsm_similarity_score, token_matching_score,
    #                    fixed_bug_reports_score, semantic_similarity_score,
    #                    stack_trace_score)
    results = evaluate(src_files, bug_reports, params, bug_has_stack_trace,
                       vsm_similarity_score, token_matching_score,
                       fixed_bug_reports_score, semantic_similarity_score,
                       stack_trace_score)
    # with open(DATASET.root / 'evaluation.json', 'w') as file:
    #     json.dump(results, file)
    print('Top N Rank:', results[0])
    print('Top N Rank %:', results[1])
    print('MRR:', results[2])
    print('MAP:', results[3])
    print(datetime.now() - time1)

# # Uncomment these for precision, recall, and f-measure results
#     print('Precision@N:', results[4])
#     print('Recall@N:', results[5])
#     print('F-measure@N:', results[6])


if __name__ == '__main__':
    main()
