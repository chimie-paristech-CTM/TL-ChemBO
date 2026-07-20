from base.benchmarking import load_data, create_search_space, generate_FP
import pandas as pd
from baybe.utils.random import set_random_seed
from baybe.parameters import TaskParameter
from base.transfer_utils import _load_partial_data, _build_campaign, _filter_lookup, _unwrapped_simulate, _plot, _filter_campaign, select_substrates, _AUC, _final_yield, select_substrates_v2
import os
import hashlib
import json
import time

IMPUTE_MODE = 'ignore'
N_ITER_ALL = 30

def run_phase_1(dataset, fp_type, threshold, variable:str, accessible:list[str], switch_after=5, batch_size=1, n_iter=N_ITER_ALL, mc_runs=50, campaign_name='phase_1', individual=False, prior=None)->list[pd.DataFrame]:
    '''
    Use mc_runs to simulate different random seeds.
    
    Set random_seed outside so that accessible can be changed.
    '''
    
    if individual:
        if len(accessible) != 1:
            raise ValueError(f'With this setting, only one individual ligand/base.. at a time.')
    
    data = _load_partial_data(dataset=dataset, variable=variable, accessible=accessible)
    lookup = data['lookup']
    F_BEST = data['F_BEST']
    objective = data['objective']
    numerical_params = data['numerical_params']
    discrete_data = data['discrete_data']
    
    if individual:
        discrete_data[variable] = None

    PCA = threshold == "PCA"
    decorr_threshold = 0.0 if PCA else float(threshold)

    FP = generate_FP(discrete_data, fp_type=fp_type, PCA=PCA, decorr_threshold=decorr_threshold)
    
    searchspace = create_search_space(search_params_dict=FP, numeric_params_dict=numerical_params, task_param=None)
    
    feat_dim = len(searchspace.comp_rep_columns)

    campaign = _build_campaign(searchspace, objective, feat_dim, switch_after, prior)
    
    result = _unwrapped_simulate(campaign, lookup, init_data=None, batch_size=batch_size, n_iter=n_iter, mc_runs=mc_runs, campaign_name=campaign_name, impute_mode=IMPUTE_MODE, use_unwrapped_simul=True)
    
    recommendations = []
    for mc_run, group in result.groupby("Monte_Carlo_Run"):
        measured_list = group["Measured"].tolist()
        merged = pd.concat(measured_list)
        if individual:
            merged[variable] = accessible[0]
        recommendations.append(merged)
    
    return recommendations
    
    
def run_phase_2(dataset, fp_type, threshold, init_data:list[pd.DataFrame], variable:str, accessible:list[str], switch_after=0, batch_size=1, n_iter=N_ITER_ALL, campaign_name='phase_2', mc_runs=1, use_task_p=False, prior=None):
    '''
    0 - switch_after - task_p_iter - n_iter
    '''
    
    data = load_data(dataset=dataset)
    lookup = data['lookup']
    F_BEST = data['F_BEST']
    objective = data['objective']
    numerical_params = data['numerical_params']
    discrete_data = data['discrete_data']

    PCA = threshold == "PCA"
    decorr_threshold = 0.0 if PCA else float(threshold)

    FP = generate_FP(discrete_data, fp_type=fp_type, PCA=PCA, decorr_threshold=decorr_threshold)
    
    if use_task_p:
        task_param = TaskParameter(
                name='transfer_task',
                values=["test", "training"],
                active_values=["test"],
        )
        
        for df in init_data:
            df['transfer_task'] = 'training'
    else:
        task_param = None
    
    searchspace = create_search_space(search_params_dict=FP, numeric_params_dict=numerical_params, task_param=task_param)
    
    feat_dim = len(searchspace.comp_rep_columns) - len([p for p in searchspace.parameters if isinstance(p, TaskParameter)])

    campaign = _build_campaign(searchspace, objective, feat_dim, switch_after, prior)
    
    lookup = _filter_lookup(lookup, variable, accessible)
    campaign = _filter_campaign(campaign, variable, accessible)
    
    if use_task_p:
        lookup['transfer_task'] = 'test'
    
    result = _unwrapped_simulate(campaign, lookup, init_data=init_data, batch_size=batch_size, n_iter=n_iter, mc_runs=mc_runs, campaign_name=campaign_name, impute_mode=IMPUTE_MODE, use_unwrapped_simul=True)
    
    return result

def _collect_init_data_B(results_transfer_B:pd.DataFrame, init_data_A:list[pd.DataFrame])->list[pd.DataFrame]:
    '''
    No deepcopy
    '''
    
    init_data_B = []
    i = 0
    for sub_init_data, group in results_transfer_B.groupby("Initial_Data"): 
        # the order of Init_Data should correspond to the input init_data
        sub_measured_list_B = group["Measured"].tolist()
        sub_merged_B = pd.concat(sub_measured_list_B) # recommendations by campaign B with one group of init_data from campaign A.
        merged = pd.concat([init_data_A[i],sub_merged_B])
        init_data_B.append(merged)
        i += 1
        
    return init_data_B

def _merge_curve_B_and_C(results_transfer_B:pd.DataFrame, results_transfer_C:pd.DataFrame, task_p_iter:int):
    '''
    deepcopied.
    just for plot.
    '''
    df_B = results_transfer_B.copy(deep=True)
    df_C = results_transfer_C.copy(deep=True)
    
    df_C["Iteration"] = df_C["Iteration"] + task_p_iter
    df_C["Num_Experiments"] = df_C["Num_Experiments"] + task_p_iter
    
    df = pd.concat([df_B, df_C], ignore_index=True)
    
    df['Scenario'] = f'transfer_learning_B+C_{task_p_iter}'
    
    df = df.sort_values(
        ["Monte_Carlo_Run", "Initial_Data", "Iteration"]
    ).reset_index(drop=True)
    
    df["yield_CumBest"] = (
        df.groupby(["Monte_Carlo_Run", "Initial_Data"])["yield_IterBest"].cummax()
    )
    
    return df

def short_hash(x, n=8):
    s = json.dumps(x, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:n]

def _record(results:pd.DataFrame, init_data_A:list[pd.DataFrame], config):
    unique_id = f"{os.getpid()}_{int(time.time()*1000)}"
    
    sub_folder_name = f"{config['dataset']}_{config['fp']}_{config['variable']}_{config['mode']}_{config['n_train']}train{short_hash(config['train'])}_{config['n_test']}test{short_hash(config['test'])}_Seed{config['seed']}_uid{unique_id}"
    
    output_dir = os.path.join("output", sub_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config['unique_id'] = unique_id
    with open(os.path.join(output_dir, 'config.json'), "w") as f:
        json.dump(config, f, indent=2)
    
    # results.to_csv(os.path.join(output_dir, 'results.csv'))
    results.to_pickle(os.path.join(output_dir, 'results.pkl.gz'), compression='gzip')
    
    _plot(results, config['best_yield_test'], os.path.join(output_dir, 'plot.pdf'), config['best_yield_train'])
    
    INIT_DATA = []
    for i, init_data in enumerate(init_data_A):
        init_data['Initial_Data'] = i
        INIT_DATA.append(init_data)
        
    INIT_DATA = pd.concat(INIT_DATA, ignore_index=True)
    # INIT_DATA.to_csv(os.path.join(output_dir, 'init_data_A.csv'))
    INIT_DATA.to_pickle(os.path.join(output_dir, 'init_data_A.pkl.gz'), compression='gzip')
    
    return
    
def run(
    dataset = 'shields',
    variable = 'Ligand',
    mode = 'challenging',
    n_train = 3,
    n_test = 1,
    fp = 'chemberta_large',
    threshold = 0.7,
    N = 20,
    seed = 0,
    task_p_iter_list=[10],
    use_RandomSelect_v2=False,
    use_lab_style=False,
    prior=None,
):
    '''
    Campaign A: generates N groups of 30 recommendations.
    Campaign B: takes campaign A as init_data. Run 10 iters (task_p_iter=10).
    Campaign C: takes campaign A and B as init_data and uses TaskParameter. Run 20 iters.
    '''
    
    set_random_seed(seed)
    
    if use_RandomSelect_v2:
        train, best_yield_train, test, best_yield_test, variable, n_train, n_test = select_substrates_v2(dataset)
        print('Using entirely random experiment settings. Randomly sample: variable, n_train, n_test. Override input arguments: variable, n_train, n_test, mode.')
        mode = 'random_v2'
    else: # For challenging scenarios only
        # mode = 'challenging' is input value
        train, best_yield_train, test, best_yield_test = select_substrates(dataset, variable, mode, n_train, n_test)
        print('Generate challenging scenarios, with fixed source and randomly selected target (given n_test).')
        
    if use_lab_style:
        mode = mode + '_lab_style'
    
        print(f'Generate recommendation for each individual {variable} in the sampled training set.')
        init_data_indivs = []
        for val in train:
            init_data_indivs.append(run_phase_1(dataset, fp, threshold, variable, [val], n_iter=N_ITER_ALL, mc_runs=N, campaign_name=f'init_recom_A_indiv_{val}', individual=True, prior=prior))
        
        # merge them into normal init_data form: Nx30
        init_data_A = [pd.concat(dfs, ignore_index=True) for dfs in zip(*init_data_indivs)]
    else:
        print(f'Generating initial recommendations (Campaign A)...')
        init_data_A = run_phase_1(dataset, fp, threshold, variable, train, n_iter=N_ITER_ALL, mc_runs=N, campaign_name='init_recom_A', prior=prior)
    
    config = {
        'dataset':dataset,
        'variable':variable,
        'mode':mode,
        'n_train':n_train,
        'n_test':n_test,
        'fp':fp,
        'threshold':threshold,
        'N':N,
        'seed':seed,
        'task_p_iter':task_p_iter_list,
        'train':train,
        'best_yield_train':best_yield_train,
        'test':test,
        'best_yield_test':best_yield_test,
        'prior':prior,
    }
    
    results_B_C_dict = {}
    for task_p_iter in task_p_iter_list:
        # NOTE when init_data (not None) is given, set mc_runs to 1. Because init_data itself is generated and indexed via multiple MC runs, and MC runs can be equivalently employed.
        print(f'Initialize Campaign B with Campaign A, and run {task_p_iter} iterations...')
        results_transfer_B = run_phase_2(dataset, fp, threshold, init_data_A, variable, test, switch_after=0, batch_size=1, n_iter=task_p_iter, campaign_name=f'transfer_learning_B_{task_p_iter}', use_task_p=False, mc_runs=1, prior=prior)
        
        init_data_B = _collect_init_data_B(results_transfer_B, init_data_A)
        
        print(f'Initialize Campaign C with Campaign A+B, and run the rest {N_ITER_ALL-task_p_iter} iterations with TaskParameter.')
        results_transfer_C = run_phase_2(dataset, fp, threshold, init_data_B, variable, test, switch_after=0, batch_size=1, n_iter=N_ITER_ALL-task_p_iter, campaign_name=f'transfer_learning_C_{task_p_iter}', use_task_p=True, mc_runs=1, prior=prior)
        
        results_transfer = _merge_curve_B_and_C(results_transfer_B, results_transfer_C, task_p_iter)
        
        results_B_C_dict[task_p_iter] = results_transfer
    
    # Naive_Transfer, no TaskParamter
    print('Initialize with Campaign A, and do naive transfer learning without task parameter...')
    results_naive_transfer = run_phase_2(dataset, fp, threshold, init_data_A, variable, test, switch_after=0, batch_size=1, n_iter=N_ITER_ALL, campaign_name='naive_transfer', use_task_p=False, mc_runs=1, prior=prior)
    
    # Naive_Transfer, with TaskParamter
    print('Initialize with Campaign A, and do naive transfer learning with task parameter...')
    results_naive_transfer_taskP = run_phase_2(dataset, fp, threshold, init_data_A, variable, test, switch_after=0, batch_size=1, n_iter=N_ITER_ALL, campaign_name='naive_transfer_taskParam', use_task_p=True, mc_runs=1, prior=prior)
    
    # No_Transfer, just do 2 phases.
    print('No Transfer Learning. Baseline. 5 initial random samples.')
    results_no_transfer = run_phase_2(dataset, fp, threshold, None, variable, test, switch_after=5, batch_size=1, n_iter=N_ITER_ALL, campaign_name='no_transfer', use_task_p=False, mc_runs=N, prior=prior)
    
    results = [results_no_transfer, results_naive_transfer, results_naive_transfer_taskP]
    for task_p_iter, results_transfer in results_B_C_dict.items():
        results.append(results_transfer)
        
        config[f'AUC_transfer_B+C_{task_p_iter}'] = _AUC(results_transfer, F_BEST=best_yield_test)
        
        config[f'final_Y_transfer_B+C_{task_p_iter}'] = _final_yield(results_transfer)
        
    results = pd.concat(results, ignore_index=True)
    
    # NOTE AUC is locally normalized by best_yield_test here
    config['AUC_no_transfer'] = _AUC(results_no_transfer, F_BEST=best_yield_test)
    config['AUC_naive_transfer'] = _AUC(results_naive_transfer, F_BEST=best_yield_test)
    config['AUC_naive_transfer_TaskP'] = _AUC(results_naive_transfer_taskP, F_BEST=best_yield_test)
    # In long term, we could look into final yield.
    config['final_Y_no_transfer'] = _final_yield(results_no_transfer)
    config['final_Y_naive_transfer'] = _final_yield(results_naive_transfer)
    config['final_Y_naive_transfer_TaskP'] = _final_yield(results_naive_transfer_taskP)
    
    _record(results, init_data_A, config)
    
    return results

if __name__ == '__main__':
    import argparse
    
    def parse_threshold(x):
        try:
            return float(x)
        except ValueError:
            if x == "PCA":
                return x
            raise argparse.ArgumentTypeError(
                "threshold must be float or 'PCA'"
        )
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default='shields')
    parser.add_argument("--fp", type=str, default='t5-base-chem')
    parser.add_argument("--threshold", type=parse_threshold, default=0.7)
    parser.add_argument("--variable", type=str, default='Ligand')
    parser.add_argument("--mode", type=str, default='challenging', help='challenging or random')
    parser.add_argument("--n_train", type=int, default=3)
    parser.add_argument("--n_test", type=int, default=1)
    parser.add_argument("--task_p_iter_list", type=int, nargs="+", default=[10], help='list of IntType task_p_iters')
    parser.add_argument("--N", help='Number of MC_RUN/Init_Data', type=int, default=50)
    parser.add_argument("--seed", help='Random seed for (1)Selecting train/test; (2)RandomRecommender', type=int, default=0)
    parser.add_argument("--prior", type=str, default='adaptive_emilien')
    
    parser.add_argument("--use_RandomSelect_v2", action="store_true", help="Entirely randomized experiments: random variables, random n_train, random n_test. No challenging mode.")
    
    parser.add_argument("--use_lab_style", action="store_true", help="Use a lab-style transfer learning: (1) randomly select N_1 for training and N_2 for test. (2) run campaign A, individually, for each element in the training set. (3) collect all initial data and apply transfer learning.")
    
    args = parser.parse_args()
    
    run(dataset=args.dataset,variable=args.variable, mode=args.mode, n_train=args.n_train, n_test=args.n_test, fp=args.fp, threshold=args.threshold, N=args.N, seed=args.seed, task_p_iter_list=args.task_p_iter_list, use_RandomSelect_v2=args.use_RandomSelect_v2, use_lab_style=args.use_lab_style, prior=args.prior)