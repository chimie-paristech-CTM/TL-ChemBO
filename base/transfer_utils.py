import pandas as pd
from base.benchmarking import _normalize
import numpy as np
from baybe.parameters import TaskParameter, NumericalDiscreteParameter
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe import Campaign
from baybe.recommenders import (
    BotorchRecommender,
    RandomRecommender,
    SequentialMetaRecommender,
)
from baybe.objectives import SingleTargetObjective
from base.kernels import MaternKernelFactory
from baybe.simulation import simulate_scenarios
from base.simulation import simulate_scenarios_in_house
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from baybe.targets import NumericalTarget
from typing import Literal
import random
import warnings

def _random_sample(l:list, n:int)->list:
    results = random.sample(l, n)
    print(f'Random Sampled: {results}')
    return results

def _list_subs(A:list,B:list):
    '''
    list A-B
    '''
    result = []
    for e in A:
        if not e in B:
            result.append(e)
            
    return result

def select_substrates(
    dataset:Literal['shields','buchwald_hartwig']='shields',
    variable:Literal['Ligand','Base','Solvent','aryl_halide_smiles','base_smiles','ligand_smiles','additive_smiles']='Ligand',
    mode:Literal['challenging','random']='challenging',
    n_train:int=3,
    n_test:int=1,
):
    '''
    Set random seed before calling this function!
    '''
    
    LIGAND_SHIELDS = ['(t-Bu)PhCPhos', '1268824-69-6', 'BrettPhos', 'Di-tert-butylphenylphosphine', 'JackiePhos', 'Me2PPh', 'Methyldiphenylphosphine', 'P(2-furyl)3', 'PPh3', 'SCHEMBL15068049', 'Tricyclohexylphosphine', 'XPhos']
    BASE_SHIELDS = ['Cesium acetate', 'Cesium pivalate', 'Potassium acetate', 'Potassium pivalate']
    SOLVENT_SHIELDS = ['Butyl Ester', 'Butyornitrile', 'DMAc', 'p-Xylene']
    
    ARYL_HALIDE_BUCHWALD = ['BrC1=CC=C(C(F)(F)F)C=C1', 'BrC1=CC=C(CC)C=C1', 'BrC1=CC=C(OC)C=C1', 'BrC1=CN=CC=C1', 'BrC1=NC=CC=C1', 'ClC1=CC=C(C(F)(F)F)C=C1', 'ClC1=CC=C(CC)C=C1', 'ClC1=CC=C(OC)C=C1', 'ClC1=CN=CC=C1', 'ClC1=NC=CC=C1', 'IC1=CC=C(C(F)(F)F)C=C1', 'IC1=CC=C(CC)C=C1', 'IC1=CC=C(OC)C=C1', 'IC1=CN=CC=C1', 'IC1=NC=CC=C1']
    ADDITIVE_BUCHWALD = ['C1(C2=CC=CC=C2)=CC=NO1', 'C1(C2=CC=CC=C2)=CON=C1', 'C1(C2=CC=CC=C2)=NOC=C1', 'C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=CC=NO1', 'C1(N(CC2=CC=CC=C2)CC3=CC=CC=C3)=NOC=C1', 'C12=C(C=CC=C2)ON=C1', 'C12=CON=C1C=CC=C2', 'CC1=C(C(OCC)=O)C=NO1', 'CC1=CC(C(OCC)=O)=NO1', 'CC1=CC(C)=NO1', 'CC1=CC(N2C=CC=C2)=NO1', 'CC1=CC=NO1', 'CC1=NOC(C(OCC)=O)=C1', 'CC1=NOC(C2=CC=CC=C2)=C1', 'CC1=NOC=C1', 'CCOC(C1=CON=C1)=O', 'CCOC(C1=NOC=C1)=O', 'COC1=NOC(C(OCC)=O)=C1', 'FC(C=CC=C1F)=C1C2=CC=NO2', 'O=C(OC)C1=CC=NO1', 'O=C(OC)C1=NOC(C2=CC=CO2)=C1', 'O=C(OC)C1=NOC(C2=CC=CS2)=C1']
    BASE_BUCHWALD = ['CC(C)(C)/N=C(N(C)C)/N(C)C', 'CN(C)P(N(C)C)(N(C)C)=NP(N(C)C)(N(C)C)=NCC', 'CN1CCCN2C1=NCCC2']
    LIGAND_BUCHWALD = ['CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C(OC)=CC=C2OC', 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C(C)(C)C)C(C)(C)C)C=CC=C2', 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P(C3CCCCC3)C4CCCCC4)C=CC=C2', 'CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=C(P([C@@]3(C[C@@H]4C5)C[C@H](C4)C[C@H]5C3)[C@]6(C7)C[C@@H](C[C@@H]7C8)C[C@@H]8C6)C(OC)=CC=C2OC']
    
    from base.global_config import FILEPATH_DICT
    
    file_path = FILEPATH_DICT[dataset]

    if file_path.endswith('.xlsx'):
        lookup = pd.read_excel(file_path, index_col=0 if dataset in ['shields'] else None)
    elif file_path.endswith('.csv'):
        lookup = pd.read_csv(file_path, index_col=0 if dataset in ['shields'] else None)
    else:
        pass
        
    choices = lookup[variable].unique().tolist()
    n_max = len(choices)
    if n_train + n_test > n_max:
        raise ValueError(f'n_train={n_train} and n_test={n_test}, while total number of choices is {n_max}')
    if dataset=='shields' and mode=='challenging' and variable=='Ligand':
        train = ['BrettPhos', '(t-Bu)PhCPhos', 'P(2-furyl)3']
    elif dataset=='buchwald_hartwig' and mode=='challenging' and variable=='aryl_halide_smiles':
        train = ['ClC1=CC=C(C(F)(F)F)C=C1', 'BrC1=NC=CC=C1', 'ClC1=CC=C(CC)C=C1']
        # test = ['ClC1=CC=C(OC)C=C1']
    else:
        train:list[str] = _random_sample(choices, n_train)
    test:list[str] = _random_sample(_list_subs(choices,train), n_test)
    
    best_train:float = lookup[lookup[variable].isin(train)]['yield'].max()
    best_test:float = lookup[lookup[variable].isin(test)]['yield'].max()
    
    return train, best_train, test, best_test
    
def select_substrates_v2(dataset:Literal['shields','buchwald_hartwig']='shields'):
    
    from base.global_config import FILEPATH_DICT, DATASET_VARS
    
    file_path = FILEPATH_DICT[dataset]

    if file_path.endswith('.xlsx'):
        lookup = pd.read_excel(file_path, index_col=0 if dataset in ['shields'] else None)
    elif file_path.endswith('.csv'):
        lookup = pd.read_csv(file_path, index_col=0 if dataset in ['shields'] else None)
    else:
        pass
    
    # Decide a variable. More possible values, higher prob to be sampled.
    var_dict = DATASET_VARS[dataset]
    variables, weights = zip(*var_dict.items())
    variable =  np.random.choice(
        variables,
        p=np.array(weights)/sum(weights)
    )
    # Decide n_train(>=2) and n_test(>=1). 
    choices = lookup[variable].unique().tolist()
    n_max = len(choices)
    
    n_train = np.random.randint(2, n_max)
    n_test = np.random.randint(1, n_max-n_train+1)
    
    train:list[str] = _random_sample(choices, n_train)
    test:list[str] = _random_sample(_list_subs(choices,train), n_test)
    
    best_train:float = lookup[lookup[variable].isin(train)]['yield'].max()
    best_test:float = lookup[lookup[variable].isin(test)]['yield'].max()
    
    return train, best_train, test, best_test, variable, n_train, n_test

def _filter_lookup(lookup:pd.DataFrame, variable:str=None, accessible:list[str]=None):
    '''
    Not in_place modification;
    return a deepcopied object.
    '''
    
    lookup = lookup.copy(deep=True)
    
    if variable is None or accessible is None:
        return lookup
    
    if not variable in lookup.columns:
        raise ValueError(f'{variable} not exist in df columns.') 
    
    lookup = lookup[lookup[variable].isin(accessible)]
    return lookup

def _filter_campaign(campaign:Campaign, variable:str=None, accessible:list[str]=None):
    campaign = deepcopy(campaign)
    
    campaign.toggle_discrete_candidates(pd.DataFrame({variable: accessible}), exclude=True, complement=True) # Exclude *non-matching* rows == accessible
    
    return campaign

def _load_partial_data(dataset='shields', variable:str=None, accessible:list[str]=None):
    """
    when we don't have access to the whole search space.
    """
    dataset = dataset.lower()
    
    from base.global_config import FILEPATH_DICT
    
    file_path = FILEPATH_DICT[dataset]

    if file_path.endswith('.xlsx'):
        lookup = pd.read_excel(file_path, index_col=0 if dataset in ['shields'] else None)
    elif file_path.endswith('.csv'):
        lookup = pd.read_csv(file_path, index_col=0 if dataset in ['shields'] else None)
    else:
        raise ValueError("Unsupported file format. Must be .xlsx or .csv")
    
    lookup = _filter_lookup(lookup, variable, accessible) # change

    # --- Dataset-specific parsing ---
    if dataset == 'shields':
        # if not args.no_normalize:
        lookup["Temp_C"] = _normalize(lookup["Temp_C"])
        lookup["Concentration"] = _normalize(lookup["Concentration"])

        solvent_data = dict(sorted(set(zip(lookup.Solvent, lookup.Solvent_SMILES))))
        base_data = dict(sorted(set(zip(lookup.Base, lookup.Base_SMILES))))
        ligand_data = dict(sorted(set(zip(lookup.Ligand, lookup.Ligand_SMILES))))

        
        numerical_params = {
            'Temp_C': NumericalDiscreteParameter(name="Temp_C", values=set(lookup.Temp_C)),
            'Concentration': NumericalDiscreteParameter(name="Concentration", values=set(lookup.Concentration)),
        }
        
        discrete_data = {'Solvent': solvent_data,
                        'Base': base_data,
                        'Ligand': ligand_data,
                        'rxn_name': None,
                        }
        

    elif dataset == 'buchwald_hartwig':
        ikeys = lookup.keys()
        for key in ikeys:
            lookup[key+'_name'] = lookup[key]
            
        aryl_halide_data = dict(sorted(set(zip(lookup.aryl_halide_smiles_name, lookup.aryl_halide_smiles))))
        base_data = dict(sorted(set(zip(lookup.base_smiles_name, lookup.base_smiles))))
        ligand_data = dict(sorted(set(zip(lookup.ligand_smiles_name, lookup.ligand_smiles))))
        additive_data = dict(sorted(set(zip(lookup.additive_smiles_name, lookup.additive_smiles))))
        
        numerical_params = {}
        # TODO variable name changed in order to match column name in original dataset (before '_name' is added to the end).
        # warnings.warn('For transfer learning aim, \'aryl_halide_smiles\' instead of \'aryl_halide_smiles_name\' is used as Parameter name, which should be searched in lookup table. Thus no modification (e.g., canonical smiles) to the aryl_halide_smiles column is allowed!')
        
        discrete_data = {'aryl_halide_smiles': aryl_halide_data,
                        'base_smiles': base_data,
                        'ligand_smiles': ligand_data,
                        'additive_smiles': additive_data,
                        'rxn_name': None,
                        }
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    # lookup['yield'] = _normalize(lookup['yield']) # This normalization doesn't matter because: 
    # (1) we use only one single target;
    # (2) target standardization is automatically handled in BoTorch: see GaussianProcessSurrogate class of BayBE.
    F_BEST = lookup['yield'].max()
    objective = SingleTargetObjective(target=NumericalTarget(name="yield", mode="MAX"))

    # --- Return unified structure ---
    return {
        'lookup': lookup,
        'F_BEST': F_BEST,
        'objective': objective,
        'numerical_params': numerical_params,
        'discrete_data': discrete_data,
    }
    
def _build_campaign(searchspace:SearchSpace, objective:SingleTargetObjective, f_dim:int, switch_after=0, prior=None):
    
    surrogate = GaussianProcessSurrogate(
        kernel_or_factory= MaternKernelFactory(prior_set=prior if prior is not None else 'adaptive_emilien', n_dim=f_dim, kernel_name_user='Matern')
    )

    init_recommenders = [RandomRecommender()] * switch_after
    final_recommender = [BotorchRecommender(surrogate_model=surrogate, acquisition_function='qLogEI')]
    recommender_list = init_recommenders + final_recommender

    recommender = SequentialMetaRecommender(recommender_list, mode='reuse_last')

    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender
    )
    
    return campaign

def _unwrapped_simulate(campaign, lookup_curr, init_data=None, batch_size=1, n_iter=30, mc_runs=1, campaign_name='campaign', impute_mode='ignore', use_unwrapped_simul=False):
    
    if use_unwrapped_simul: # just to record each measurement
        result = simulate_scenarios_in_house(
            {campaign_name: campaign},
            lookup_curr,
            initial_data=init_data,
            batch_size=batch_size,
            n_doe_iterations=n_iter,
            n_mc_iterations=mc_runs if init_data is None else 1,
            impute_mode=impute_mode
        )
    else: # BayBE default
        result = simulate_scenarios(
            {campaign_name: campaign},
            lookup_curr,
            initial_data=init_data,
            batch_size=batch_size,
            n_doe_iterations=n_iter,
            n_mc_iterations=mc_runs if init_data is None else 1,
            impute_mode=impute_mode
        )
    
    return result

def _plot(df, F_BEST=100, path='transfer_learning.png', F_REF=None):
    PLOTARGS = {
        'linestyle': 'solid',
        'marker': 'o',
        'markersize': 6,
        'markeredgecolor': 'none'
    }
    
    sns.lineplot(data = df,
            x = "Num_Experiments",
            y = "yield_CumBest",
            hue = "Scenario",
            **PLOTARGS)

    plt.axhline(y = F_BEST, color = 'red', linestyle = '--', label = 'Best Possible Test')
    if F_REF is not None:
        plt.axhline(y = F_REF, color = 'black', linestyle = '--', label = 'Best Possible Train')
    plt.gcf().set_size_inches((11,6))
    plt.gca().set_ylim(0, F_BEST+5)
    plt.title(f'BO performance')
    plt.savefig(path, dpi=300)
    plt.close()

def _AUC(df:pd.DataFrame, F_BEST=100, metric_y='yield_CumBest', metric_x='Num_Experiments')->float:
    '''
    Naive AUC
    '''
    
    grouped = df.groupby(metric_x)[metric_y]
    xs = grouped.mean().index.values.astype(int)
    ys = grouped.mean().values

    # Ensure sorted by x
    idx_sorted = np.argsort(xs)
    xs, ys = xs[idx_sorted], ys[idx_sorted]

    try:
        auc = np.trapz(ys, xs)
    except: # Numpy >= 2.0
        auc = np.trapezoid(ys, xs)
    SCORE_AUC =  (auc) / (F_BEST * xs[-1])
    return SCORE_AUC

def _final_yield(df:pd.DataFrame, metric_y='yield_CumBest', metric_x='Num_Experiments'):
    grouped = df.groupby(metric_x)[metric_y]
    xs = grouped.mean().index.values.astype(int)
    ys = grouped.mean().values

    # Ensure sorted by x
    idx_sorted = np.argsort(xs)
    xs, ys = xs[idx_sorted], ys[idx_sorted]
    
    return ys[-1]