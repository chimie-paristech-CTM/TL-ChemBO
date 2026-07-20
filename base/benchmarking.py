import pandas as pd
import numpy as np
from baybe.parameters import NumericalDiscreteParameter, SubstanceParameter, CustomDiscreteParameter, CategoricalParameter, TaskParameter
from baybe.objectives import SingleTargetObjective
from baybe.targets import NumericalTarget
from baybe.searchspace import SearchSpace
from baybe.surrogates import GaussianProcessSurrogate
from baybe import Campaign
from baybe.recommenders import BotorchRecommender, RandomRecommender, TwoPhaseMetaRecommender

from base.kernels import MaternKernelFactory
from base.pretrained_repr import PretrainedWrapper, ChemBERTa_Fingerprint, CheMeleonFingerprint, LLM_Fingerprint
from base.utils import custom_fingerprinter, custom_PCA_fingerprinter, custom_PCA_from_substance

from baybe.simulation import simulate_scenarios

from base.simulation import simulate_scenarios_in_house

import warnings
    
def _normalize(pd_col):
    '''
    between 0 and 1.
    '''
    pd_col = pd_col - pd_col.min()
    pd_col = pd_col / ( pd_col.max() - pd_col.min()  )
    return pd_col

def load_data(dataset):
    """
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
        # FIXME variable name changed on 26.04.2026 for Transfer Learning substrate sampling
        # warnings.warn('For transfer learning aim, \'aryl_halide_smiles\' instead of \'aryl_halide_smiles_name\' is used as Parameter name, which should be searched in lookup table. Thus no modification (e.g., canonical smiles) to the aryl_halide_smiles column is allowed!')
        # FIXME aryl_halide_smiles_name is 
        # (1) the name of a CustomDiscreteParameter
        # (2) the name of a col in searchspace's comp_df and exp_df
        # (3) the lookup table col to match yield
        # Theredore, one should be able to look a key in aryl_halide_data up in the aryl_halide_smiles_name col in the lookup table.
        # So, when aryl_halide_smiles is transferred, aryl_halide_smiles can no longer be the name of CustomDiscreteParameter below
        # 'Var_Name' : {'A':'CNOOOC', 'B':'NCCCCO'} ==> look the 'Var_Name' col up in the lookup table, and find the info for 'A' and 'B'
        
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

def generate_FP(data_dict, fp_type, PCA=False, decorr_threshold=0.7):
    """
    data_dict = {
            'solvent': solvent_data | None,
            'base': base_data | None,
            'ligand': ligand_data | None,
            'additive': ...,
            'aryl_halide': ...,
        }
    """
    if fp_type in ['chemeleon', 'CheMeleon']:
        fingerprinter = PretrainedWrapper(CheMeleonFingerprint)
    elif fp_type in ['chemberta_small', 'ChemBERTa_s']:
        fingerprinter = PretrainedWrapper(ChemBERTa_Fingerprint, variant='zinc-base-v1')
    elif fp_type in ['chemberta_large', 'ChemBERTa_l']:
        fingerprinter = PretrainedWrapper(ChemBERTa_Fingerprint, variant='deepchem-100M-MLM')
    elif fp_type in ['t5-base', 'T5']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='t5-base', pooling_method='average', normalize_embeddings=False)
    elif fp_type in ['t5-base-chem', 'T5Chem']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='GT4SD/multitask-text-and-chemistry-t5-base-augm', pooling_method='average', normalize_embeddings=False)
    elif fp_type in ['UAE-Large-V1', 'UAE-Large']:
        fingerprinter = PretrainedWrapper(LLM_Fingerprint, model_name='WhereIsAI/UAE-Large-V1', pooling_method='average', normalize_embeddings=False)
    else:
        fingerprinter = None
    
    def make_param(name, data):
        if data is None:
            return None
        
        NORMALIZE = None # 'global' | 'local' 
        # NOTE This is not used because BayBE calls BoTorch to handle the input (local) normalization and outcome standardization automatically.
        
        if fp_type in ['mordred', 'Mordred']:
            if PCA:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_PCA_from_substance(data=data, encoding='MORDRED', norm=NORMALIZE),
                    decorrelate=False,
                )
            else:
                return SubstanceParameter(name=name, data=data, encoding='MORDRED', decorrelate=decorr_threshold) # global normalization for MORDRED not implemented explicitly. This could be done by modifying BayBE source code.
            
        elif fp_type in ['one_hot', 'OHE', 'One-Hot']:
            return CategoricalParameter(name=name, values=data.keys(), encoding="OHE")
        
        else:
            if PCA:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_PCA_fingerprinter(data, fingerprinter, norm=NORMALIZE),
                    decorrelate=False,
                )
            else:
                return CustomDiscreteParameter(
                    name=name,
                    data=custom_fingerprinter(data, fingerprinter, norm=NORMALIZE),
                    decorrelate=decorr_threshold,
                )
            
    # Automatically handle whatever data keys are available
    param_dict = {name: make_param(name, data) for name, data in data_dict.items() if data is not None and name!= 'rxn_name'}
    return param_dict
    
def create_recommender(kernel_prior:str, switch_after:int, acq_func:str, searchspace=None, init_method='random', feat_dim=None, kernel_name='Matern'):
    '''
    Give searchspace when LHS is wanted.
    '''

    custom_surrogate = GaussianProcessSurrogate(
        kernel_or_factory= MaternKernelFactory(
            prior_set=kernel_prior,
            n_dim=feat_dim,
            kernel_name_user = kernel_name,
        )
    )

    if switch_after > 0:
        recommender=TwoPhaseMetaRecommender(
                initial_recommender=RandomRecommender(),
                recommender=BotorchRecommender(
                    surrogate_model=custom_surrogate,
                    acquisition_function=acq_func,
                ),
                switch_after=switch_after # number of experiments required.
            )
    else:
        recommender=BotorchRecommender(
            surrogate_model=custom_surrogate,
            acquisition_function=acq_func,
            )
    
    return recommender

def create_search_space(search_params_dict, numeric_params_dict, task_param=None):
    search_params_list = []
    
    for name, parameter in search_params_dict.items():
        if parameter is not None:
            search_params_list.append(parameter)
    
    for name, parameter in numeric_params_dict.items():
        search_params_list.append(parameter)
        
    if task_param is not None:
        search_params_list.append(task_param)
    
    searchspace = SearchSpace.from_product(parameters=search_params_list)
    
    return searchspace

def create_campaign(searchspace, objective, recommender):
    
    campaign = Campaign(
        searchspace=searchspace,
        objective=objective,
        recommender=recommender,
    )
    
    return campaign

def unwrapped_simulate(args, campaign_name, campaign, lookup_curr, init_data):
    
    if args.use_unwrapped_simul: # just to have access to the GP lengthscales
        result = simulate_scenarios_in_house(
            {campaign_name: campaign},
            lookup_curr,
            initial_data=init_data,
            batch_size=args.batch_size,
            n_doe_iterations=args.n_iter,
            n_mc_iterations=1 if (args.use_transfer or args.test_metric) else args.mc_runs,
            impute_mode=args.impute_mode
        )
    else: # BayBE default
        result = simulate_scenarios(
            {campaign_name: campaign},
            lookup_curr,
            initial_data=init_data,
            batch_size=args.batch_size,
            n_doe_iterations=args.n_iter,
            n_mc_iterations=1 if (args.use_transfer or args.test_metric) else args.mc_runs,
            impute_mode=args.impute_mode
        )
    
    return result