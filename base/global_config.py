FILEPATH_DICT = {
    'shields': 'datasets/shields_dataset.xlsx',
    'buchwald_hartwig': 'datasets/buchwald_hartwig_Dreher_and_Doyle_input_data.xlsx',
}

DATASET_VARS = {
    'shields':{
        'Ligand' : 12,
        'Solvent' : 4,
        'Base' : 4
    },
    'buchwald_hartwig':{
        'aryl_halide_smiles' : 15,
        'base_smiles' : 3,
        'ligand_smiles' : 4,
        'additive_smiles' : 22,
    }
}