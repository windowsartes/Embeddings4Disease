# Config for preprocessing

In general, preprocessor's super class doesn't have any implemented methods, so working with your own dataset you shouldn't rely on any form from the config, except the line "dataset: dataset-name", so  the factory can properly create the class "dataset-namePreprocessorFactory".

For example, in the config for the MIMIC-4 dataset there is a line "dataset: MIMIC", so the factory will create a MIMICPreprocessorFactory instance.

**IMPORTANT**:

All the paths must be relative to the current working directory or absolute where you are using the cli.

## MIMIC

Here you can find a description of MIMIC's config example.

- dataset: MIMIC
- data: str; path to the raw data.
- storage_dir: str; path to the directory in which the preprocessed data will be stored.
- code_length: uint; how many first characters from ICD-10 do we use.
- code_lower_bound: str; lower limit for relevant codes, inclusive.
- code_upper_bound: str; upper limit for relevant codes, inclusive. 
- epsilon: float; probability to which the last 2 transactions of the patient will be sent for validation.
- random_seed: int; random seed.
