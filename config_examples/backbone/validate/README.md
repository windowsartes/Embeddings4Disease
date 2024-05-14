# Config for validation

Here you will find a description of a structure of the config using by the validation CLI.

**IMPORTANT**:

All the paths must be relative to the current working directory or absolute where you are using the cli.

## Config structure

- model:
  - type: str; BERT/ConvBERT/DeBERTa/FNet/FunnelTransformer/MobileBERT/RoBERTa/RoFormer/XLMRoBERTa/your-custom-backbone.
  - use_pretrained: **always true**; this is necessary for compatibility with the training config.
  - path_to_saved_weights: null/str; if null, then the model is loaded from hugging face, otherwise - from the specified directory.
- tokenizer:
  - use_pretrained: **always true**; this is necessary for compatibility with the training config.
  - path_to_saved_tokenizer: null/str; if null, then the tokenizer is loaded from hugging face, otherwise - from the specified directory.
- hyperparameters:
  - batch_size: uint; batch size.
  - seq_len: uint; maximum size of the input sequence (not including utility tokens).
 - validation:
   - path_to_data: str; path to the file with validation data.
   - top_k: uint; how many predictions will the model generate for the [MASK] token.
   - metrics: which metrics will be used.
     - reciprocal_rank: bool
     - simplified_dcg: bool
     - precision: bool
     - recall: bool
     - f_score: bool
     - hit: bool
     - confidence: bool