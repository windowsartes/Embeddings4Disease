# Config for training

Here I describe the structure of the configs so that you can customize the training processs in the way you need.
For the most ot the models, the config.yaml file is suitable, but some models still differ in their parameters, so some of the architectures have their own templates.

**IMPORTANT**:

1. All the paths must be relative to the current working directory or absolute where you are using the cli.
2. Floats must be written explicitly: 0.0001 instead of 1e-4.

## Config structure

This is a standart template - it works with BERT/ConvBERT/DeBERTa/RoBERTa/RoFormer/XLMRoBERTa.

- model
  - type: str; BERT/ConvBERT/DeBERTa/MobileBERT/RoBERTa/RoFormer/XLMRoBERTa/your-custom-backbone.
  - use_pretrained: bool; if true, then the pretrained model is loaded, if false, then a new model is created based on the config section.
  - path_to_saved_weights: null/str; if null, then the model is loaded from hugging face, otherwise - from the specified directory.
  - config: will be ignored if use_pretrained == false.
    - hidden_size: uint; embedding size.
    - num_attention_heads: uint; number of heads.
    - num_hidden_layers: uint; number of encoder layers.
- tokenizer:
  - use_pretrained: bool; use a pretrained tokenizer or not.
  - path_to_saved_tokenizer: null/str; **Will be ignored, if use_pretrained == false**. if null, then the tokenizer is loaded from hugging face, otherwise - from the specified directory.
  - path_to_vocab_file: str; **Will be ignored, if use_pretrained == true**. Path to the vocab.txt that lists the tokens.
- hyperparameters:
  - batch_size: uint; batch size.
  - seq_len: uint; maximum size of the input sequence (not including utility tokens).
- training:
  - device: cpu/cuda:0/...; device you want to use.
  - path_to_data: str;
  - n_epochs: uint; number of epochs.
  - n_warmup_epochs: uint; number of warm-up epochs.
  - n_checkpoints: uint; number of checkpoints.
  - optimizer_parameters:
    - learning_rate: float
    - adam_beta1: float
    - adam_beta2: float
    - weight_decay: float
- validation:
  - path_to_data: str
  - period: uint; once every prediod epochs the model will be validated. 
  - top_k: uint; how many predictions will the model generate for the [MASK] token.
  - save_graphs: bool; whether to save metric and loss values during the training process in the image format.
  - metrics: which metrics will be used.
    - reciprocal_rank: bool
    - simplified_dcg: bool
    - precision: bool
    - recall: bool
    - f_score: bool
    - hit: bool
    - confidence: bool
- save_trained_model: bool; save trained model of not.
- wandb - parameters for the Weights and Biases integration. The optional "wandb" dependency is required.
  - use: bool
  - project: str; projects name.
  - api_key: str; your api-key.

Some models have different architecture from BERT, so their model config section will be different. Everything else is the same.

### FNet

- model:
  - type: str; FNet.
  - config: will be ignored if use_pretrained == false.
    - hidden_size: uint; embedding size.
    - num_hidden_layers: uint; number of jidden layers.
    - intermediate_size: uint; in the original paper equals to 4*hidden_size.

### FunnelTransformer

- model:
  - type: str; FunnelTransformer
  - config:
    - d_model: uint; hidden size.
    - n_head: uint; number of heads in a single encoder layer. In the paper equals 6.
    - d_inner: uint; FF's first linear layer's output.

## Model and optimizer parameters.

The table will show you the hyperparameters of the models: the number of heads, the number of encoders and optimizer parameters.
The pre-trained models that stored on huggingface were trained in exactly this configuration. I took the model parameters from the articles based on equivalence to BERT base, and the optimizer parameters were those that the authors of the articles used for pre-training. The batch size here is based on the maximum size that will fit on one P100.
If the parameters were not specified in the article and I did not find them in the open source realizations, then I used the parameters from BERT.


| model/parameters | BERT | ConvBERT | DeBERTa | MobileBERT | RoBERTa | RoFormer |XLMRoBERTa |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size | 1024 | 512 | 512 | 1024 | 1024 | 1024 | 1024 |
| num_attention_heads | 12 | 12 | 12 | 4 | 12 | 12 | 12 |
| num_hidden_layers | 12 | 12 | 12 | 24 | 12 | 12 | 12 |
| lr | 0.0001 | 0.0002 | 0.0002 | 0.0006 | 0.0007 | 0.0005 | 0.0001 |
| beta1 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| beta2 | 0.999 | 0.999 | 0.98 | 0.98 | 0.98 | 0.98 | 0.999 |


| model/parameters | FNet | FunnelTransformer |
| -------- | ------- | ------- |
| batch_size | 1024 | 512 |
| hidden_size | 384 | - |
| num_hidden_layers | 12 | - |
| intermediate_size | 1536 | - |
| d_model | - | 384 |
| n_head | - | 6 |
| d_inner | - | 1536 |
| lr | 0.0001 | 0.0001 |
| beta1 | 0.9 | 0.9 |
| beta2 | 0.999 | 0.999 |