# Config for validation

Здесь я опишу структуру конфигов, чтобы вы смогли настроить обучение так, как вам будет нужно.
Для дольшинства моделей подойдёт файл config.yaml, но некоторые всё же отличаются своими параметрами, поэтому для части архитектур существуют собственные шаблоны.

**ВАЖНО**:
1. Все пути следует указывать относительно той директории, откуда вы используйте cli.
2. Float'ы следует писать в явном виде: 0.0001 вместо 1e-4.

## Config

Стандартный шаблон, подойдёт для BERT/ConvBERT/DeBERTa/RoBERTa/RoFormer/XLMRoBERTa

Структура:

- model - конфигурация модели.
  - type: (str) - BERT/ConvBERT/DeBERTa/MobileBERT/RoBERTa/RoFormer/XLMRoBERTa.
  - use_pretrained: false/true - если true, то загружается предобученная модель, если false, то создаётся новая модель по секции config.
  - path_to_saved_weights: null/путь-до-модели - если null, то загружается модель с hugging face, иначе - из указанной          директории.
 - config - игнорируется, если use_pretrained == false.
  - hidden_size: (uint) - размер эмбеддинга.
  - num_attention_heads: (uint) - количество голов.
  - num_hidden_layers: (uint) - количество энкодеров (encoder layer'ов).
- tokenizer:
  - use_pretrained: false/true - использовать предобученный токенайзер или нет.
  - path_to_saved_tokenizer: none/путь-до-токенайзера - игнорируется, если use_pretrained == true. Если none, то используется токенайзер с hugging face, иначе - тот, до которого указан путь.
- hyperparameters - гиперпараметры обучения.
  - batch_size: (uint) - размер батча.
  - seq_len: (uint) - макисмальный размер транзакции (не считая служебных токенов).
- storage_path - путь до директории, в которую будут сохраняться логи, чекпоинты и т.д.
- training - параметры обучения.
  - device: cpu/cuda:0/... - дейвайс, на котором будут происходить вычисления.
  - path_to_data - путь до файла с тренировочными данными.
  - n_epochs: (uint) - количество эпох.
  - n_warmup_epochs: (uint) - количество warm-up эпох.
  - n_checkpoints: (uint) - количество одновременно существующих чекпоинтов.
  - optimizer_parameters - параметры для AdamW.
    - learning_rate: (float)
    - adam_beta1: (float)
    - adam_beta2: (float)
    - weight_decay: (float)
- validation - параметры валидации.
  - path_to_data - путь до файла с валидационными данными.
  - period: (uint) - раз в сколько эпох нужно проводить валидацию модели. 
  - top_k: (uint) - сколько предсказаний делать для [MASK]. 
  - save_graphs: false/true - сохранять ли значения метрик и лосс в процессе обучения в виде графиков.
  - metrics - какие метрики использовать для валидации.
    - reciprocal_rank: false/true
    - simplified_dcg: false/true
    - precision: false/true
    - recall: false/true
    - f_score: false/true
    - hit: false/true
- save_trained_model: false/true - нужно ли сохранять модель после обучения.

У некоторых моделей архитектура отличается от BERT'a, поэтому их секция model будет отличатся, в остальном - всё то же самое.


### FNet

- model - конфигурация модели.
  - type: (str) - FNet.
  - use_pretrained: false/true - если true, то загружается предобученная модель, если false, то создаётся новая модель по секции config. 
  - path_to_saved_weights: null/путь-до-модели - если null, то загружается модель с hugging face, иначе - из указанной          директории.
  - config - игнорируется, если use_pretrained == false.
    - hidden_size: (uint) - размер эмбеддинга.
    - num_hidden_layers: (uint) - количество слоёв.
    - intermediate_size: (uint) - в оригинальной статье = 4 * hidden_szie.

### FunnelTransformer

- model - конфигурация модели.
  - type: FunnelTransformer
  - use_pretrained: false
  - path_to_saved_weights: null
  - config:
    - d_model: (uint) - размер скрытого состояния.
    - n_head: (uint) 6 - количество голов в одном layer'e. В статье 3 слоя (значение по умолчанию, оставил таким же) с 6 головами эквивалентны BERT_base.
    - d_inner: (uint) - выход после первого линейного слоя в FF.

## Параметры моделей и оптимизаторов

Здесь в табличке будут указаны гиперпараметры моделей: количество голов, количество энкодеров и параметры оптимизаторы. 
Предобученные модели, которые лежат на huggingface, обучались именно в такой конфигурации. Параметры моделей я брал из статей из расчёта на эквивалентность BERT_base'у, а параметры оптимизаторов - те, которые авторы статей использовали для предобучения. Размер батча здесь указан из расчёта на максимальный размер, который влезет на одну P100.
Если параметров в статье указано не было и я не нашёл их в репках, то я использовал параметры от BERT'а.


| модель/параметр | BERT | ConvBERT | DeBERTa | MobileBERT | RoBERTa | RoFormer |XLMRoBERTa |
| -------- | ------- | ------- | ------- | ------- | ------- | ------- | ------- |
| batch_size | 1024 | 512 | 512 | 1024 | 1024 | 1024 | 1024 |
| num_attention_heads | 12 | 12 | 12 | 4 | 12 | 12 | 12 |
| num_hidden_layers | 12 | 12 | 12 | 24 | 12 | 12 | 12 |
| lr | 0.0001 | 0.0002 | 0.0002 | 0.0006 | 0.0007 | 0.0005 | 0.0001 |
| beta1 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 | 0.9 |
| beta2 | 0.999 | 0.999 | 0.98 | 0.98 | 0.98 | 0.98 | 0.999 |


| модель/параметр | FNet | FunnelTransformer |
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