# Config for validation

А здесь я опишу структуру конфига, который используется для валидации моделей.

**ВАЖНО**:

Все пути следует указывать относительно той директории, откуда вы используйте cli.

## Config

- model - конфигурация модели
  - type: (str) - BERT/ConvBERT/DeBERTa/FNet/FunnelTransformer/MobileBERT/RoBERTa/RoFormer/XLMRoBERTa.
  - use_pretrained: false/true - **всегда true**, это нужно для совместимости с конфигом для обучения.
  - path_to_saved_weights: null/путь-до-модели - если null, то загружается модель с hugging face, иначе - из указанной          директории.
- tokenizer - конфигурация токенайзера.
  - use_pretrained: false/true - **всегда true**, это нужно для совместимости с конфигом для обучения.
  - path_to_saved_tokenizer: null/путь-до-токенайзера - если none, то используется токенайзер с hugging face, иначе - тот, до которого указан путь.
- hyperparameters гиперпараметры обучения.
  - batch_size: (uint) - размер батча.
  - seq_len: (uint) - макисмальный размер транзакции (не считая служебных токенов).
 - validation - параметры валидации.
   - path_to_data - путь до файла с валидационными данными.t
   - top_k: (uint) - сколько предсказаний делать для [MASK].
   - metrics - какие метрики использовать для валидации.
     - reciprocal_rank: false/true
     - simplified_dcg: false/true
     - precision: false/true
     - recall: false/true
     - f_score: false/true
     - hit: false/true