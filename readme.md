CaTHPO is a code-aware cross-program transfer hyperparameter tuning framework.

# CaTHPO

## Requirement

- OS: Ubuntu 18.04

- packages:

  ```
  python 3.6.13
  pytorch 1.10.2
  tensorflow 1.15.0
  ```

## Code-aware Program Representation

1. Preprocess for program source code

   ```
   python ./pretrain/process_code_ast.py
   ```

2. Trian and evaluate the pre-trained model

   ```
   python ./pretrain/pretrain.py
   ```

## Cross-program Transfer BO

1. Meta-train the corss-program transfer BO

   ```
   python meta_train.py
   ```

2. Get the evaluation for all programs

   ```
   python evaluate.py
   ```
