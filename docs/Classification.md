## Classifier Training

To train classifiers for distinguishing between original and unlearned model responses, you can use the provided classification script. Below is an example command for binary classification between original and unlearned models:

```bash
python classification.py \
    --response_paths ./path/to/model1-train.json \
                     ./path/to/model1-unlearn-train.json \
    --classifier llm2vec \
    --num_train_samples 5700 \
    --num_test_samples 100 \
    --output_dir ./classification_models/
```

For N-way classification with multiple models, you can include N response paths:

```bash
python classification.py \
    --response_paths ./path/to/model1-train.json \
                     ./path/to/model1-unlearn-train.json \
                     ./path/to/model2-train.json \
                     ./path/to/model2-unlearn-train.json \
                     ./path/to/model3-train.json \
                     ./path/to/model3-unlearn-train.json \
                     ./path/to/model4-train.json \
                     ./path/to/model4-unlearn-train.json \
    --classifier llm2vec \
    --num_train_samples 5700 \
    --num_test_samples 100 \
    --output_dir ./multi_classification_models/
```

**Parameters**:
- `--response_paths`: Paths to response files from different models (space-separated)
- `--classifier`: Type of classifier to use (`llm2vec`, `gpt2`, `t5`, or `bert`)
- `--num_train_samples`: Number of training samples per class
- `--num_test_samples`: Number of test samples per class
- `--output_dir`: Directory to save trained classifier models

## Classifier Evaluation

To evaluate trained classifiers on new data, you can use the classification script with the `--eval_only` flag. Below is an example command for binary classification evaluation:

```bash
python classification.py \
    --response_paths ./path/to/model1-eval.json \
                     ./path/to/model1-unlearn-eval.json \
    --classifier llm2vec \
    --eval_only \
    --num_train_samples 5 \
    --num_test_samples 355 \
    --resume_from_checkpoint ./classification_models \
    --output_dir ./classification_models_eval/
```

For multi-class classification evaluation with multiple models:

```bash
python classification.py \
    --response_paths ./path/to/model1-eval.json \
                     ./path/to/model1-unlearn-eval.json \
                     ./path/to/model2-eval.json \
                     ./path/to/model2-unlearn-eval.json \
                     ./path/to/model3-eval.json \
                     ./path/to/model3-unlearn-eval.json \
                     ./path/to/model4-eval.json \
                     ./path/to/model4-unlearn-eval.json \
    --classifier llm2vec \
    --eval_only \
    --num_train_samples 5 \
    --num_test_samples 355 \
    --resume_from_checkpoint ./classification_models \
    --output_dir ./multi_classification_models_eval/
```

**Note**: When using `--eval_only`, the script loads pre-trained classifier models from the checkpoint directory and evaluates them on the specified response data without further training.

**Additional Parameters**:
- `--eval_only`: Flag to enable evaluation mode (no training)
- `--resume_from_checkpoint`: Path to directory containing pre-trained classifier models
