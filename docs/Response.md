## Response Generation

Customize the `model_name` in both `generate_response.py` and `generate_wmdp_response.py` to point to your unlearned model's location. Ensure you have downloaded the WMDP multiple-choice question JSON files as described in [Data.md](./Data.md).

### Forget response generation

Generate responses for **forget-relevant** WMDP questions:

```
python generate_wmdp_responses.py \
    --model Yi-34B-Chat --temperature 0 \
    --dataset_path ./data/wmdp-mcqs/cyber_questions.json \
    --output_path ./responses/wmdp-cyber/Yi-34B-Chat.json \
    --num_gpus 4\
```
- `--model`: Name of the model to evaluate  
- `--temperature`: Sampling temperature (`0` for deterministic outputs)  
- `--dataset_path`: Path to WMDP JSON file (e.g., `cyber_questions.json` or `bio_questions.json`)  
- `--output_path`: File path to save generated responses  
- `--num_gpus`: Number of GPUs to use  

You can swap `--dataset_path` between `cyber_questions.json` and `bio_questions.json`, or modify `--model` as needed.  

### Combining WMDP Datasets

After generating responses for both bio and cyber datasets, you can combine them into a single WMDP dataset using the provided combination script:

```
python data_process/wmdp_combine.py
```

**Note**: Before running the script, customize the folder paths in `wmdp_combine.py` according to your directory structure:
- `bio_dir`: Path to your bio response files
- `cyber_dir`: Path to your cyber response files  
- `output_dir`: Path where you want the combined files saved

### Forget-irrelevant response generation

Generate responses for **forget-irrelevant** benchmarks (e.g., MMLU or UltraChat):

```
python generate_responses.py \
    --model Yi-34B-Chat --temperature 0 \
    --dataset MMLU --num_samples 11_000 \
    --output_path ./responses/MMLU/Yi-34B-Chat.json \
    --num_gpus 4\
```
- `--dataset`: Dataset name (`MMLU` or `UltraChat`)  
- `--num_samples`: Number of samples to generate  

Feel free to adjust `--model`, `--dataset`, `--temperature`, and other flags to match your experimental setup.  

### Data Split