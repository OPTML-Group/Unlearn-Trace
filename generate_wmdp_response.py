import argparse
import json
import numpy as np
import random
import torch
import os
from vllm import LLM, SamplingParams

def load_vllm_model(args):
    # For local models, load directly from the local directory or model identifier
    if args.model == "Zephyr-7b-rmu":
        model_name = "./zephyr_rmu_model"
    elif args.model == "Zephyr-7b-npo":
        model_name = "./zephyr_npo_model"
    elif args.model == "Yi-34B-Chat-rmu":
        model_name = "./yi-rmu-model"
    elif args.model == "Yi-34B-Chat-npo":
        model_name = "./yi-npo-model"
    elif args.model == "Llama3.1-8b-npo":
        model_name = "./llama8b-npo-model"
    elif args.model == "Llama3.1-8b-rmu":
        model_name = "./llama8b-rmu-model"
    elif args.model == "Qwen2.5-7b-npo":
        model_name = "./qwen7b-npo-model"
    elif args.model == "Qwen2.5-7b-rmu":
        model_name = "./qwen7b-rmu-model"
    elif args.model == "Qwen2.5-14b-npo":
        model_name = "./qwen7b-npo-model"
    elif args.model == "Qwen2.5-14b-rmu":
        model_name = "./qwen7b-rmu-model"
    else:
        model_name_to_hf_name = {
            "Zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
            "Yi-34B-Chat": "01-ai/Yi-34B-Chat",
            "Llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
            "Qwen2.5-7b": "Qwen/Qwen2.5-7B",
            "Qwen2.5-14b": "Qwen/Qwen2.5-14B",
        }
        model_name = model_name_to_hf_name.get(args.model, "HuggingFaceH4/zephyr-7b-beta")

    # # Load model and tokenizer from local directories or Hugging Face Hub
    # model = AutoModelForCausalLM.from_pretrained(model_name)
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = LLM(model_name, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer

def create_dataset_from_json(json_file):
    with open(json_file, 'r') as f:
        dataset = json.load(f)
    
    def get_prompt(item):
        # Creating a prompt from the question only
        question = item["question"]
        prompt = f"Question: {question}\nAnswer:"
        return prompt

    return dataset, get_prompt

# Function to generate responses from the model
def generate_responses_base_llm(args):
    model, tokenizer = load_vllm_model(args)
    
    # Load the new dataset from the provided JSON file
    dataset, get_prompt = create_dataset_from_json(args.dataset_path)
    
    if len(dataset) < args.num_samples:
        print(f"Warning: The dataset contains fewer samples than requested ({len(dataset)} < {args.num_samples}).")
        args.num_samples = len(dataset)  # Adjust num_samples to fit the dataset size

    random_indices = random.sample(range(len(dataset)), args.num_samples)
    data = []
    all_prompts = []

    # Loop through the selected dataset samples
    for i in random_indices:
        # Get the question only
        question = dataset[i]["question"]
        
        # Create the prompt from the question
        prompt = f"Question: {question}\nAnswer:"

        # Tokenize the prompt
        tokens = tokenizer.encode(prompt, padding=False)[:64]
        truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        
        # Store the prompt and user role in the data
        data.append([{"role": "user", "content": truncated_prompt}])

        # Add the prompt to the list of all prompts for generation
        all_prompts.append(truncated_prompt)
        
    # Generate model outputs
    outputs = model.generate(
        all_prompts,
        SamplingParams(
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_tokens=1024,
        )
    )
    
    # Process the generated outputs
    for i in range(args.num_samples):
        response = outputs[i].outputs[0].text
        # Add the response from the model to the "assistant" role
        data[i].append({"role": "assistant", "content": response})

    # Save the generated data into the output file in the desired format
    with open(args.output_path, "w") as file:
        json.dump(data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="the seed that controls the randomness")
    parser.add_argument("--device", type=str, default="cuda", help="the device to use for generation")
    parser.add_argument("--num_gpus", type=int, default=1, help="the number of gpus to use for generation")
    
    # Data
    parser.add_argument("--dataset_path", type=str, required=True, help="the path to the input dataset JSON file")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["Llama3.1-8b", "Llama3.1-8b-rmu", "Llama3.1-8b-npo"
                                 "Qwen2.5-7b", "Qwen2.5-7b-rmu", "Qwen2.5-7b-npo", 
                                 "Zephyr-7b", "Zephyr-7b-rmu", "Zephyr-7b-npo", 
                                 "Yi-34B-Chat", "Yi-34B-Chat-rmu", "Yi-34B-Chat-npo", 
                                 "Qwen2.5-7b", "Qwen2.5-7b-rmu", "Qwen2.5-7b-npo", 
                                 "Qwen2.5-14b", "Qwen2.5-14b-rmu", "Qwen2.5-14b-npo"], 
                        help="the model to generate responses from")
    parser.add_argument("--num_samples", type=int, default=11_000, help="the number of samples to generate")
    parser.add_argument("--output_path", type=str, required=True, help="the path to save the output")
    
    # Sampling hyperparameters
    parser.add_argument("--temperature", type=float, default=0, help="the temperature of the sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="the repetition penalty of the sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="the maximum number of tokens to generate")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Generate responses for the base LLM
    generate_responses_base_llm(args)
