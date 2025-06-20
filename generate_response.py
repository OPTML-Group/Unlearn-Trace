import argparse
from anthropic import Anthropic
from datasets import load_dataset
import google.generativeai as genai
import json
import numpy as np
from openai import OpenAI
import os
import io
from tqdm import tqdm
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
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

    model = LLM(model_name, tensor_parallel_size=args.num_gpus, trust_remote_code=True)
    tokenizer = model.get_tokenizer()
    
    return model, tokenizer

def create_dataset(args):
    if args.dataset == "UltraChat":
        dataset = load_dataset("HuggingFaceH4/ultrachat_200k", trust_remote_code=True)["train_sft"]
        get_prompt = lambda x: x["prompt"]
    elif args.dataset == "Cosmopedia":
        dataset = load_dataset("HuggingFaceTB/cosmopedia-100k", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["prompt"]
    # elif args.dataset == "MMLU":
    #     dataset = load_dataset("cais/mmlu", name="all", trust_remote_code=True)["test"]
    #     get_prompt = lambda x: x["question"]
    elif args.dataset == "MMLU":
        dataset = load_dataset("cais/mmlu", name="all", trust_remote_code=True)["test"]
        def get_prompt(x):
            question = x["question"].strip()
            choices = x["choices"]
            return f"{question}\n{choices}\n\nPlease provide your analysis, then give the final answer.\n\nAnalysis:"
    elif args.dataset == "LmsysChat":
        dataset = load_dataset("lmsys/lmsys-chat-1m", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["conversation"][0]["content"]
    elif args.dataset == "WildChat":
        dataset = load_dataset("allenai/WildChat", trust_remote_code=True)["train"]
        get_prompt = lambda x: x["conversation"][0]["content"]
    elif args.dataset == "FineWeb":
        dataset = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", trust_remote_code=True)
        get_prompt = lambda x: x["text"]
    return dataset, get_prompt

def generate_responses_instruct_llm(args):
    model, tokenizer = load_vllm_model(args)
    dataset, get_prompt = create_dataset(args)

    random_indices = random.sample(range(len(dataset)), args.num_samples)
    all_prompts = []
    data = []

    # 1. collect and filter the prompt
    for i in random_indices:
        prompt = get_prompt(dataset[i])
        if not prompt or not isinstance(prompt, str):
            continue
        dialog = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            dialog, tokenize=False, add_generation_prompt=True
        )
        if not formatted or not isinstance(formatted, str):
            continue
        if not tokenizer.encode(formatted, add_special_tokens=False):
            continue

        data.append(dialog)
        all_prompts.append(formatted)

    print(f"Collected {len(all_prompts)} valid prompts")

    # 2. use batch to call vLLM
    batch_size = 64
    for batch_start in range(0, len(all_prompts), batch_size):
        batch_prompts = all_prompts[batch_start : batch_start + batch_size]
        batch_data    = data[batch_start : batch_start + batch_size]

        try:
            outputs = model.generate(
                batch_prompts,
                SamplingParams(
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
            )
        except Exception as e:
            print(f"vLLM generate lost at batch {batch_start}:{batch_start+batch_size}", e)
            continue

        for idx, dialog in enumerate(batch_data):
            text = outputs[idx].outputs[0].text or ""
            if "mistral" in args.model:
                text = text.strip()
            elif "gemma" in args.model:
                text = text.rstrip("\n")
            dialog.append({"role": "assistant", "content": text})

    # 3. write the log（utf-8-safe）
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    import io
    json_text = json.dumps(data, ensure_ascii=False, indent=2)
    with io.open(args.output_path, "w", encoding="utf-8") as f:
        f.write(json_text)

def generate_responses_base_llm(args):
    model, tokenizer = load_vllm_model(args)
    dataset, get_prompt = create_dataset(args)

    random_indices = random.sample(range(len(dataset)), args.num_samples)
    data = []
    all_prompts = []
    for i in random_indices:
        prompt = get_prompt(dataset[i])

        # Tokenization and prompt validation
        tokens = tokenizer.encode(prompt, padding=False)[:128]
        if not tokens or len(tokens) == 0:
            print(f"Error: Tokenization failed for prompt at index {i}")
            continue

        truncated_prompt = tokenizer.decode(tokens, skip_special_tokens=True)
        
        all_prompts.append(truncated_prompt)
        data.append([{"role": "user", "content": truncated_prompt}])

    outputs = model.generate(
        all_prompts,
        SamplingParams(
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_tokens=1024,
        )
    )
    
    for i in range(len(data)):
        data[i].append({"role": "assistant", "content": outputs[i].outputs[0].text})

    # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    # with open(args.output_path, "w", encoding="utf-8") as f:
    #     json.dump(data, f, ensure_ascii=False, indent=2)
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    json_text = json.dumps(data, ensure_ascii=False, indent=2)

    with io.open(args.output_path, "w", encoding="utf-8") as f:
        f.write(json_text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Miscellaneous
    parser.add_argument("--seed", type=int, default=42, help="the seed that controls the randomness")
    parser.add_argument("--device", type=str, default="cuda", help="the device to use for generation")
    parser.add_argument("--num_gpus", type=int, default=1, help="the number of gpus to use for generation")
    
    # Data
    parser.add_argument("--dataset", type=str, default="UltraChat", 
                        choices=["UltraChat", "Cosmopedia", "MMLU", "LmsysChat", "WildChat", "FineWeb"], 
                        help="the dataset to generate responses from")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["Llama3.1-8b", "Llama3.1-8b-rmu", "Llama3.1-8b-npo"
                                 "Qwen2.5-7b", "Qwen2.5-7b-rmu", "Qwen2.5-7b-npo", 
                                 "Zephyr-7b", "Zephyr-7b-rmu", "Zephyr-7b-npo", 
                                 "Yi-34B-Chat", "Yi-34B-Chat-rmu", "Yi-34B-Chat-npo", 
                                 "Qwen2.5-7b", "Qwen2.5-7b-rmu", "Qwen2.5-7b-npo", 
                                 "Qwen2.5-14b", "Qwen2.5-14b-rmu", "Qwen2.5-14b-npo"], 
                        help="the model to generate responses from")
    parser.add_argument("--num_samples", type=int, default=11_000, help="the number of samples to generate")
    parser.add_argument("--output_path", type=str, default=None, help="the path to save the output")
    
    # Sampling hyperparameters
    parser.add_argument("--temperature", type=float, default=0, help="the temperature of the sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="the repetition penalty of the sampling")
    parser.add_argument("--max_tokens", type=int, default=1024, help="the maximum number of tokens to generate")
    
    args = parser.parse_args()
    print(args)
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    if args.model in ["Yi-34B-Chat"]:
        generate_responses_instruct_llm(args)
    else:
        generate_responses_base_llm(args)
