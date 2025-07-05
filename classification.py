import argparse
import evaluate
import json
import numpy as np
import random
import os
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from llm2vec import LLM2Vec
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

class LLM2VecCollator:
    def __init__(self, model):
        self.model = model

    def __call__(self, batch):
        num_texts = len(batch)
        texts = []
        labels = []
        for example in batch:
            text = self.model.prepare_for_tokenization(example["text"])
            texts.append(text)
            labels.append(example["target"])
            
        labels = torch.tensor(labels)
        inputs = self.model.tokenize(texts)
        inputs["labels"] = labels
        return inputs

class SequenceClassificationCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        num_texts = len(batch)
        texts = []
        labels = []
        for example in batch:
            texts.append(example["text"])
            labels.append(example["target"])
            
        labels = torch.tensor(labels)
        inputs = self.tokenizer(texts, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
        inputs["labels"] = labels
        return inputs

def load_dataset(args):
    all_responses = [list() for _ in range(len(args.response_paths))]
    for label, response_path in enumerate(args.response_paths):
        with open(response_path, "r") as f:
            data = json.load(f)

        for i in range(len(data)):
            response = data[i][-1]["content"]
            all_responses[label].append({"text": response, "target": label})

    all_train_datasets = []
    all_test_datasets = []
    for label in range(len(args.response_paths)):
        dataset = Dataset.from_list([each for each in all_responses[label]])
        # the seed ensures that the train and test splits are the same for each label
        dataset = dataset.train_test_split(train_size=args.num_train_samples, test_size=args.num_test_samples, seed=42)
        all_train_datasets.append(dataset['train'])
        all_test_datasets.append(dataset['test'])
    combined_train_dataset = concatenate_datasets(all_train_datasets)
    combined_train_dataset = combined_train_dataset.shuffle(seed=42)
    combined_test_dataset = concatenate_datasets(all_test_datasets)
    combined_test_dataset = combined_test_dataset.shuffle(seed=42)
    
    dataset = DatasetDict({
        'train': combined_train_dataset,
        'test': combined_test_dataset
    })

    print("Number of training samples", len(dataset['train']))
    print("Number of testing samples", len(dataset['test']))
    return dataset

def load_model(args):
    classifier_to_hf_name = {
        "bert": "bert-base-uncased",
        "t5": "google-t5/t5-base",
        "gpt2": "openai-community/gpt2",
        "llm2vec": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    }
    
    tokenizer = AutoTokenizer.from_pretrained(classifier_to_hf_name[args.classifier], trust_remote_code=True)
    
    if args.classifier == "llm2vec":
        config = AutoConfig.from_pretrained(
            classifier_to_hf_name[args.classifier], 
            trust_remote_code=True,
        )
        model = AutoModel.from_pretrained(
            classifier_to_hf_name[args.classifier],
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )

        model = PeftModel.from_pretrained(
            model,
            classifier_to_hf_name[args.classifier],
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        model = model.merge_and_unload()
        
        model = PeftModel.from_pretrained(
            model,
            f"{classifier_to_hf_name[args.classifier]}-supervised",
            is_trainable=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True,
        )
        
        # check the trainable parameters
        model.print_trainable_parameters()
        model = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
        
        hidden_size = list(model.modules())[-1].weight.shape[0]
        model.head = torch.nn.Linear(hidden_size, len(args.response_paths), dtype=torch.bfloat16)
        
        old_forward = model.forward
        # hacky way to turn LLM2Vec into a sequence classification model compatible with the HF Trainer
        def forward(**kwargs):
            if "labels" in kwargs:
                kwargs.pop("labels")
            return {"logits": model.head(old_forward(kwargs).to(torch.bfloat16))}
        model.forward = forward
    else:
        # use the sequence classification model from huggingface
        model = AutoModelForSequenceClassification.from_pretrained(
            classifier_to_hf_name[args.classifier],
            num_labels=len(args.response_paths),
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            trust_remote_code=True
        )
        
        # check the trainable parameters
        num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # add a comma between every three digits
        print(f"trainable params: {'{:,}'.format(num_trainable_params)} || all params: {'{:,}'.format(model.num_parameters())} || trainable%: {num_trainable_params / model.num_parameters():.4f}")
        if args.classifier == "bert":
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        elif args.classifier == "gpt2":
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer.eos_token
        elif args.classifier == "t5":
            # t5 has defined its padding token id
            # https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer
            pass
    
    model.config.pad_token_id = model.config.eos_token_id
    
    return model, tokenizer

def classification(args):
    # load dataset
    dataset = load_dataset(args)
    
    # load model
    model, tokenizer = load_model(args)

    # createdata collator
    if args.classifier == "llm2vec":
        data_collator = LLM2VecCollator(model)
    else:
        data_collator = SequenceClassificationCollator(tokenizer)
    
    # compute loss
    class SequenceClassificationTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model.forward(**inputs)
            logits = outputs.get("logits")
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, len(args.response_paths)), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

        def save_model(self, output_dir, _internal_call=False):
            super().save_model(output_dir)
            if args.classifier == "llm2vec":
                torch.save(self.model.head.state_dict(), os.path.join(output_dir, "head.pt"))
        
        def _load_from_checkpoint(self, checkpoint, model=None):
            super()._load_from_checkpoint(checkpoint, model=model)
            if args.classifier == "llm2vec":
                self.model.head.load_state_dict(torch.load(os.path.join(checkpoint, "head.pt")))

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy_metric = evaluate.load("accuracy")
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        return {"accuracy": accuracy}
    
    if args.eval_only:
        print("Evaluating the model...")
        training_args = TrainingArguments(
            output_dir = args.output_dir,
            do_train = False,
            do_eval = True,
            per_device_eval_batch_size = args.batch_size,
            remove_unused_columns = False,
            label_names = ["labels"],
        )
        
        trainer = SequenceClassificationTrainer(
            model = model,
            args = training_args,
            train_dataset = dataset['train'],
            eval_dataset = dataset["test"],
            data_collator = data_collator,
            compute_metrics = compute_metrics,
        )
        
        trainer._load_from_checkpoint(args.resume_from_checkpoint)
        eval_result = trainer.evaluate(ignore_keys=["past_key_values", "encoder_last_hidden_state"] if args.classifier == "t5" else None)
        print(eval_result)
        return
        
    # training arguments
    training_args = TrainingArguments(
        output_dir = args.output_dir,
        learning_rate = args.lr,
        lr_scheduler_type = "cosine",
        warmup_ratio = args.warmup_ratio,
        max_grad_norm = args.gradient_clipping,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size = args.batch_size,
        num_train_epochs = args.epochs,
        weight_decay = args.weight_decay,
        # eval_strategy = "epoch",
        eval_strategy="steps",    
        eval_steps=200,        
        logging_steps=600,  
        # report_to = "tensorboard",
        report_to = [],
        save_strategy = "epoch",
        save_total_limit = 1,
        remove_unused_columns = False,
        bf16 = True,
        gradient_checkpointing = True,
        label_names = ["labels"],
    )
    
    train_ds = dataset["train"]
    eval_ds = train_ds.select(range(min(args.num_test_samples, len(train_ds))))
    
    trainer = SequenceClassificationTrainer(
        model = model,
        args = training_args,
        # train_dataset = dataset['train'],
        # eval_dataset = dataset["test"],
        train_dataset=train_ds, 
        eval_dataset=eval_ds,
        data_collator = data_collator,
        compute_metrics = compute_metrics,
    )

    trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint,
        ignore_keys_for_eval=["past_key_values", "encoder_last_hidden_state"] if args.classifier == "t5" else None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # miscelaneous
    parser.add_argument('--seed',type=int, default=42, help="the seed that controls the randomness")
    parser.add_argument('--device', type=str, default='cuda', help="the device to use for training / evaluation")
    
    # data
    parser.add_argument("--response_paths", nargs='+', help="a list of paths to load the generated responses from")
    parser.add_argument("--num_train_samples", type=int, default=1_000, help="the number of training samples")
    parser.add_argument("--num_test_samples", type=int, default=200, help="the number of testing samples")
    
    # classifer
    parser.add_argument('--classifier', type=str, default="llm2vec", 
                        choices=["llm2vec", "bert", "t5", "gpt2"],
                        help='the text embedding model to perform sequence classification')
    
    # training hyperparameters
    parser.add_argument("--epochs", type=int, default=3, help="the number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="the batch size")
    parser.add_argument('--lr', default=9e-5, type=float, help="the learning rate")
    parser.add_argument("--gradient_clipping", type=float, default=0.3, help="the gradient clipping")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="the weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="the number of warmup steps")
    
    # evaluation
    parser.add_argument("--eval_only", action="store_true", default=False, help="only evaluate the model")
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help="the checkpoint to evaluate")
    
    # output related
    parser.add_argument("--output_dir", type=str, default=None, help="the directory to save the output")
    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    classification(args)