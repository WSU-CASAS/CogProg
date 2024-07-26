"""llama_finetune.py
Fine tunes a 7B or 13B Llama 2 model for forecasting.

@author Gina Sprint
@date 12/15/23
"""
import os
import re
import argparse
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, pipeline, IntervalStrategy, EarlyStoppingCallback
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

import utils

# BitsAndBytesConfig int-4 config
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA config based on QLoRA paper
PEFT_CONFIG = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
)

# Llama chat prompt template: https://huggingface.co/blog/llama2
def format_sample(sample):
    return f"""
    <s>[INST] <<SYS>>
    You are an expert in health and behavior analytics. Please analyze behavior in the text to forecast someone's next timepoint health trend. Do not summarize the information. Answer only with a number.
    <</SYS>>

    {sample['prompt']} [/INST]"""

def format_train_eval_sample(sample):
    formatted_sample = format_sample(sample)
    return f"{formatted_sample} {sample['completion']} </s>"

def get_response(generate_pipeline, obj):
    prompt = format_sample(obj)
    resp_obj = generate_pipeline(prompt)
    resp_text = resp_obj[0]["generated_text"]
    
    actual = float(re.search("(\d+[.,]?\d*)", obj["completion"]).group(0))
    pred = float(re.findall("(\d+[.,]?\d*)", resp_text)[-1]) # grab the last number provided

    result = {  "actual_score": actual,
                "predicted_score": pred,
                "predicted_completion": resp_text, 
                "actual_completion": obj["completion"], 
                "pid": obj["pid"], 
                "score_label": obj["score_label"],
                "prompt": obj["prompt"]}
    return result

def run_inference(tokenizer, best_ckpt_path, test_dataset):
    model = AutoPeftModelForCausalLM.from_pretrained(best_ckpt_path, device_map="auto", torch_dtype=torch.bfloat16)
    model = model.merge_and_unload()

    generate_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task='text-generation',
        return_full_text=False,
        max_new_tokens=128,
        repetition_penalty=1.1
    )

    # one at a time
    results = []
    for i, obj in enumerate(test_dataset):
        print(f"running inference for #{i + 1}/{len(test_dataset)}")
        result = get_response(generate_pipeline, obj)
        results.append(result)
    df = pd.DataFrame(results)
    return df

def run_finetuning(output_dir, data_files, model_id):
    # load dataset
    datasets = load_dataset("json", data_files=data_files)
    for name in ["train", "val", "test"]:
        print(f"{name} # samples = {len(datasets[name])}")

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=BNB_CONFIG, use_cache=False, device_map="auto")
    base_model.config.pretraining_tp = 1

    # prepare model for training
    model = prepare_model_for_kbit_training(base_model)
    model = get_peft_model(model, PEFT_CONFIG)
    batch_size = 4
    eval_steps = len(datasets["val"]) // batch_size if len(datasets["val"]) // batch_size != 0 else 10
    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=10,
        evaluation_strategy = IntervalStrategy.STEPS,
        eval_steps=eval_steps,
        save_strategy=IntervalStrategy.STEPS,
        save_steps=eval_steps,
        logging_strategy=IntervalStrategy.STEPS,
        logging_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=2,
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        seed=0
    )

    # train model
    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        peft_config=PEFT_CONFIG,
        max_seq_length=2048,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=format_train_eval_sample,
        args=args,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    trainer.save_model()
    best_ckpt_path = trainer.state.best_model_checkpoint

    # running inference on trained model
    df = run_inference(tokenizer, best_ckpt_path, datasets["test"])
    return df

def setup_and_run(dataset_dirname, results_dirname, model_name):
    data_files = {"train": os.path.join(dataset_dirname, "train.jsonl"), 
            "val": os.path.join(dataset_dirname, "val.jsonl"),
            "test": os.path.join(dataset_dirname, "test.jsonl")}
    pred_df = run_finetuning(results_dirname, data_files, model_name)
    pred_fname = os.path.join(results_dirname, f"test_predictions.csv")
    pred_df.to_csv(pred_fname, index=False)

    result_df = utils.compute_metrics_per_score_label(pred_df)
    res_fname = os.path.join(results_dirname, "test_results.csv")
    result_df.to_csv(res_fname)
    print(result_df)

if __name__ == "__main__":
    # example run
    # python llama_finetune.py -i daily_ema_base_context -r daily_ema_base_context_llama_results -m meta-llama/Llama-2-7b-chat-hf
    parser = argparse.ArgumentParser(description="Fine tune Llama model")
    parser.add_argument("-i",
                        type=str,
                        dest="ema_directory_name",
                        default="daily_ema_base_context",
                        help="The directory to read the ema prompt/completion pairs from")
    parser.add_argument("-r",
                        type=str,
                        dest="results_directory_name",
                        default="daily_ema_base_context_llama_results",
                        help="The directory to write the results to")
    parser.add_argument("-m",
                        type=str,
                        dest="model_id",
                        default="meta-llama/Llama-2-7b-chat-hf",
                        help="Hugging face llama model id to fine tune (7B or 13B)")
    args = parser.parse_args()

    setup_and_run(args.ema_directory_name, args.results_directory_name, args.model_id)