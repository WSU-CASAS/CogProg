"""llama_itercot.py
Uses iterative bootstrapping in chain of thought prompt (Iter-CoT) algorithm from this paper:
"Enhancing Chain-of-Thoughts Prompting with Iterative Bootstrapping in Large Language Models"
https://arxiv.org/abs/2304.11657

Adapted to our EMA forecasting problem and Llama 2 70B

@author Gina Sprint
@date 12/15/23
"""
import os
import re
import argparse
from inspect import cleandoc
from datetime import datetime

import jsonlines
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import numpy as np
import pandas as pd

import utils

name = "meta-llama/Llama-2-70b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(name)
tokenizer.pad_token_id = tokenizer.eos_token_id    # for open-ended generation

DEMO_POOL_CONSTRUCTION_TEMPERATURE = 0.7
INFERENCE_TEMPERATURE = 0.0

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    name,
    quantization_config=BNB_CONFIG,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

generation_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    device_map="auto",    # finds GPU
)

system_str = "You are an expert in health and behavior analytics. Please analyze the text to forecast someone's next health trend health score (a single number). Do not answer with a range or multiple numbers."
format_str = "Do not answer with a range or multiple numbers. Use format Reasoning process:... Final answer: <single number>"

# https://www.geeksforgeeks.org/python-find-closest-number-to-k-in-given-list/
def closest(lst, K):
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

def get_completion(prompt, temperature, display=True):
    if temperature == 0.0:
        temperature = 0.01 # TODO: figure this out
    do_sample = True if temperature > 0.0 else False # https://github.com/facebookresearch/llama/issues/687
    top_k = 10 if do_sample else None
    sequences = generation_pipe(
        prompt,
        max_new_tokens=1024,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=do_sample,
        top_k=top_k,
        temperature=temperature, # from paper, this is what they use for construction stage of demonstration pool
        top_p=0.9,
        return_full_text=False, # True you get prompt as well
    )
    completion = sequences[0]["generated_text"]
    if display:
        print(f"\n\nPROMPT:\n{prompt}\nEND PROMPT\n")
        print(f"GENERATED COMPLETION:\n{completion}") #repr(completion))
    return completion

def check_prediction_correct(completion, actual_score):
    # have to use llama to extract the score
    temp_system_string = "You are a helpful assistant that can extract a numeric answer from text."
    temp_instruction = f"Extract the predicted numeric score, which is labeled as Final answer:, from this text:\"\n{completion}\"\nTherefore, the answer (arabic numerals) is"
    temp_prompt = cleandoc(f"""\
    <s>[INST] <<SYS>>
    {temp_system_string}
    <</SYS>>
    {temp_instruction} [/INST]""")

    temp_completion = get_completion(temp_prompt, temperature=INFERENCE_TEMPERATURE, display=False)
    num_strs = re.findall("([-+]?\d+[.,]?\d*)", temp_completion)
    float_nums = []
    for num_str in num_strs:
        if num_str.endswith(","): # handle cases like "08," which can't be converted to float
            num_str = num_str[:-1]
        try:
            float_nums.append(float(num_str))
        except:
            pass
    pred_score = np.nan
    correct = False
    
    if len(float_nums) > 1:
        # pred_score = closest(float_nums, ref_score)
        # check if all items in an array are equal
        all_same = np.max(float_nums) == np.min(float_nums)
        if all_same:
            float_nums = [float_nums[0]] # collapse down to single value

    if len(float_nums) == 1:
        pred_score = float_nums[0]
        if np.isclose(actual_score, pred_score):
            correct = True
    else:
        print("Couldn't extract single number from temp completion:", temp_completion)
        print("Source completion:", completion)
    print(num_strs, float_nums, "pred:", pred_score, "actual:", actual_score)
    return correct, pred_score

def read_jsonlines_file(fname):
    objs = []
    with jsonlines.open(fname, mode="r") as reader:
        for obj in reader.iter():
            objs.append(obj)
    return objs

def get_train_test_objs(dataset_dirname):
    train_objs = read_jsonlines_file(os.path.join(dataset_dirname, "train.jsonl"))
    val_objs = read_jsonlines_file(os.path.join(dataset_dirname, "val.jsonl"))
    train_objs += val_objs # have no purpose for validation set at this point, so combine
    test_objs = read_jsonlines_file(os.path.join(dataset_dirname, "test.jsonl"))
    return train_objs, test_objs

def get_initial_completion(prompt_str, actual_score, temperature):
    instruction = f"{prompt_str} Let's think step by step. {format_str}"
    orig_prompt = cleandoc(f"""\
        <s>[INST] <<SYS>>
        {system_str}
        <</SYS>>
        {instruction} [/INST]""")

    completion = get_completion(orig_prompt, temperature=temperature, display=False)
    correct, pred_score = check_prediction_correct(completion, actual_score)
    return instruction, completion, pred_score, correct

def get_initial_completions(objs):
    initials = []
    for i, obj in enumerate(objs):
        print(f"Processing #{i + 1}/{len(objs)}")
        prompt_str = obj["prompt"]
        actual_score = obj["actual_score"]
        instruction, completion, pred_score, correct = get_initial_completion(prompt_str, actual_score, DEMO_POOL_CONSTRUCTION_TEMPERATURE)
        initials.append({
            "pid": obj["pid"],
            "instructions": [instruction],
            "completions": [completion],
            "predicted_scores": [pred_score],
            "actual_score": obj["actual_score"],
            "score_label": obj["score_label"],
            "correct": correct,
            "summary_completion": np.nan # only retries will have summary
        })
    return initials

def get_summary_reasoning_chain(prev_instruction, prev_completion):
    print(f"\n~~~~~~~~~~~~~~~~~~~~~~CORRECT AFTER RETRIES~~~~~~~~~~~~~~~~~~~~~~\n")
    memory_str = "Here is the latest conversation between Assistant and User:\n"
    memory_str += cleandoc(f"""\
    User: {prev_instruction}
    Assistant: {prev_completion}
    """)
    instruction = f"Can you give me a complete solution reasoning process and final answer again?"
    prompt = cleandoc(f"""\
    <s>[INST] <<SYS>>
    {system_str}

    {memory_str}
    <</SYS>>
    {instruction} [/INST]""")

    completion = get_completion(prompt, temperature=DEMO_POOL_CONSTRUCTION_TEMPERATURE, display=False)
    # correct, pred_score = check_prediction_correct(completion, actual_score)
    return completion

def run_retry_bootstrap(incorrect):
    actual_score = incorrect["actual_score"]
    memory_str = "Here is the latest conversation between Assistant and User:\n"
    for i in range(-1, -len(incorrect["instructions"]) - 1, -1): # newest is last
        if i >= -2: # only use two most recent back and forths to avoid hitting max token limit
            memory_str += cleandoc(f"""\
            User: {incorrect["instructions"][i]}
            Assistant: {incorrect["completions"][i]}
            """)
    instruction = f"This answer is not right."
    if len(incorrect["predicted_scores"]) >= 2:
        prev_prev_diff = np.abs(incorrect["predicted_scores"][-2] - actual_score)
        prev_diff = np.abs(incorrect["predicted_scores"][-1] - actual_score)
        direction = None
        if prev_diff < prev_prev_diff:
            direction = "closer to"
        elif prev_diff > prev_prev_diff:
            direction = "farther from"
        if direction is not None:
            instruction = f"This answer is not right, you are getting {direction} the final answer."
    orig_instruction = incorrect["instructions"][0]
    instruction += f" As a reminder, here was the original question: {orig_instruction}. Can you think more carefully and give me the final answer?"
    prompt = cleandoc(f"""\
    <s>[INST] <<SYS>>
    {system_str}

    {memory_str}
    <</SYS>>
    {instruction} [/INST]""")

    completion = get_completion(prompt, temperature=DEMO_POOL_CONSTRUCTION_TEMPERATURE, display=False)
    correct, pred_score = check_prediction_correct(completion, actual_score)
    if correct:
        summary_completion = get_summary_reasoning_chain(instruction, completion)
        incorrect["summary_completion"] = summary_completion
    incorrect["correct"] = correct
    incorrect["instructions"].append(instruction)
    incorrect["completions"].append(completion)
    incorrect["predicted_scores"].append(pred_score)

    return incorrect


def save_demonstration_pool_to_file(corrects, incorrects, pool_fname):
    all_insts = corrects.copy()
    all_insts.extend(incorrects)
    for inst in all_insts:
        inst["first_instruction"] = inst["instructions"][0]
    df = pd.DataFrame(all_insts)
    print(df)
    df.to_csv(pool_fname) 

def run_retry_bootstrapping(corrects, incorrects, pool_fname):
    curr_num_correct = len(corrects)

    iter_num = 1
    iter_num_correct = [curr_num_correct]
    # from paper: "the process is repeated for six iterations until the number of correctly answereed questions no longer increases"
    while True:
        print(f"\n\n********************BOOTSTRAP ITERATION #{iter_num} #correct: {curr_num_correct}/{curr_num_correct + len(incorrects)}********************")
        new_corrects = [] # so the count doesn't get off in the display message as we remove
        for i, incorrect in enumerate(incorrects):
            print(f"Processing #{i + 1}/{len(incorrects)}")
            retry = run_retry_bootstrap(incorrect)
            if retry["correct"]:
                new_corrects.append({"incorrect": incorrect, "retry": retry})
        
        for new_correct_dict in new_corrects:
            incorrects.remove(new_correct_dict["incorrect"])
            corrects.append(new_correct_dict["retry"])
        
        # save to file as part of caching
        save_demonstration_pool_to_file(corrects, incorrects, pool_fname)
        curr_num_correct = len(corrects)
        if curr_num_correct == iter_num_correct[-1]: # no more new correct -> stopping condition
            break
        iter_num_correct.append(curr_num_correct)
        iter_num += 1

def construct_demonstration_pool(objs, pool_fname):
    if not os.path.exists(pool_fname):
        initials = get_initial_completions(objs)
        corrects = [initial_dict for initial_dict in initials if initial_dict["correct"]]
        # only care about the erroneous examples for bootstrapping
        incorrects = [initial_dict for initial_dict in initials if not initial_dict["correct"]]
        save_demonstration_pool_to_file(corrects, incorrects, pool_fname)
    else:
        pool_df = pd.read_csv(pool_fname, index_col=0)
        corrects = pool_df[pool_df["correct"] == True].to_dict('records') # to list of dicts
        incorrects = pool_df[pool_df["correct"] == False].to_dict('records') # to list of dicts
        # now need to convert strs of lists into lists
        list_cols = ["instructions", "completions", "predicted_scores"]
        for list_col in list_cols:
            for i in range(len(corrects)):
                corrects[i][list_col] = corrects[i][list_col].replace("nan", "np.nan")
                corrects[i][list_col] = eval(corrects[i][list_col])
            for i in range(len(incorrects)):
                incorrects[i][list_col] = incorrects[i][list_col].replace("nan", "np.nan")
                incorrects[i][list_col] = eval(incorrects[i][list_col])

    run_retry_bootstrapping(corrects, incorrects, pool_fname)

def create_exemplar_context_str(sample_df):
    context_str = ""
    for i, ind in enumerate(sample_df.index):
        exemplar_ser = sample_df.loc[ind]
        instruction_str = exemplar_ser["first_instruction"]
        completion_str = exemplar_ser["summary_completion"]
        assert not pd.isnull(completion_str)
        context_str += f"Example #{i + 1} Question: {instruction_str}\nExample #{i + 1} Answer: {completion_str}\n\n"
    return context_str

def create_few_shot_prompt(obj, pool_df, N=3):
    # use only bootstrapped examples e.g. ones that were correct but not correct on first try
    pool_df = pool_df[~pd.isnull(pool_df["summary_completion"])]
    # only use examples with different pid!!
    pool_df = pool_df[pool_df["pid"] != obj["pid"]]
    # only use examples from the same assessment
    pool_df = pool_df.groupby("score_label").get_group(obj["score_label"])
    assert len(pool_df) >= N
    sample_df = pool_df.sample(n=N, random_state=0)
    for ind in sample_df.index: # make sure not using test instance as few shot example
        assert sample_df.loc[ind]["pid"] != obj["pid"]
    context_str = create_exemplar_context_str(sample_df)
    prompt = obj["prompt"]
    prompt_str = f"{context_str}\n{prompt}"
    return prompt_str

def run_inference(objs, pool_df, pred_fname):
    preds = []
    if os.path.exists(pred_fname):
        pred_df = pd.read_csv(pred_fname)
        preds = pred_df.to_dict("records")

    for i in range(len(preds), len(objs)):
        obj = objs[i]
        print(f"Processing #{i + 1}/{len(objs)}")
        prompt_str = create_few_shot_prompt(obj, pool_df)
        actual_score = obj["actual_score"]
        pred_score = np.nan
        num_retries = 0
        while pd.isnull(pred_score) and num_retries < 10: # retry until we get a number, try 10 times then give up
            if num_retries >= 1:
                print("Run inference # of retries to get a pred_score:", num_retries)
            instruction, completion, pred_score, correct = get_initial_completion(prompt_str, actual_score, INFERENCE_TEMPERATURE)
            num_retries += 1

        preds.append({
            "instruction": instruction,
            "completion": completion,
            "predicted_score": pred_score,
            "actual_score": actual_score,
            "score_label": obj["score_label"],
            "correct": correct
        })
        # save every 5 as we go, also save on last one
        if i % 5 == 0 or i == len(objs) - 1:
            pred_df = pd.DataFrame(preds)
            pred_df.to_csv(pred_fname, index=False)

    return pred_df

def calculate_results(pred_df, res_fname):
    result_dict = utils.compute_metrics_per_score_label(pred_df)
    result_df = pd.DataFrame(result_dict).T.sort_values(f"_{utils.METRIC}")
    result_df.to_csv(res_fname)
    return result_df

def setup_and_run(dataset_dirname, demo_dirname, ts, pool_complete=False):
    pool_fname = os.path.join(demo_dirname, f"exemplars_{ts}.csv")
    if not pool_complete:
        # "training" e.g. construct demonstration pool    
        train_objs, test_objs = get_train_test_objs(dataset_dirname)
        train_objs += test_objs # try to build exemplars for all of them. during inference we can filter out by pid
        # train_objs = train_objs[:5] # just for now
        construct_demonstration_pool(train_objs, pool_fname)

    # inference
    _, test_objs = get_train_test_objs(dataset_dirname) 

    # "testing" e.g. run few-shot inference
    # test_objs = test_objs[:3] # just for now
    pool_df = pd.read_csv(pool_fname, index_col=0)
    pred_df = run_inference(test_objs, pool_df, pred_fname)
    pred_fname = os.path.join(demo_dirname, f"test_predictions_{ts}.csv")
    pred_df.to_csv(pred_fname, index=False)
    res_fname = os.path.join(demo_dirname, f"test_results_{ts}.csv")
    res_df = calculate_results(pred_df, res_fname)
    print(res_df)

if __name__ == "__main__":
    # example run
    # python llama_itercot.py daily_ema_base_context daily_ema_base_context_itercot_results
    parser = argparse.ArgumentParser(description="Iter-CoT w/Llama model")
    parser.add_argument("-i",
                        type=str,
                        dest="ema_directory_name",
                        default="daily_ema_base_context",
                        help="The directory to read the ema prompt/completion pairs from")
    parser.add_argument("-r",
                        type=str,
                        dest="results_directory_name",
                        default="daily_ema_base_context_itercot_results",
                        help="The directory to write the results to")
    parser.add_argument("-t",
                        type=str,
                        dest="demo_pool_timestamp",
                        default=datetime.now().strftime("%m-%d_%H:%M"),
                        help="The timestamp of the demonstration pool to use (omit if no existing demonstration pool has been created)")
    args = parser.parse_args()

    setup_and_run(args.ema_directory_name, args.results_directory_name, args.demo_pool_timestamp, pool_complete=False)
