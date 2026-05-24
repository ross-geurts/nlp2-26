import os
import sys
import sacrebleu
import datasets
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import accelerate
import re
import torch
import gc
from datasets import Dataset, load_dataset
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig
from peft import get_peft_model, LoraConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig


# load data
def generate_track1_dev_splits(language_pair):
    # Given a path to the dev jsonl file, load the lines and return three lists:
    # - noterm: list of dicts with 'en' and 'de'
    # - proper: list of dicts with 'en', 'de', and 'terms' from 'proper_terms'
    # - random: list of dicts with 'en', 'de', and 'terms' from 'random_terms'
    dev_jsonl_path = f"dev-data/{language_pair}_dev.jsonl"
    src, tgt = language_pair[0:2], language_pair[2:4]
    noterm = [] 
    proper = []
    random_ = []
    with open(dev_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip("\n"))
            # noterm: only source and target
            noterm.append({
                src: entry[src],
                tgt: entry[tgt],
            })
            # proper: terminology from proper_terms
            proper.append({
                src: entry[src],
                tgt: entry[tgt],
                "terms": entry.get("proper_terms", {}),
            })
            # random: terminology from random_terms
            random_.append({
                src: entry[src],
                tgt: entry[tgt],
                "terms": entry.get("random_terms", {}),
            })
    return noterm, proper, random_

ende_noterm, ende_proper, ende_random = generate_track1_dev_splits("ende")
enes_noterm, enes_proper, enes_random = generate_track1_dev_splits("enes")
enru_noterm, enru_proper, enru_random = generate_track1_dev_splits("enru")


def generate_track1_test_splits(language_pair):
    test_jsonl_path = f"test-data/track1/{language_pair}_test.jsonl"
    src, tgt = language_pair[0:2], language_pair[2:4]
    noterm = [] 
    proper = []
    random_ = []
    with open(test_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip("\n"))
            noterm.append({
                src: entry[src],
            })
            proper.append({
                src: entry[src],
                "terms": entry.get("proper_terms", {}),
            })
            random_.append({
                src: entry[src],
                "terms": entry.get("random_terms", {}),
            })
    return noterm, proper, random_

# check data loaded correctly
# print(ende_noterm[0])
# print(ende_proper[0])
# print(ende_random[0])

 # Change this to the exact Gemma variant you want, e.g. "google/gemma-2-4b-it" or a local path.
model_id = "CohereLabs/tiny-aya-global"  # example; replace with e.g. "gemma-3-4b-it" 

src_tgt = {
    "ende": {
        "noterm": ende_noterm,
        "proper": ende_proper,
        "random": ende_random,
        "src": "en",
        "tgt": "de",
        "src_full": "English",
        "tgt_full": "German",
    },
    "enes": {
        "noterm": enes_noterm,
        "proper": enes_proper,
        "random": enes_random,
        "src": "en",
        "tgt": "es",
        "src_full": "English",
        "tgt_full": "Spanish",
    },
    "enru": {
        "noterm": enru_noterm,
        "proper": enru_proper,
        "random": enru_random,
        "src": "en",
        "tgt": "ru",
        "src_full": "English",
        "tgt_full": "Russian",
    },
}

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # Optimized NormalFloat4 data format
    bnb_4bit_compute_dtype=torch.bfloat16,   # Fast 16-bit processing inside computation layers
    bnb_4bit_use_double_quant=True        # Compresses quantization constants to save extra VRAM
)

def generate_translations(inputs, model_id, src_tgt_pair, batch_size=16):
    """
    Args:
        inputs: list of dicts, each dict should contain keys:
            - "source": source sentence
            - "terminology": string or list of term mappings (optional, use '' or [] for none)
        model: pretrained language model (AutoModelForCausalLM or compatible)
        tokenizer: tokenizer for the model
    Returns:
        List of generated outputs: translations as strings, in input order.
    """

    if "checkpoints" in model_id:
        tokenizer_path = "CohereLabs/tiny-aya-global"
    else:
        tokenizer_path = model_id
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        low_cpu_mem_usage=False,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto",
    )
    prompts = []
    for entry in inputs:
        source = entry.get(src_tgt[src_tgt_pair]["src"], "")
        terminology = entry.get("terms", "")
        prompt = (
            f"Translate the following sentence from {src_tgt[src_tgt_pair]['src_full']} to {src_tgt[src_tgt_pair]['tgt_full']}, respecting the given terminology.\n\n"
            f"Source: {source}\n"
            f"Terminology: {terminology}\n\n"
            "Translation:"
        )
        prompts.append(prompt)

    messages_list = [
        [{"role": "user", "content": prompt}] for prompt in prompts
    ]

    # Generate chat-formatted texts
    texts = [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_list
    ]
    # Tokenize input batch
    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        # Tokenize the specific mini-batch chunk
        model_inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(model.device)

        # Generate outputs for the mini-batch concurrently
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

        # Decode only the newly generated tokens for this batch slice
        for j in range(len(batch_texts)):
            input_ids_len = model_inputs.input_ids[j].shape[0]
            output_ids = generated_ids[j][input_ids_len:].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            outputs.append(content)

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
            
    return outputs

###  RQ1a: code-swithed inputs
def code_switch_source(source_text, terms_dict):
    if not terms_dict or not isinstance(terms_dict, dict):
        return source_text
    
    modified_text = source_text
    for src_term, target_term in sorted(terms_dict.items(), key=lambda x: len(x[0]), reverse=True):
        if src_term.strip():
            pattern = r'\b' + re.escape(src_term) + r'\b'
            modified_text = re.sub(pattern, target_term, modified_text)
    
    return modified_text

def generate_codeswitched_translations(inputs, model_id, src_target_pair, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype = torch.bfloat16,
        device_map = "auto",
    )

    prompts = []
    for entry in inputs:
        raw_source = entry.get(src_tgt[src_target_pair]["src"], "")
        terminology_dict = entry.get("terms", {})

        cs_source = code_switch_source(raw_source, terminology_dict)

        src_full = src_tgt[src_target_pair]["src_full"]
        target_full = src_tgt[src_target_pair]["tgt_full"]

        prompt = (
            f"Translate the following code-switched sentence into fluent {target_full}."
            f"The sentence is written in {src_full} but already has its specific domain terminology"
            f"translated into {target_full}. Maintain those {target_full} terms exactly as they are in your final output. \n\n"
            f"Source: {cs_source}\n\n"
            f"Translation:"
        )
        prompts.append(prompt)
    
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    all_texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    outputs = []
    for i in range(0, len(all_texts), batch_size):
        batch_prompts = all_texts[i:i + batch_size]
        
        model_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
        
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512, 
            temperature=0.3,
            top_p=0.9
        )
        
        for j in range(len(batch_prompts)):
            input_ids_len = model_inputs.input_ids[j].shape[0]
            output_ids = generated_ids[j][input_ids_len:].tolist()
            content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            outputs.append(content)
            
    return outputs

### rq3 - terminology adherence rewards

def compute_term_adherece_reward(candidate_translation, terms_dict):
    """
    computes the terminology adherence reward score:
    s_i = (1 / |T|) * sum(delta(t_i in o_i))
    """

    if not terms_dict or not isinstance(terms_dict, dict):
        return 0.0
    
    target_terms = [str(tgt_term).strip() for tgt_term in terms_dict.values() if str(tgt_term).strip()]
    if not target_terms:
        return 0.0
    
    matched = 0
    for tgt_term in target_terms:
        pattern = r"\b" + re.escape(tgt_term) + r"\b"
        if re.search(pattern, candidate_translation, re.IGNORECASE):
            matched+= 1

    return matched / len(target_terms)

def generate_preference_dataset(base_model, raw_data_list, src_tgt_pair, max_examples=2):
    # generating preference reward dataset using adherence rewards
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, quantization_config=bnb_config, device_map="auto"
    )

    src_key = src_tgt[src_tgt_pair]["src"]
    src_full = src_tgt[src_tgt_pair]["src_full"]
    tgt_full = src_tgt[src_tgt_pair]["tgt_full"]

    po_data_list = []
    subset_data = raw_data_list[:150]

    for entry in subset_data:
        source_sentence = entry[src_key]
        terms_dict = entry.get("terms", {})
        if not terms_dict:
            continue

        prompt = (
            f"Translate the following sentence from {src_tgt[src_tgt_pair]['src_full']} to {src_tgt[src_tgt_pair]['tgt_full']}, respecting the given terminology.\n\n"
            f"Source: {source_sentence}\n"
            f"Terminology: {terms_dict}\n\n"
            "Translation:"
        )

        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

        candidates = []
        for _ in range(max_examples):
            with torch.no_grad():
                outputs=model.generate(
                    **inputs,
                    max_new_tokens = 512,
                    do_sample = True
                )
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            candidates.append(gen_text)

        scores = [compute_term_adherece_reward(cand, terms_dict) for cand in candidates]

        if scores[0] != scores[1]:
            chosen_idx = 0 if scores[0] > scores[1] else 1
            rejected_idx = 1 if chosen_idx == 0 else 0

            po_data_list.append({
                "prompt": prompt,
                "chosen": candidates[chosen_idx],
                "rejected": candidates[rejected_idx]
            })
    
    del model
    gc.collect()
    torch.cuda.empty_cache()

    print(f"generated {len(po_data_list)} valid preference pairs based on adherence scores")
    return Dataset.from_list(po_data_list)

# training dataset formatting

base_model = model_id  # reuse the model defined above

def format_example(ex, src_tgt_pair):

    src = ex[src_tgt[src_tgt_pair]["src"]]
    terms = ex.get("terms", "")
    tgt = ex[src_tgt[src_tgt_pair]["tgt"]]
    prompt = f"Source: {src}\nTerminology: {terms}\nTranslation:"
    return {"text": prompt + " " + tgt}

src_tgt_pair = "ende"
train_ds = [format_example(ex, src_tgt_pair) for ex in src_tgt[src_tgt_pair]["proper"]]
# print(train_ds[0])

# peft model training setup
def run_sft():
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model_sft = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # may need tweaking per architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model_sft = get_peft_model(model_sft, peft_config)

    training_config = SFTConfig(
        output_dir="checkpoints/terminology-sft",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        max_length=512,
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
    )

    train_dataset = Dataset.from_list(train_ds[:-10])
    eval_dataset = Dataset.from_list(train_ds[-10:])


# TODO: uncomment this when you have a training dataset
# trainer = SFTTrainer(
#     model=model_sft,
#     processing_class=tokenizer,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     args=training_config,
#     formatting_func=lambda x: x["text"],
# )

# trainer.train()

# After training, you can save and later load only the LoRA adapters.
# trainer.model.save_pretrained("checkpoints/terminology-sft-lora")


# preference optimisation setup (DPO example)
# for this, you will need to create or find a preference dataset (cf. Berger et al. 2025 on post-edits)

# toy dataset; replace with real preference data where
# each example has: prompt, chosen (better output), rejected (worse output).
# po_examples = [
#     {
#         "prompt": "Translate to Chinese using the given terminology.",
#         "chosen": "Better translation that respects the given terms.",
#         "rejected": "Worse translation that ignores or mistranslates the terms.",
#     }
# ]

def run_dpo(preference_dataset):

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    po_model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config = bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # may need tweaking per architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Reuse the same LoRA config as above
    po_model = get_peft_model(po_model, peft_config)

    dpo_config = DPOConfig(
        output_dir="checkpoints/terminology-dpo",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=1,
        max_length=512,
        beta=0.1,
        report_to="none",
        remove_unused_columns=False
    )

    dpo_trainer = DPOTrainer(
        model=po_model,
        ref_model=None,  # TRL will create a frozen reference copy by default
        args=dpo_config,
        train_dataset=preference_dataset,
        processing_class=tokenizer,
        # formatting_func=lambda x: x["text"],
    )

    dpo_trainer.train()
    adapter_save_path = "checkpoints/terminology-dpo-lora"
    dpo_trainer.model.save_pretrained(adapter_save_path)
    print(f"adherence reward adapters saved to: {adapter_save_path}")

    del po_model, dpo_trainer
    gc.collect()
    torch.cuda.empty_cache()
    
    return adapter_save_path

# TODO: uncomment this when you have a preference dataset

# dpo_trainer = DPOTrainer(
#     model=po_model,
#     ref_model=None,  # TRL will create a frozen reference copy by default
#     args=dpo_config,
#     train_dataset=po_ds,
#     processing_class=tokenizer,
#     # formatting_func=lambda x: x["text"],
# )

# dpo_trainer.train()

# dpo_trainer.model.save_pretrained("checkpoints/terminology-dpo-lora")

### rq4: reasoning
def generate_reasoning_translations(inputs, model_id, src_tgt_pair, batch_size=16):
    if "checkpoints" in model_id:
        tokenizer_path = "CohereLabs/tiny-aya-global"
    else:
        tokenizer_path = model_id    
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        device_map="auto"
    )

    src_key = src_tgt[src_tgt_pair]["src"]
    src_full = src_tgt[src_tgt_pair]["src_full"]
    tgt_full = src_tgt[src_tgt_pair]["tgt_full"]

    prompts = []
    for entry in inputs:
        source = entry.get(src_key, "")
        terminology = entry.get("terms", {})

        prompt = (
            f"You are an expert translator specialising in word sense disambiguation. \n"
            f"Task: translate the following sentence from {src_full} to {tgt_full} using the provided terminology constraints. \n\n"
            f"Source sentence: {source}\n"
            f"Terminology constraints: {terminology}\n\n"
            f"Follow these instructions:\n"
            f"Step 1: analyse the provided terminology constraints in the context of the source sentence. Identify the domain or field (e.g. legal, computing, medical, finance) this group of terms might belong to.\n"
            f"Step 2: provide the fluent translation in {tgt_full}, strictly maintaining the terminology constraints and take the identified domain into account."
            f"Format your output like this, strictly using these XML tags: \n"
            f"<reasoning>\n"
            f"Domain: \n"
            f"</reasoning>\n"
            f"<translation>"
            f"Translation: "
            f"</translation>"
        )
        prompts.append(prompt)
    
    messages_list = [[{"role": "user", "content": p}] for p in prompts]
    texts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_list
    ]

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_slice = texts[i:i+batch_size]

        model_inputs = tokenizer(
            batch_slice, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens = 512,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )
        
        for j in range(len(batch_slice)):
            input_ids_len = model_inputs.input_ids[j].shape[0]
            output_ids = generated_ids[j][input_ids_len:].tolist()
            raw_content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            translation = raw_content
            
            outputs.append(translation)
    
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return outputs

# generation, saving, and evaluation

def generate_and_save_translations(src_tgt_pair, setting, model_id, local=False):
    
    inputs = src_tgt[src_tgt_pair][setting][0:10]

    translations = generate_translations(inputs, model_id, src_tgt_pair)

    if local: 
        folder = "local"
    else:
        folder = "submissions"

    # TODO: change name to your team name, for easy evaluation in wmt25 format for track1
    output_path = f"wmt25-terminology/ranking/{folder}/track1/ROSSG/ROSSG.{src_tgt_pair}.{setting}.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        for input, translation in zip(inputs, translations):
            f.write(json.dumps({
                src_tgt[src_tgt_pair]["src"]: input[src_tgt[src_tgt_pair]["src"]].strip("\n"),
                "terms": input.get("terms", ""),
                src_tgt[src_tgt_pair]["tgt"]: translation.strip("\n"),
            }) + "\n")

    return translations

def generate_and_save_full_benchmark(model_id, strategy="baseline", local=True):
    languages = ["ende", "enes", "enru"]
    settings = ["noterm", "proper", "random"]

    folder = "local" if local else "submissions"

    print(f"\nstarting benchmark using strategy: {strategy.upper()}")
    team_name = f"ROSSG_{strategy.upper()}"
    
    start_total = time.time()

    for lang in languages:
        src_key = src_tgt[lang]["src"]
        tgt_key = src_tgt[lang]["tgt"]

        for setting in settings:
            print(f" processing split: {lang} | setting: {setting}")
            inputs = src_tgt[lang][setting]

            batch_size = 4 if lang == "enru" else 16

            if strategy == "code_switch":
                translations = generate_codeswitched_translations(inputs, model_id, lang, batch_size=batch_size)
            elif strategy == "intermediate_reasoning":
                translations = generate_reasoning_translations(inputs, model_id, lang, batch_size=batch_size)
            else:
                translations = generate_translations(inputs, model_id, lang, batch_size=batch_size)
            
            output_path = f"wmt25-terminology/ranking/{folder}/track1/{team_name}/{team_name}.{lang}.{setting}.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for input, translation in zip(inputs, translations):
                    f.write(json.dumps({
                        src_key: input[src_key].strip("\n"),
                        "terms": input.get("terms", {}),
                        tgt_key: translation.strip("\n"),
                    }, ensure_ascii=False) + "\n")
            
            print(f"saved {len(translations)} rows to {output_path}")
            del translations
            gc.collect()
            torch.cuda.empty_cache()
    
    end_total = time.time()
    print(f"\nfull benchmark complete! total execution time: {(end_total - start_total)}")

# you will need to run this for all 3 settings, and all 3 language pairs (9 overall for each modification/method)
# however, don't always run all 9 -- be sensible in what you evaluate and when.
# also TODO: process the test data (with references too) so that you can also evaluate your systems on the test set.
# but always make sure you only run test at the end; make hyperparameter decisions based on dev set performance!

# translations = generate_and_save_translations("ende", "proper", model_id)

# evaluation with predefined data

# run here or from the path wmt25-terminology/ranking/metric_track1 directly

# import subprocess
# import sys


# subprocess.run(
#     [sys.executable, "evaluate_qual_acc_track1.py"],
#     cwd="wmt25-terminology/ranking/metric_track1",
#     check=True
# )


# then run: `nlp2-26/wmt25-terminology/ranking/metric_track1/consistency_script_track1.py -s {src: en} -t {tgt: de/ru/es} -m {mode: noterm/random/proper}`

if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # Change this to the exact Gemma variant you want, e.g. "google/gemma-2-4b-it" or a local path.
    model_id = "CohereLabs/tiny-aya-global"  # example; replace with e.g. "gemma-3-4b-it" 

    print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1e6} MB")
    print("Retrieving translations using the baseline model")
    generate_and_save_full_benchmark(model_id, strategy="baseline", local=True)
    
    # print("\n      rq1 benchmark   \n")
    # test_slice = src_tgt["ende"]["proper"][:3]
    # cs_outputs = generate_codeswitched_translations(test_slice, model_id, "ende")

    # for i, item in enumerate(test_slice):
    #     print(f"--- Example {i+1} ---")
    #     print(f"Source Text (EN)  : {item['en']}")
    #     print(f"Target Dictionary : {item['terms']}")
    #     print(f"Code-Switched In  : {code_switch_source(item['en'], item['terms'])}")
    #     print(f"Model Output      : {cs_outputs[i]}\n")

    print("Retrieving translations using codeswitched translations")
    generate_and_save_full_benchmark(model_id, strategy="code_switch", local=True)
    
    print("Creating a preference dataset using adherence rewards")
    languages = ["ende", "enes", "enru"]
    settings = ["proper", "random"]
    combined_preference_records = []

    for lang in languages:
        for setting in settings:
            raw_data_slice = src_tgt[lang][setting]
            split_records = generate_preference_dataset(
                base_model=model_id,
                raw_data_list = raw_data_slice,
                src_tgt_pair=lang,
                max_examples = 30
            )
            combined_preference_records.extend(split_records)

    po_dataset = Dataset.from_list(combined_preference_records)
    reward_adapter_path = run_dpo(po_dataset)
    # reward_adapter_path = "checkpoints/terminology-dpo-lora"

    print("Retrieving translations using adherence rewards (RQ3)")
    generate_and_save_full_benchmark(model_id=reward_adapter_path, strategy="term_adherence", local=True)

    print("Retrieving translations using intermediate reasoning (RQ4)")

    # test_slice = src_tgt["ende"]["proper"][3:6]
    # reasoning_outputs = generate_reasoning_translations(test_slice, model_id, "ende")

    # for i, item in enumerate(test_slice):
    #     print(f"--- Example {i+1} ---")
    #     print(f"Source Text (EN)  : {item['en']}")
    #     print(f"Target Dictionary : {item['terms']}")
    #     print(f"Model Output      : {reasoning_outputs[i]}\n")
    generate_and_save_full_benchmark(model_id, strategy="intermediate_reasoning", local=False)