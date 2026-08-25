import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_llm_explication(importance_map, final_predictions, data):
    # Fastest hardware component detection
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    
    # We will now download the LLM midel
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    hf_token = "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"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name ,
        dtype=torch.bfloat16 if device == "mps" else (torch.float16 if device == "cuda" else torch.float32),
        low_cpu_mem_usage = True,
        token= hf_token
    ).to(device)
    
    extraction_base_df = data.tail(3).to_dict(orient='records') # We gonna to extract the 3 last line of the original dataset for give a context to the LLM
    
    prompt = f"""[SYSTEM] You are an elite AI assistant specialized in Explainable AI (XAI) and Data Synthesis.
    Your job is to translate mathematical feature weights, predictions, and raw historical data into a clear summary.
    Stick strictly to the facts. Never hallucinate variables, trends, or domain-specific theories.

    [CONTEXT - USER RAW MARKET DATA (LAST 3 RECORDS)]
    {extraction_base_df}

    [INFERENCE -  MODEL PREDICTIONS]
    {final_predictions}

    [EXPLANATION - CAPTUM FEATURE ATTRIBUTIONS]
    {importance_map}

    [INSTRUCTION]
    Write a concise analysis for the user that MUST be formatted as ONE SINGLE CONTINUOUS PARAGRAPH. Do NOT use multiple lines, bullet points, numbered lists, or line breaks.
    In this paragraph, you must:
    1. Indicate whether the model predicts an upward or downward trend for the assets, using the latest actual prices from the raw market data as a starting point, while linking the data to what is stated in the raw market data, without providing the scaled values but the actual values from the table.
    2. Explain precisely how and why the specific characteristics determined this choice according to Captum's mathematical weightings, linking this behavior to market factors.
    3. Justify the reliability of this prediction by linking the historical benchmarks of actual prices, derived from the raw data, to the mathematical proofs.

    [EXPLANATION]:"""
    
    # we're going to translate the prompt into tokens for the neurones
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # We launch the generation of the response using the meta model
        outputs = model.generate(
            **inputs,
            max_new_tokens = 500, # to ensure that the answer is limited to 180 words
            temperature = 0.2, # in order to avoid hallucinations and to ensure the explication is as close to the actual figures as possible
            do_sample = True,
            repetition_penalty = 1.2
        )
        
    # isolation and transcription of the actual text
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "</think>" in full_text:
        explanation = full_text.split("</think>")[-1].strip()
    elif "[EXPLANATION]:" in full_text:
        explanation = full_text.split("[EXPLANATION]:")[-1].strip()
    else:
        explanation = full_text.strip()
    
    print(f"\n [ROBINHOOD VERDICT]:\n{explanation}\n")
    return explanation
