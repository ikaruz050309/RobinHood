import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
def generate_llm_explication(importance_map, final_predictions, data):
    # Fastest hardware component detection
    device = "cuda" if torch.cuda.is_available else ("mps" if torch.backends.mps.is_available else "cpu")
    # We will now download the LLM Meta's Muse glimmer
    meta_model = "meta-llama/Muse-Glimmer"
    tokenizer = AutoTokenizer.from_pretrained(meta_model)
    model = AutoModelForCausalLM.from_pretrained(
        meta_model,
        torch_dtype = torch.float16 if device != "cpu" else torch.float32,
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
    Write a concise 3-sentence maximum breakdown for the user:
    1. Summarize what the model predicts based on the recent raw historical context.
    2. Explain which specific features dominated this choice according to Captum's mathematical weights.
    3. Validate why this prediction can be trusted by linking the raw input data to the mathematical proofs.

    [EXPLANATION]:"""
    # we're going to translate the prompt into tokens for the neurones
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # We launch the generation of the response using the meta model
        outputs = model.generate(
            **inputs,
            max_new_tokens = 180, # to ensure that the answer is limited to 180 words
            temperature=0.2, # in order to avoid hallucinations and to ensure the explication is as close to the actual figures as possible
            do_sample=True,
            repetition_penalty= 1.2
        )
    # isolation and transcription of the actual text
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    explanation = full_text.split("[EXPLANATION]:")[-1].strip()
    return explanation
