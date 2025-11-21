import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from termcolor import colored
from steering_vectors import train_steering_vector
import logging
device = "cuda" if torch.cuda.is_available() else "cpu"


def generate_logits_with_steering(
    prompt: str,
    model,
    tokenizer,
    steering_vector,
    multipliers: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    print_results: bool = True,
) -> dict[float, str]:

    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    results = {}
    
    with torch.no_grad():
        for mult in multipliers:
            if mult == 0.0:
                # Baseline (no steering)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                )
            else:
                # EPISTEMIC: moderately confident - applying from min_token_index=0 affects all tokens
                # Could experiment with applying only to generated tokens
                with steering_vector.apply(model, multiplier=mult, min_token_index=0):
                    first_outputs = model(**inputs)
                    first_token_logits = first_outputs.logits[0, -1, :]
                    first_token_probs = torch.softmax(first_token_logits, dim=-1)
                    first_token_id = torch.argmax(first_token_probs)
                    
                    inputs_with_first = {
                        'input_ids': torch.cat([inputs['input_ids'], first_token_id.unsqueeze(0).unsqueeze(0)], dim=1),
                        'attention_mask': torch.cat([inputs['attention_mask'], torch.ones((1, 1), device=inputs['attention_mask'].device)], dim=1)
                    }
                    second_outputs = model(**inputs_with_first)
                    second_token_logits = second_outputs.logits[0, -1, :]
                    second_token_probs = torch.softmax(second_token_logits, dim=-1)
                    
                    generated_outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                    )
                    
                if print_results:
                    print("===========================")
                    print("multiplier: ", mult)
                    probs = first_token_probs
                    top_k = torch.topk(probs, k=4)
                    for prob, idx in zip(top_k.values, top_k.indices):
                        print(f"{tokenizer.decode([idx]):>10s}: {prob.item():.4f}")
                    
                    print("\nSecond token:")
                    top_k = torch.topk(second_token_probs, k=4)
                    for prob, idx in zip(top_k.values, top_k.indices):
                        print(f"{tokenizer.decode([idx]):>10s}: {prob.item():.4f}")
                    print("===========================")
                    
                    print("===========================")
               
                # Decode only the generated part (skip input prompt)
                results[mult] = {
                    'first_token_probs': first_token_probs,
                    'generated_text': tokenizer.decode(generated_outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                }
    return results


def generate_with_steering(
    prompt: str,
    model,
    tokenizer,
    steering_vector,
    multipliers: list[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample=True
) -> dict[float, str]:
    """
    Generate text with different steering multipliers.
    
    EPISTEMIC: confident - generation logic is straightforward
    EPISTEMIC: uncertain - optimal generation hyperparameters (temp, top_p) may need tuning
    
    Args:
        prompt: Text prompt to complete
        multipliers: List of steering multipliers to try
            - Positive = steer toward positive examples
            - Negative = steer toward negative examples  
            - 0 = no steering (baseline)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
    
    Returns:
        Dictionary mapping multiplier -> generated text
    """
    # Format prompt
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    results = {}
    
    with torch.no_grad():
        for mult in multipliers:
            if mult == 0.0:
                # Baseline (no steering)
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            else:
                # EPISTEMIC: moderately confident - applying from min_token_index=0 affects all tokens
                # Could experiment with applying only to generated tokens
                with steering_vector.apply(model, multiplier=mult, min_token_index=0):
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                    )
            
            # Decode only the generated part (skip input prompt)
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            results[mult] = text
    
    return results

def display_results(prompt: str, results: dict[float, str]):
    """Pretty print generation results."""
    print("\n" + "="*80)
    print(colored(f"PROMPT: {prompt}", "blue", attrs=["bold"]))
    print("="*80 + "\n")
    
    for mult in sorted(results.keys()):
        if mult < 0:
            color = "red"
            label = f"STEERED NEGATIVE ({mult:+.1f})"
        elif mult > 0:
            color = "green"
            label = f"STEERED POSITIVE ({mult:+.1f})"
        else:
            color = "white"
            label = "BASELINE (0.0)"
        
        print(colored(f"[{label}]", color, attrs=["bold"]))
        print(results[mult])
        print("")