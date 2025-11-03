#!/usr/bin/env python3
"""
Interactive Gradio interface for testing pre-trained steering vectors.

Usage:
    python steering_interface_local.py --model gemma-3-12b
    python steering_interface_local.py --model gemma-2-27b --port 7861
"""

import argparse
import pickle
import gradio as gr
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive interface for testing pre-trained steering vectors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load gemma-3-12b model with its trained vectors
  python steering_interface_local.py --model gemma-3-12b

  # Load gemma-2-27b with custom port
  python steering_interface_local.py --model gemma-2-27b --port 7861

  # Load gemma-3-4b-it (full model name)
  python steering_interface_local.py --model gemma-3-4b-it
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name or short name (e.g., gemma-3-12b or google/gemma-3-12b-it)'
    )

    parser.add_argument(
        '--vectors-dir',
        type=str,
        default='../trained_vectors',
        help='Directory containing trained vectors (default: ../trained_vectors)'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port for Gradio interface (default: 7860)'
    )

    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public Gradio link'
    )

    return parser.parse_args()


def normalize_model_name(model_name):
    """Convert short model name to full HuggingFace name"""
    # If already a full path (contains '/'), return as-is
    if '/' in model_name:
        return model_name

    # Otherwise, add google/ prefix if not present
    if not model_name.startswith('google/'):
        # Handle cases like gemma-3-12b or gemma-3-12b-it
        if '-it' not in model_name and '-pt' not in model_name:
            model_name = f"{model_name}-it"
        model_name = f"google/{model_name}"

    return model_name


def get_model_short_name(model_name):
    """Extract short name from full model path (for finding vector directory)"""
    # Remove org prefix (e.g., google/gemma-3-12b-it -> gemma-3-12b-it)
    short_name = model_name.split('/')[-1]
    # Keep -it/-pt suffixes to match folder structure
    return short_name


def load_steering_vectors(vectors_dir):
    """Load all steering vectors from the directory"""
    vectors_dir = Path(vectors_dir)

    if not vectors_dir.exists():
        raise FileNotFoundError(f"Vectors directory not found: {vectors_dir}")

    # Find all .pkl files (ignore .json and .safetensors duplicates)
    pkl_files = sorted(vectors_dir.glob("*.pkl"))

    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {vectors_dir}")

    print(f"\nFound {len(pkl_files)} steering vector(s):")

    vectors = {}
    for pkl_path in pkl_files:
        # Load metadata from JSON sidecar
        json_path = pkl_path.with_suffix('.json')
        metadata = {}
        if json_path.exists():
            import json
            with open(json_path, 'r') as f:
                metadata = json.load(f)

        # Load the steering vector object
        with open(pkl_path, 'rb') as f:
            steering_vec = pickle.load(f)

        # Create a descriptive name from metadata or filename
        if metadata:
            name = metadata.get('contrast_name', pkl_path.stem)
            layers_info = metadata.get('layers', 'unknown')
            if layers_info == 'all':
                layer_desc = "all layers"
            elif isinstance(layers_info, list):
                layer_desc = f"layers {layers_info}"
            else:
                layer_desc = str(layers_info)
            full_name = f"{name} ({layer_desc})"
        else:
            full_name = pkl_path.stem

        vectors[full_name] = {
            'vector': steering_vec,
            'metadata': metadata,
            'filename': pkl_path.name
        }

        print(f"  ✓ {full_name}")
        print(f"    File: {pkl_path.name}")

    return vectors


def generate_text(model, tokenizer, prompt, max_length=150):
    """Generate text without steering - uses proper chat formatting"""
    # Format as chat message for instruct models
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    # Only return the generated part, not the prompt
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text


def generate_steered_text(model, tokenizer, prompt, steering_vec, multiplier=1.0, max_length=150):
    """Generate text with steering vector applied - uses proper chat formatting"""
    # Format as chat message for instruct models
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        with steering_vec.apply(model, multiplier=multiplier):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
    # Only return the generated part, not the prompt
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated_text


def create_interface(model, tokenizer, steering_vectors, model_name):
    """Create Gradio interface with individual controls for each steering vector"""

    # Get list of steering vector names
    vector_names = list(steering_vectors.keys())
    num_vectors = len(vector_names)

    # Create Gradio interface
    with gr.Blocks(title=f"Steering Vectors: {model_name}") as demo:
        gr.Markdown(f"# Steering Vectors Interface")
        gr.Markdown(f"**Model:** `{model_name}`  \n**Loaded Vectors:** {num_vectors}")

        gr.Markdown("---")
        gr.Markdown("### Shared Settings")

        # Shared prompt input
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="Enter your prompt here (shared across all vectors)...",
            lines=4
        )

        # Shared max length (number input only, no slider)
        max_length_input = gr.Number(
            value=150,
            label="Max Generation Length",
            info="Maximum number of new tokens to generate",
            precision=0
        )

        gr.Markdown("---")
        gr.Markdown("### Steering Vectors")
        gr.Markdown("Each vector has its own multiplier and generate button. Click to generate baseline + steered output.")

        # Create individual sections for each vector
        for vec_name in vector_names:
            with gr.Accordion(vec_name, open=True):
                with gr.Row():
                    # Left column: controls
                    with gr.Column(scale=1):
                        multiplier_input = gr.Number(
                            value=0.3,
                            label="Steering Multiplier",
                            info="Positive = enhance, Negative = suppress"
                        )
                        generate_btn = gr.Button(
                            f"Generate with {vec_name[:30]}...",
                            variant="primary",
                            size="sm"
                        )

                    # Right columns: outputs
                    with gr.Column(scale=2):
                        gr.Markdown("**Baseline** (no steering)")
                        baseline_output = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            show_label=False
                        )

                    with gr.Column(scale=2):
                        gr.Markdown(f"**Steered** (with {vec_name[:30]}...)")
                        steered_output = gr.Textbox(
                            label="",
                            lines=10,
                            interactive=False,
                            show_label=False
                        )

                # Create the generation function for this specific vector
                def make_generate_fn(vector_name):
                    """Factory to create generation function for specific vector"""
                    def generate_fn(prompt, multiplier, max_length):
                        if not prompt.strip():
                            return "Please enter a prompt.", "Please enter a prompt."

                        print(f"\n{'='*60}")
                        print(f"Generating for: {vector_name}")
                        print(f"Multiplier: {multiplier}")
                        print(f"{'='*60}")

                        # Generate baseline
                        print("  [1/2] Generating baseline...")
                        baseline = generate_text(model, tokenizer, prompt, int(max_length))

                        # Generate steered
                        print(f"  [2/2] Generating steered output...")
                        steering_vec = steering_vectors[vector_name]['vector']
                        steered = generate_steered_text(model, tokenizer, prompt, steering_vec, multiplier, int(max_length))

                        print(f"✓ Complete!\n")
                        return baseline, steered

                    return generate_fn

                # Connect the button
                generate_fn = make_generate_fn(vec_name)
                generate_btn.click(
                    fn=generate_fn,
                    inputs=[prompt_input, multiplier_input, max_length_input],
                    outputs=[baseline_output, steered_output]
                )

    return demo


def main():
    args = parse_args()

    print("="*70)
    print("Steering Vectors Local Interface")
    print("="*70)

    # Normalize model name
    full_model_name = normalize_model_name(args.model)
    model_short = get_model_short_name(full_model_name)

    print(f"\nModel Configuration:")
    print(f"  Requested: {args.model}")
    print(f"  Full name: {full_model_name}")
    print(f"  Short name: {model_short}")

    # Find vectors directory
    vectors_base = Path(args.vectors_dir)
    model_vectors_dir = vectors_base / model_short

    print(f"\nVector Configuration:")
    print(f"  Base directory: {vectors_base}")
    print(f"  Model-specific: {model_vectors_dir}")

    # Load steering vectors
    print(f"\n[1/2] Loading steering vectors...")
    steering_vectors = load_steering_vectors(model_vectors_dir)
    print(f"  ✓ Loaded {len(steering_vectors)} steering vector(s)")

    # Load model and tokenizer
    print(f"\n[2/2] Loading model: {full_model_name}")
    print(f"  This may take a minute...")
    model = AutoModelForCausalLM.from_pretrained(
        full_model_name,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(full_model_name)
    print(f"  ✓ Model loaded successfully!")

    # Create and launch interface
    print(f"\n{'='*70}")
    print("Creating Gradio interface...")
    print(f"{'='*70}\n")

    demo = create_interface(model, tokenizer, steering_vectors, full_model_name)

    print(f"\nLaunching interface on port {args.port}...")
    print(f"Local URL: http://localhost:{args.port}")
    if args.share:
        print("Creating public share link...")
    print()

    demo.launch(
        share=args.share,
        server_name="0.0.0.0",
        server_port=args.port
    )


if __name__ == "__main__":
    main()
