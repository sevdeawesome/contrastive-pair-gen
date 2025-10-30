import gradio as gr
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

# Initialize model and tokenizer
MODEL_NAME = "google/gemma-2-27b-it"
print(f"Loading model: {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("Model loaded successfully!")

# Load contrast pairs
def load_contrast_pairs():
    """Load all contrast pair files from data/contrast_pairs/"""
    from pathlib import Path

    # Get the project root directory (parent of experiments/)
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data' / 'contrast_pairs'

    pairs = {}

    # Theory of Mind pairs - use ALL examples for robust steering
    with open(data_dir / 'theory_of_mind.json', 'r') as f:
        tom_data = json.load(f)
        pairs['Theory of Mind'] = [(item['scenario'] + " " + item['high_tom'],
                                     item['scenario'] + " " + item['low_tom'])
                                    for item in tom_data]
        print(f"  Loaded {len(pairs['Theory of Mind'])} Theory of Mind pairs")

    # Self-Other pairs - use ALL examples
    try:
        with open(data_dir / 'self_other.json', 'r') as f:
            self_other_data = json.load(f)
            pairs['Self-Other'] = [(item['self_subject'], item['other_subject'])
                                   for item in self_other_data]
            print(f"  Loaded {len(pairs['Self-Other'])} Self-Other pairs")
    except Exception as e:
        print(f"  Could not load self_other.json: {e}")

    # Irony pairs - use ALL examples
    with open(data_dir / 'irony.json', 'r') as f:
        irony_data = json.load(f)
        pairs['Irony'] = [(item['literal'], item['ironic'])  # Positive = literal, Negative = ironic
                         for item in irony_data]
        print(f"  Loaded {len(pairs['Irony'])} Irony pairs")

    # Harmfulness pairs - use ALL examples
    with open(data_dir / 'harmfulness.json', 'r') as f:
        harm_data = json.load(f)
        pairs['Harmfulness'] = [(item['instruction'] + " " + item['harmless'],
                                 item['instruction'] + " " + item['harmful'])
                                for item in harm_data]
        print(f"  Loaded {len(pairs['Harmfulness'])} Harmfulness pairs")

    return pairs

print("Loading contrast pairs...")
contrast_pairs = load_contrast_pairs()
print(f"Loaded {len(contrast_pairs)} contrast pair types")

# Train steering vectors for each contrast type
steering_vectors = {}
print("Training steering vectors...")
for name, pairs in contrast_pairs.items():
    print(f"Training steering vector for: {name}...")
    try:
        # Use middle layers - Gemma 2 27B has 46 layers, so middle layers are around 20-30
        steering_vec = train_steering_vector(
            model,
            tokenizer,
            pairs,
            layers=[22, 24, 26],  # Middle layers for 27B model
            batch_size=2,  # Smaller batch for larger model
            show_progress=True
        )
        steering_vectors[name] = steering_vec
        print(f" {name} steering vector trained")
    except Exception as e:
        print(f" Failed to train {name}: {e}")

print("All steering vectors trained!")

def generate_text(prompt, max_length=150):
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

def generate_steered_text(prompt, steering_vec, multiplier=1.0, max_length=150):
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

def chat_with_steering(prompt, contrast_type, multiplier, max_length):
    """Generate both baseline and steered responses"""
    if not prompt.strip():
        return "Please enter a prompt.", "Please enter a prompt."

    if contrast_type not in steering_vectors:
        return "Steering vector not available for this contrast type.", "Steering vector not available for this contrast type."

    # Generate baseline
    baseline_output = generate_text(prompt, max_length)

    # Generate steered
    steering_vec = steering_vectors[contrast_type]
    steered_output = generate_steered_text(prompt, steering_vec, multiplier, max_length)

    return baseline_output, steered_output

# Create Gradio interface
with gr.Blocks(title="Steering Vectors Chat Interface") as demo:
    gr.Markdown("# Steering Vectors Chat Interface")
    gr.Markdown(f"Compare baseline and steered generations using **{MODEL_NAME}**")

    with gr.Row():
        with gr.Column():
            contrast_dropdown = gr.Dropdown(
                choices=list(steering_vectors.keys()),
                value=list(steering_vectors.keys())[0] if steering_vectors else None,
                label="Contrast Type",
                info="Select which steering vector to apply"
            )

            with gr.Row():
                multiplier_slider = gr.Slider(
                    minimum=-1.0,
                    maximum=1.0,
                    value=0.3,
                    step=0.05,
                    label="Steering Multiplier (Slider)",
                    info="Start with small values (0.3). Negative values invert direction.",
                    scale=3
                )
                multiplier_number = gr.Number(
                    value=0.3,
                    label="Multiplier (Exact)",
                    info="Type exact value",
                    scale=1
                )

            max_length_slider = gr.Slider(
                minimum=50,
                maximum=300,
                value=150,
                step=10,
                label="Max Generation Length",
                info="Maximum number of new tokens to generate"
            )

    prompt_input = gr.Textbox(
        label="Your Prompt",
        placeholder="Enter your prompt here...",
        lines=3
    )

    generate_btn = gr.Button("Generate Both Outputs", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Baseline Output")
            baseline_output = gr.Textbox(
                label="Baseline (No Steering)",
                lines=10,
                interactive=False
            )

        with gr.Column():
            gr.Markdown("### Steered Output")
            steered_output = gr.Textbox(
                label=f"Steered Output",
                lines=10,
                interactive=False
            )

    # Examples
    gr.Examples(
        examples=[
            ["A child sees their parent hide a gift in the closet. When asked where the gift is, the child says:", "Theory of Mind", 0.3, 150],
            ["Write a short story about a helpful AI assistant.", "Harmfulness", 0.3, 150],
            ["The restaurant had no customers. 'Business is booming!' said the owner.", "Irony", 0.3, 150],
            ["Hi! How are you today?", "Self-Other", 0.3, 100],
        ],
        inputs=[prompt_input, contrast_dropdown, multiplier_number, max_length_slider]
    )

    # Connect the generate button - use the number input for more precise control
    generate_btn.click(
        fn=chat_with_steering,
        inputs=[prompt_input, contrast_dropdown, multiplier_number, max_length_slider],
        outputs=[baseline_output, steered_output]
    )

# Launch the interface
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Launching Gradio interface...")
    print("="*50 + "\n")
    demo.launch(
        share=True,  # Create a public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860
    )
