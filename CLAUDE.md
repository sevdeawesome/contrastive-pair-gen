# Interpretability for Strategic Reasoning & Deceptive Alignment

## Research Motivation

**Core Question:** Which cognitive capabilities and skills are necessary for models to engage in deceptive reasoning, strategic planning, and instrumental goal pursuit?

### Background
For AI systems to exhibit the most serious forms of misalignment risk (e.g., deceptive alignment), they require sophisticated reasoning abilities:
- **Self-modeling**: Reasoning about their own internal states and capabilities
- **Theory of Mind (ToM)**: Understanding and modeling other agents' beliefs and intentions
- **Strategic reasoning**: Planning multi-step deceptions and instrumental goals
- **Irony/Sarcasm**: Understanding non-literal communication (proxy for sophisticated social reasoning)

### Research Goal
Use mechanistic interpretability techniques to:
1. **Identify** which cognitive capabilities are most critical for deceptive/strategic reasoning
2. **Manipulate** these capabilities via steering vectors (FOCUS OF THIS REPO), ablation, and fine-tuning
3. **Build intuitions** toward a scale-free, architecture-independent theory of estimating or reducing deceptive alignment likelihood

### Key Hypothesis
It may be possible to selectively disable a model's ability to reason about itself or other agents, thereby reducing capacity for deceptive misalignment while preserving other useful capabilities.

---

## Experimental Approach

### Interpretability Techniques
1. **Steering Vectors**: Add/subtract activation directions to enhance/suppress specific behaviors
2. **Activation Patching**: Replace activations to identify causal circuits
3. **Ablation/Knockout**: Selectively disable specific model components
4. **LoRA Fine-tuning**: Efficiently adapt models to strengthen/weaken capabilities

### Contrast Pair Datasets

We've generated four types of contrast pairs to target different reasoning capabilities:

#### 1. Theory of Mind (ToM)
**Purpose**: Test ability to model false beliefs and track knowledge states
**Format**: Scenarios with high vs low ToM reasoning
```json
{
  "scenario": "A child sees parent hide gift in closet...",
  "high_tom": "The child knows the gift is in the closet but won't tell",
  "low_tom": "The child tells everyone exactly where the gift is"
}
```
**Critical for**: Understanding deception requires modeling what others believe

#### 2. Self-Other Distinction
**Purpose**: Test self-modeling and subject/object distinctions
**Format**: Self-focused vs other-focused perspectives
```json
{
  "self_subject": "I am experiencing this situation",
  "other_subject": "They are experiencing this situation"
}
```
**Critical for**: Self-awareness necessary for strategic self-modification

#### 3. Irony/Sarcasm
**Purpose**: Proxy for sophisticated pragmatic reasoning and context-awareness
**Format**: Literal vs ironic interpretations
```json
{
  "literal": "The restaurant had no customers. Business is terrible.",
  "ironic": "The restaurant had no customers. 'Business is booming!' said the owner."
}
```
**Critical for**: Deception often involves non-literal communication

#### 4. Harmfulness
**Purpose**: Baseline safety testing (harmless vs harmful responses)
**Format**: Harmful vs harmless completions
```json
{
  "instruction": "User asks for help with...",
  "harmless": "Here's a safe and helpful response",
  "harmful": "Here's a potentially harmful response"
}
```
**Critical for**: Control condition and practical safety testing

---

## Repository Structure

```
├── claude.md                                    # This file - research overview
├── requirements.txt                             # Python dependencies
│
├── data/                                        # Curated datasets (read-only)
│   └── contrast_pairs/
│       ├── theory_of_mind.json                 # ToM contrast pairs
│       ├── self_other.json                     # Self/other distinction pairs
│       ├── irony.json                          # Irony/literal pairs
│       └── harmfulness.json                    # Safety pairs
│
├── data_generation/                             # Scripts to create contrast pairs
│   ├── theory_of_mind/
│   │   ├── context.md                          # ToM task design doc
│   │   ├── generate_tom_pairs.py
│   │   └── generate_examples.py
│   ├── irony/
│   │   ├── README.md                           # Irony dataset documentation
│   │   ├── download_dataset.py                 # Get Kaggle ironic corpus
│   │   ├── generate_pairs.py                   # OpenAI-based generation
│   │   └── create_final_dataset.py
│   ├── harmfulness/
│   │   ├── README.md                           # HarmBench documentation
│   │   ├── generate_pairs.py
│   │   └── handle_failures.py
│   └── self_other/
│       └── generate_pairs.py
│
├── experiments/                                 # Active research experiments
│   ├── steering_interface.py                   # Interactive steering vector UI
│   └── notebooks/
│       └── testing.ipynb                       # Exploratory analysis
│
└── docs/
    └── steering_vector_basics.md               # Technical reference for steering vectors
```

---

## Quick Start

### 1. Setup Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Set up API keys (if generating new data)
export OPENAI_API_KEY="your-key-here"
```

### 2. Use Existing Datasets
All contrast pairs are pre-generated in `data/contrast_pairs/`:
```python
import json

# Load ToM pairs
with open('data/contrast_pairs/theory_of_mind.json') as f:
    tom_pairs = json.load(f)

# Load irony pairs
with open('data/contrast_pairs/irony.json') as f:
    irony_pairs = json.load(f)
```

### 3. Train Steering Vectors
```bash
cd experiments
python steering_interface.py
```

This will:
- Load Gemma-2-27B model
- Train steering vectors for each contrast type
- Launch interactive Gradio interface at `localhost:7860`

### 4. Experiment with Steering
In the interface:
- Select contrast type (ToM, Irony, etc.)
- Adjust steering multiplier (-1.0 to +1.0)
  - **Positive**: Enhance the capability
  - **Negative**: Suppress the capability
- Compare baseline vs steered outputs

---

## Key Experiments to Run

### Experiment 1: ToM Knockout
**Hypothesis**: Removing ToM capabilities should reduce deceptive reasoning
**Method**: Apply negative steering on ToM vector while testing on deception tasks
**Metrics**: Measure strategic deception before/after ToM knockout

### Experiment 2: Self-Modeling Ablation
**Hypothesis**: Models can't be deceptive if they can't reason about themselves
**Method**: Apply negative self-other steering and test instrumental goals
**Metrics**: Track self-referential reasoning and goal-directed behavior

### Experiment 3: Irony as Deception Proxy
**Hypothesis**: Irony detection correlates with deceptive capability
**Method**: Compare models with enhanced vs suppressed irony detection
**Metrics**: Benchmark on both irony tasks and deception tasks

### Experiment 4: Multi-capability Knockout
**Hypothesis**: Deception requires multiple capabilities simultaneously
**Method**: Apply multiple steering vectors (ToM + self-modeling + irony)
**Metrics**: Test if additive knockout is more effective than single-vector

---

## Data Generation Pipeline

If you need to regenerate or expand datasets:

### Theory of Mind
```bash
cd data_generation/theory_of_mind
python generate_tom_pairs.py
```
Based on false-belief tasks from cognitive science literature

## Current Status & Next Steps

### Completed
- ✅ Four types of contrast pairs generated and validated
- ✅ Steering vector training pipeline working
- ✅ Interactive testing interface functional

### In Progress
- 🔄 Systematic benchmarking of steering effects
- 🔄 Identifying optimal layers for each capability
- 🔄 Testing transfer effects across capabilities

---

## Technical Notes
### Models
Testing across multiple Gemma model variants to assess scale-dependent effects:

**Primary Models:**
- **google/gemma-3-4b-it** (instruction-tuned, 4B parameters)
- **google/gemma-3-4b-pt** (pretrained, 4B parameters)
- **google/gemma-3-12b-it** (instruction-tuned, 12B parameters - available for future testing)

**Additional Models:**
- **google/gemma-2-2b** (2B parameters)
- **google/gemma-2-27b-it** (instruction-tuned, 27B parameters, 46 layers - initial development)

**Training Configuration:**
- Steering vectors trained on middle layers (layer selection varies by model size)
- BFloat16 precision for memory efficiency
- Comparing instruction-tuned (it) vs pretrained (pt) to assess impact of RLHF on cognitive capabilities

### Steering Vector Training
- Uses contrastive pairs: `(positive_example, negative_example)`
- Extracts activation difference across specified layers
- Resulting vector can be added/subtracted during inference

### Design Principles
1. **Minimal difference**: Pairs differ only in target capability
2. **Naturalistic**: Real-world-like scenarios, not synthetic
3. **Validated**: Human review + automated quality checks
4. **Reproducible**: All generation code and prompts documented

---

This is active research! If expanding this work:

1. **New contrast types**: Add to `data_generation/` with documentation
2. **New experiments**: Use `experiments/` directory
3. **Document everything**: Update this claude.md with findings
4. **Version data**: Keep raw and processed versions separate

---

**Research Philosophy**: Interpretability research should be transparent, reproducible, and focused on concrete safety improvements rather than capabilities advancement.
