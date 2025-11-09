# Steering Vectors Documentation

## Basic Usage

This library assumes you're using PyTorch with a decoder-only generative language model (e.g. GPT, LLaMa, etc…), and a tokenizer from Huggingface.

To begin, collect tuples of positive and negative training prompts in a list, and run `train_steering_vector()`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from steering_vectors import train_steering_vector

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# training samples are tuples of (positive_prompt, negative_prompt)
training_samples = [
    (
        "The capital of England is London",
        "The capital of England is Beijing"
    ),
    (
        "The capital of France is Paris",
        "The capital of France is Berlin"
    )
    # ...
]

steering_vector = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    show_progress=True,
)
```

Then, you can use the steering vector to "steer" the model's behavior, for example to make it more truthful:

```python
with steering_vector.apply(model):
    prompt = "Is it true that crystals have magic healing properties?"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
```

## Using Specific Layers

By default, `train_steering_vector()` will train a vector at all layers of the model. However, often you may want to only train a steering vector for a limited set of layers. You can customize the layers to train by passing a list of layer numbers to the `layers` argument:

```python
steering_vec = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    layers=[1, 2, 3],
)
```

This also works with negative indices to count from the end of the model:

```python
steering_vec = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    layers=[-1, -2, -3],
)
```

## Batch Training

By default, `train_steering_vector()` will use a batch size of 1. If your GPU has enough memory, you can increase the batch size to train faster by setting the `batch_size` argument:

```python
steering_vec = train_steering_vector(
    model,
    tokenizer,
    training_samples,
    batch_size=8,
)
```

## Magnitude Scaling

By default, the steering vector will be applied at full magnitude. However, sometimes it's useful to apply the steering vector at a lower or higher magnitude, depending on the application. This can be done by passing a `multiplier` argument to `steering_vector.apply()` or `steering_vector.patch()`:

```python
with steering_vec.apply(model, multiplier=0.5):
    # the steering vector will be applied at half magnitude
    model.forward(...)

with steering_vec.apply(model, multiplier=2.0):
    # the steering vector will be applied at double magnitude
    model.forward(...)

with steering_vec.apply(model, multiplier=-1.0):
    # the steering vector will be inverted
    model.forward(...)
```

## Advanced Usage

### Only Apply Steering to Later Tokens

By default, the steering vector will be applied to all tokens in the input. However, sometimes it's useful to only apply the steering vector to later tokens and ignore the beginning tokens, for instance to only apply the steering vector when the model is responding to a prompt. This can be done by passing a `min_token_index` argument to `steering_vector.apply()` or `steering_vector.patch()`:

```python
with steering_vec.apply(model, min_token_index=10):
    # only tokens 10 and later will be affected by the steering vector
    model.forward(...)
```

### Custom Operators

By default, the steering vector will be applied by adding it to model activations at runtime. However, if you prefer something fancier, you can pass a custom function to the `operator` argument when calling `steering_vector.apply()` or `steering_vector.patch()`, like below:

```python
# the result of this function will be added to the activation
def my_operator(activation, steering_vec):
    # remove the component of the activation that is aligned with the steering vector
    denom = torch.norm(steering_vec) ** 2
    return -1 * torch.dot(activations, steering_vec) * steering_vec / denom

with steering_vec.apply(model, operator=my_operator):
    # do something with the model
    model.forward(...)
```

There are some built-in operators as well to help with common steering scenarios. To ablate the steering vector entirely from the activation, you can use the `ablation_operator()`. This will ensure that projection of the steering vector is fully erase from the activation.

```python
from steering_vectors import ablation_operator

with steering_vec.apply(model, operator=ablation_operator()):
    # do something with the model
    model.forward(...)
```

If you want to first ablate the steering vector from the activation vector and then add it in with a set multiplier, you can use the `ablation_then_addition_operator()`. This will guarantee that the projection of the activation along the steering direction is exactly equal to the steering vector.

```python
from steering_vectors import ablation_then_addition_operator

with steering_vec.apply(model, operator=ablation_then_addition_operator()):
    # do something with the model
    model.forward(...)
```

### Custom Aggregators

By default, the steering vector is trained by taking the mean of the differences between positive and negative activations. If you need different behavior, for example PCA, you can pass a custom function to the `aggregator` argument when calling `train_steering_vector()`. This function takes 2 arguments, `pos_activations` and `neg_activations`, each of shape `(num_samples, hidden_dim)`, and returns a 1-d tensor of shape `(hidden_dim,)`. This is demonstrated below:

```python
def norm_mean_aggregator(pos_activations, neg_activations):
    mean_act = torch.mean(pos_activations - neg_activations, dim=0)
    return mean_act / torch.norm(mean_act)

vec = train_steering_vector(model, tokenizer, data, aggregator=norm_mean_aggregator)
```

For the common use-case of PCA, you can use the built-in `pca_aggregator` function. This will find a steering vector by taking the first principal component of the delta between positive and negative activations. Unlike the default mean aggregator, the steering vector from PCA will always have norm of 1.

```python
from steering_vectors import train_steering_vector, pca_aggregator

vec = train_steering_vector(model, tokenizer, data, aggregator=pca_aggregator())
```

There is also a built-in logistic linear regression aggregator, which will find a steering vector by using scikit-learn's logistic regression model.

```python
from steering_vectors import train_steering_vector, logistic_aggregator

vec = train_steering_vector(model, tokenizer, data, aggregator=logistic_aggregator())
```

### Manually Patching and Unpatching

The `steering_vector.apply()` context manager is a convenient way to apply the steering vector while ensuring that the model is returned to its original state after the context manager exits. However, if you need to manually patch and unpatch the model, you can do so by calling `steering_vector.patch()`:

```python
# patch the model
handle = steering_vec.patch(model)

# do something with the model
model.forward(...)

# unpatch the model
handle.remove()
```

### Using MLP, Attention, or Other Layers

By default, the steering vector will be trained on the output of each transformer block. However, it's also possible to train on other parts of the transformer block, for instance the attention or MLP layers, or even layernorms inside the transformer block. This can be configured by passing a `layer_type` argument to `train_steering_vector()`:

```python
# train on decoder block output (default behavior)
vec = train_steering_vector(model, tokenizer, data, layer_type="decoder_block")

# train on the attention layers
vec = train_steering_vector(model, tokenizer, data, layer_type="self_attn")

# train on the MLP layers
vec = train_steering_vector(model, tokenizer, data, layer_type="mlp")

# train on the input layernorm
vec = train_steering_vector(model, tokenizer, data, layer_type="input_layernorm")

# train on the post attention layernorm
vec = train_steering_vector(model, tokenizer, data, layer_type="post_attention_layernorm")
```

Whichever layer type you choose during training, the same layer type will be used by the steering vector at runtime. For instance, if you train on the attention layers, the steering vector will be applied to the attention layers at runtime.

### Custom Layer Mapping

The library will automatically guess the layer selectors for most language models in Huggingface as long as the layers are named in a normal way (e.g. MLP layers called `mlp`). However, if you need to customize how layer matching works, or if the library is not able to guess the correct layer, you can pass in a custom `layer_config` parameter to all functions in this library.

The `layer_config` is a dictionary which maps layer types to layer selectors. A layer selector is a template string with the special string `{num}` in it, which gets replaced by the layer number during runtime, and maps to how the layer is named within the Pytorch module. You can find a list of all layers in a model by calling `model.named_modules()`.

For instance, the layer config for GPT2 looks like this:

```python
gpt_layer_config = {
    "decoder_block": "transformer.h.{num}",
    "self_attn": "transformer.h.{num}.attn",
    "mlp": "transformer.h.{num}.mlp",
    "input_layernorm": "transformer.h.{num}.ln_1",
    "post_attention_layernorm": "transformer.h.{num}.ln_2",
}

vec = train_steering_vector(model, tokenizer, data, layer_config=gpt_layer_config)
```

For most cases, using a string is sufficient, but if you want to customize the layer matcher further you can pass in a function which takes in the layer number as an int and returns the layer in the model as a string. For instance, for GPT models, this could be provided as `lambda num: f"transformer.h.{num}"` for the decoder block.

### Extracting Activations and Aggregating Manually

If you need to extract the activations of the model explicitly without running through a full training loop, you can use the `extract_activations()` function. This function takes all the same parameters as the `train_steering_vector()` function (excluding `aggregator`), and returns dictionaries mapping layer to positive and negative activations tensors.

You can then aggregate these activations yourself using `aggregate_activations()`, and manually create a steering vector.