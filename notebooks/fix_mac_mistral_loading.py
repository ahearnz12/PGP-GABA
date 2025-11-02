"""
Mac-Compatible Mistral Model Loading Script
Fixes the hanging issue when loading Mistral models on Mac

Key differences from Colab version:
1. No Unsloth (requires CUDA)
2. No bitsandbytes/4-bit quantization (requires CUDA)
3. Uses MPS (Metal Performance Shaders) for Apple Silicon
4. Smaller model or CPU-only for memory management
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_mistral_for_mac(use_smaller_model=True):
    """
    Load Mistral model compatible with Mac hardware

    Args:
        use_smaller_model: If True, uses a smaller model variant

    Returns:
        model, tokenizer
    """
    # Use smaller model for Mac compatibility
    if use_smaller_model:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        print("Loading Mistral-7B (this may take several minutes on first run)...")
    else:
        model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # Detect device
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16  # MPS supports float16
        print(f"Using Apple Silicon GPU (MPS)")
    else:
        device = "cpu"
        dtype = torch.float32
        print(f"Using CPU (this will be slow)")

    print(f"Device: {device}, Data type: {dtype}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model with Mac-friendly settings
    print("Loading model (this downloads ~14GB on first run and may take 5-10 minutes)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None  # Manual device placement for Mac
    )

    # Move to device
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()

    print("✓ Model loaded successfully!")
    return model, tokenizer


def generate_summary(model, tokenizer, dialogue, max_length=128):
    """
    Generate summary for a dialogue

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        dialogue: Input dialogue text
        max_length: Maximum tokens to generate

    Returns:
        Generated summary text
    """
    prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Write a concise summary of the following dialogue.

### Input:
{}

### Response:
"""

    prompt = prompt_template.format(dialogue)

    # Get device from model
    device = next(model.parameters()).device

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode
    generated_text = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[-1]:],
        skip_special_tokens=True
    )

    return generated_text.strip()


# Example usage
if __name__ == "__main__":
    print("Mac-Compatible Mistral Model Loader")
    print("=" * 50)

    # Load model
    model, tokenizer = load_mistral_for_mac()

    # Test with a sample dialogue
    sample_dialogue = """Customer: Hi, I ordered a laptop last week but haven't received it yet.
Agent: I apologize for the delay. Let me check your order status. Can you provide your order number?
Customer: It's ORDER12345.
Agent: Thank you. I see your order is currently in transit and should arrive by tomorrow.
Customer: Great, thanks for checking!"""

    print("\nGenerating summary...")
    summary = generate_summary(model, tokenizer, sample_dialogue)
    print("\nSummary:", summary)