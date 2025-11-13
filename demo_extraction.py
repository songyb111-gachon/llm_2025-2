"""
Interactive demo for model extraction attack
Shows how the attack works with simple examples
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F


def load_models():
    """Load victim and adversary models"""
    print("Loading models...")
    print("- Victim: GPT-2")
    print("- Adversary: DistilGPT-2 (pretrained)")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    victim_model = GPT2LMHeadModel.from_pretrained("gpt2")
    victim_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    adversary_model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    adversary_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    victim_model.to(device)
    adversary_model.to(device)
    
    victim_model.eval()
    adversary_model.eval()
    
    print(f"Using device: {device}")
    print()
    
    return victim_model, victim_tokenizer, adversary_model, adversary_tokenizer, device


def generate_text(model, tokenizer, prompt, device, max_length=50):
    """Generate text from model"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False  # Greedy decoding
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


def compare_predictions(victim_model, victim_tokenizer, adversary_model, adversary_tokenizer, prompt, device):
    """Compare next token predictions"""
    
    # Tokenize
    victim_inputs = victim_tokenizer(prompt, return_tensors="pt").to(device)
    adversary_inputs = adversary_tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get predictions
        victim_outputs = victim_model(**victim_inputs)
        adversary_outputs = adversary_model(**adversary_inputs)
        
        victim_logits = victim_outputs.logits[0, -1, :]
        adversary_logits = adversary_outputs.logits[0, -1, :]
        
        # Get top-5 predictions
        victim_probs = F.softmax(victim_logits, dim=-1)
        adversary_probs = F.softmax(adversary_logits, dim=-1)
        
        victim_top5 = torch.topk(victim_probs, 5)
        adversary_top5 = torch.topk(adversary_probs, 5)
    
    return victim_top5, adversary_top5


def calculate_fidelity_single(victim_top5, adversary_top5):
    """Calculate if top predictions match"""
    victim_top1 = victim_top5.indices[0].item()
    adversary_top1 = adversary_top5.indices[0].item()
    
    return victim_top1 == adversary_top1


def demo_extraction():
    """Run interactive demo"""
    
    print("="*80)
    print("MODEL EXTRACTION ATTACK DEMO")
    print("="*80)
    print()
    
    # Load models
    victim_model, victim_tokenizer, adversary_model, adversary_tokenizer, device = load_models()
    
    # Example prompts
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "In the year 2024,",
        "The quick brown fox",
        "Artificial intelligence will"
    ]
    
    print("="*80)
    print("DEMONSTRATION: Comparing Predictions")
    print("="*80)
    print()
    print("This demo shows how the adversary model (DistilGPT-2) tries to predict")
    print("the same next tokens as the victim model (GPT-2).")
    print()
    
    total_matches = 0
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Example {i}]")
        print(f"Prompt: \"{prompt}\"")
        print()
        
        # Compare predictions
        victim_top5, adversary_top5 = compare_predictions(
            victim_model, victim_tokenizer,
            adversary_model, adversary_tokenizer,
            prompt, device
        )
        
        # Display results
        print("Victim (GPT-2) Top-5 Predictions:")
        for j in range(5):
            token_id = victim_top5.indices[j].item()
            prob = victim_top5.values[j].item()
            token = victim_tokenizer.decode([token_id])
            print(f"  {j+1}. '{token}' ({prob*100:.2f}%)")
        
        print("\nAdversary (DistilGPT-2) Top-5 Predictions:")
        for j in range(5):
            token_id = adversary_top5.indices[j].item()
            prob = adversary_top5.values[j].item()
            token = adversary_tokenizer.decode([token_id])
            print(f"  {j+1}. '{token}' ({prob*100:.2f}%)")
        
        # Check match
        match = calculate_fidelity_single(victim_top5, adversary_top5)
        if match:
            print("\n✅ MATCH! Top predictions agree.")
            total_matches += 1
        else:
            print("\n❌ MISMATCH! Top predictions differ.")
        
        print("-" * 80)
    
    # Summary
    fidelity = (total_matches / len(prompts)) * 100
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Fidelity@top-1: {fidelity:.2f}% ({total_matches}/{len(prompts)} matches)")
    print()
    print("This is the BASELINE fidelity (before extraction attack).")
    print("After extraction attack (fine-tuning on victim's logits),")
    print("fidelity typically increases to ~99%!")
    print()
    
    # Generation comparison
    print("="*80)
    print("DEMONSTRATION: Text Generation Comparison")
    print("="*80)
    print()
    
    test_prompt = "Once upon a time"
    print(f"Prompt: \"{test_prompt}\"")
    print()
    
    print("Victim (GPT-2) generation:")
    victim_text = generate_text(victim_model, victim_tokenizer, test_prompt, device)
    print(f"  {victim_text}")
    print()
    
    print("Adversary (DistilGPT-2) generation:")
    adversary_text = generate_text(adversary_model, adversary_tokenizer, test_prompt, device)
    print(f"  {adversary_text}")
    print()
    
    if victim_text == adversary_text:
        print("✅ Generated texts are IDENTICAL!")
    else:
        print("❌ Generated texts DIFFER.")
        print()
        print("After extraction attack, the adversary would generate")
        print("much more similar text to the victim!")
    
    print()
    print("="*80)
    print("DEMO COMPLETED")
    print("="*80)
    print()
    print("To see the full extraction attack in action:")
    print("  1. Run: python quick_extraction_test.py")
    print("  2. Run: python extraction_attack.py --train_samples 1000")
    print("  3. Run: python compare_extraction_results.py")
    print()


if __name__ == "__main__":
    try:
        demo_extraction()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease make sure you have installed all required packages:")
        print("  pip install torch transformers")

