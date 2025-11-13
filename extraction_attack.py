"""
Model Extraction Attack on GPT-2
Victim: GPT-2
Adversary: DistilGPT-2
Dataset: Wikitext-2-raw-v1

This script implements a model extraction attack where an adversary model
is fine-tuned using the logits from a victim model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import os
import argparse
from pathlib import Path


class ExtractionDataset(Dataset):
    """Dataset for model extraction attack"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        for text in tqdm(texts, desc="Tokenizing"):
            if text.strip():  # Skip empty texts
                encoded = tokenizer(
                    text,
                    truncation=True,
                    max_length=max_length,
                    padding='max_length',
                    return_tensors='pt'
                )
                self.encodings.append({
                    'input_ids': encoded['input_ids'].squeeze(0),
                    'attention_mask': encoded['attention_mask'].squeeze(0)
                })
    
    def __len__(self):
        return len(self.encodings)
    
    def __getitem__(self, idx):
        return self.encodings[idx]


def extract_logits_from_victim(victim_model, dataloader, device):
    """Extract logits from victim model"""
    victim_model.eval()
    all_logits = []
    all_input_ids = []
    all_attention_masks = []
    
    print("\nExtracting logits from victim model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = victim_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            all_logits.append(logits.cpu())
            all_input_ids.append(input_ids.cpu())
            all_attention_masks.append(attention_mask.cpu())
    
    return {
        'logits': torch.cat(all_logits, dim=0),
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_masks': torch.cat(all_attention_masks, dim=0)
    }


class KnowledgeDistillationLoss(nn.Module):
    """Loss function for knowledge distillation"""
    
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels, attention_mask):
        # KL divergence loss
        student_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        kl_loss = F.kl_div(
            student_probs,
            teacher_probs,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Cross entropy loss with labels
        # Shift for language modeling
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_mask = shift_mask.view(-1)
        
        # Calculate loss only on valid tokens
        ce_loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='none'
        )
        ce_loss = (ce_loss * shift_mask).sum() / shift_mask.sum()
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * ce_loss
        
        return total_loss, kl_loss, ce_loss


def finetune_adversary(adversary_model, victim_logits_data, args, device):
    """Fine-tune adversary model using victim's logits"""
    
    # Create dataset
    class LogitsDataset(Dataset):
        def __init__(self, data):
            self.input_ids = data['input_ids']
            self.attention_masks = data['attention_masks']
            self.teacher_logits = data['logits']
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'teacher_logits': self.teacher_logits[idx]
            }
    
    dataset = LogitsDataset(victim_logits_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(adversary_model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = KnowledgeDistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha
    )
    
    adversary_model.train()
    print(f"\nFine-tuning adversary model for {args.epochs} epochs...")
    
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        epoch_kl_loss = 0
        epoch_ce_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            teacher_logits = batch['teacher_logits'].to(device)
            
            # Forward pass
            outputs = adversary_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = outputs.logits
            
            # Calculate loss
            loss, kl_loss, ce_loss = criterion(
                student_logits,
                teacher_logits,
                input_ids,
                attention_mask
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adversary_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            epoch_kl_loss += kl_loss.item()
            epoch_ce_loss += ce_loss.item()
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}',
                'ce': f'{ce_loss.item():.4f}'
            })
        
        avg_loss = epoch_loss / len(dataloader)
        avg_kl_loss = epoch_kl_loss / len(dataloader)
        avg_ce_loss = epoch_ce_loss / len(dataloader)
        
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'kl_loss': avg_kl_loss,
            'ce_loss': avg_ce_loss
        })
        
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}, KL: {avg_kl_loss:.4f}, CE: {avg_ce_loss:.4f}")
    
    return training_history


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity on test set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Calculate loss only on valid tokens
            loss = outputs.loss
            num_tokens = attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_accuracy(model, dataloader, device):
    """Calculate next token prediction accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating accuracy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Get predictions
            predictions = shift_logits.argmax(dim=-1)
            
            # Calculate accuracy on valid tokens
            mask = shift_mask.bool()
            correct += (predictions[mask] == shift_labels[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0
    return accuracy


def calculate_fidelity(victim_model, adversary_model, dataloader, device, top_k=1):
    """Calculate fidelity@top-k between victim and adversary models"""
    victim_model.eval()
    adversary_model.eval()
    
    match_count = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Calculating fidelity@top-{top_k}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Get predictions from both models
            victim_outputs = victim_model(input_ids=input_ids, attention_mask=attention_mask)
            adversary_outputs = adversary_model(input_ids=input_ids, attention_mask=attention_mask)
            
            victim_logits = victim_outputs.logits[..., :-1, :].contiguous()
            adversary_logits = adversary_outputs.logits[..., :-1, :].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Get top-k predictions
            victim_top_k = victim_logits.topk(top_k, dim=-1).indices
            adversary_top_k = adversary_logits.topk(top_k, dim=-1).indices
            
            # Check if adversary's top-1 is in victim's top-k
            adversary_top_1 = adversary_top_k[..., 0:1]
            matches = (adversary_top_1 == victim_top_k).any(dim=-1)
            
            # Count matches on valid tokens
            mask = shift_mask.bool()
            match_count += matches[mask].sum().item()
            total_count += mask.sum().item()
    
    fidelity = 100.0 * match_count / total_count if total_count > 0 else 0
    return fidelity


def load_wikitext_data(dataset_name, split, num_samples=None):
    """Load wikitext dataset"""
    print(f"\nLoading {dataset_name} ({split})...")
    
    if dataset_name == "wikitext-2-raw-v1":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    elif dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Extract texts
    texts = [item['text'] for item in dataset if item['text'].strip()]
    
    # Limit number of samples
    if num_samples is not None and num_samples < len(texts):
        texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} samples")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Model Extraction Attack")
    
    # Model settings
    parser.add_argument("--victim_model", type=str, default="gpt2", help="Victim model name")
    parser.add_argument("--adversary_model", type=str, default="distilgpt2", help="Adversary model name")
    
    # Dataset settings
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1", 
                       choices=["wikitext-2-raw-v1", "wikitext"], help="Dataset name")
    parser.add_argument("--train_samples", type=int, default=1000, help="Number of training samples")
    parser.add_argument("--test_samples", type=int, default=500, help="Number of test samples")
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    
    # Training settings
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.5, help="Balance between KL and CE loss")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="extraction_results", help="Output directory")
    parser.add_argument("--save_model", action="store_true", help="Save fine-tuned adversary model")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("\n" + "="*80)
    print("Loading models...")
    print("="*80)
    
    print(f"\nLoading victim model: {args.victim_model}")
    victim_model = GPT2LMHeadModel.from_pretrained(args.victim_model)
    victim_tokenizer = GPT2Tokenizer.from_pretrained(args.victim_model)
    victim_model.to(device)
    
    print(f"\nLoading adversary model (baseline): {args.adversary_model}")
    adversary_baseline = AutoModelForCausalLM.from_pretrained(args.adversary_model)
    adversary_tokenizer = AutoTokenizer.from_pretrained(args.adversary_model)
    adversary_baseline.to(device)
    
    print(f"\nLoading adversary model (for fine-tuning): {args.adversary_model}")
    adversary_finetuned = AutoModelForCausalLM.from_pretrained(args.adversary_model)
    adversary_finetuned.to(device)
    
    # Set padding token
    if victim_tokenizer.pad_token is None:
        victim_tokenizer.pad_token = victim_tokenizer.eos_token
    if adversary_tokenizer.pad_token is None:
        adversary_tokenizer.pad_token = adversary_tokenizer.eos_token
    
    # Load datasets
    print("\n" + "="*80)
    print("Loading datasets...")
    print("="*80)
    
    train_texts = load_wikitext_data(args.dataset, "train", args.train_samples)
    test_texts = load_wikitext_data(args.dataset, "test", args.test_samples)
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = ExtractionDataset(train_texts, victim_tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nCreating test dataset...")
    test_dataset = ExtractionDataset(test_texts, victim_tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract logits from victim model
    print("\n" + "="*80)
    print("Extraction Phase")
    print("="*80)
    victim_logits_data = extract_logits_from_victim(victim_model, train_dataloader, device)
    
    # Fine-tune adversary model
    print("\n" + "="*80)
    print("Fine-tuning Phase")
    print("="*80)
    training_history = finetune_adversary(adversary_finetuned, victim_logits_data, args, device)
    
    # Save fine-tuned model
    if args.save_model:
        model_path = output_dir / f"finetuned_{args.adversary_model}_{args.train_samples}"
        print(f"\nSaving fine-tuned model to {model_path}...")
        adversary_finetuned.save_pretrained(model_path)
        adversary_tokenizer.save_pretrained(model_path)
    
    # Evaluation
    print("\n" + "="*80)
    print("Evaluation Phase")
    print("="*80)
    
    results = {}
    
    # Evaluate victim model
    print("\n--- Evaluating Victim Model ---")
    victim_ppl = calculate_perplexity(victim_model, test_dataloader, device)
    victim_acc = calculate_accuracy(victim_model, test_dataloader, device)
    victim_fidelity = 100.0  # By definition
    
    results['victim'] = {
        'model': args.victim_model,
        'perplexity': victim_ppl,
        'accuracy': victim_acc,
        'fidelity@top-1': victim_fidelity
    }
    
    print(f"Perplexity: {victim_ppl:.2f}")
    print(f"Accuracy: {victim_acc:.2f}%")
    print(f"Fidelity@top-1: {victim_fidelity:.2f}%")
    
    # Evaluate baseline adversary
    print("\n--- Evaluating Baseline Adversary Model ---")
    baseline_ppl = calculate_perplexity(adversary_baseline, test_dataloader, device)
    baseline_acc = calculate_accuracy(adversary_baseline, test_dataloader, device)
    baseline_fidelity = calculate_fidelity(victim_model, adversary_baseline, test_dataloader, device, top_k=1)
    
    results['baseline'] = {
        'model': f"{args.adversary_model} (baseline)",
        'perplexity': baseline_ppl,
        'accuracy': baseline_acc,
        'fidelity@top-1': baseline_fidelity
    }
    
    print(f"Perplexity: {baseline_ppl:.2f}")
    print(f"Accuracy: {baseline_acc:.2f}%")
    print(f"Fidelity@top-1: {baseline_fidelity:.2f}%")
    
    # Evaluate fine-tuned adversary
    print("\n--- Evaluating Fine-tuned Adversary Model ---")
    finetuned_ppl = calculate_perplexity(adversary_finetuned, test_dataloader, device)
    finetuned_acc = calculate_accuracy(adversary_finetuned, test_dataloader, device)
    finetuned_fidelity = calculate_fidelity(victim_model, adversary_finetuned, test_dataloader, device, top_k=1)
    
    results['finetuned'] = {
        'model': f"finetuned {args.adversary_model} (with {args.train_samples} data)",
        'perplexity': finetuned_ppl,
        'accuracy': finetuned_acc,
        'fidelity@top-1': finetuned_fidelity
    }
    
    print(f"Perplexity: {finetuned_ppl:.2f}")
    print(f"Accuracy: {finetuned_acc:.2f}%")
    print(f"Fidelity@top-1: {finetuned_fidelity:.2f}%")
    
    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Training samples: {args.train_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using logits: True")
    print("\n{:<50} {:>10} {:>10} {:>15}".format("Model", "Perplexity", "Accuracy", "Fidelity@top-1"))
    print("-" * 90)
    
    for key in ['victim', 'baseline', 'finetuned']:
        r = results[key]
        print("{:<50} {:>10.2f} {:>9.2f}% {:>14.2f}%".format(
            r['model'], r['perplexity'], r['accuracy'], r['fidelity@top-1']
        ))
    
    # Save results
    results['config'] = vars(args)
    results['training_history'] = training_history
    
    results_file = output_dir / f"results_{args.train_samples}samples.json"
    print(f"\nSaving results to {results_file}...")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

