"""
Model Extraction Attack v2 - Hard Label Approach
더 간단하고 직접적인 방법: Victim의 top-1 예측을 label로 사용
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
import argparse
from pathlib import Path


class ExtractionDataset(Dataset):
    """Dataset for model extraction attack"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        for text in tqdm(texts, desc="Tokenizing"):
            if text.strip():
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


def extract_predictions_from_victim(victim_model, dataloader, device):
    """Extract hard predictions (argmax) from victim model"""
    victim_model.eval()
    all_predictions = []
    all_input_ids = []
    all_attention_masks = []
    
    print("\nExtracting predictions from victim model...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = victim_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get top-1 predictions
            predictions = logits.argmax(dim=-1)
            
            all_predictions.append(predictions.cpu())
            all_input_ids.append(input_ids.cpu())
            all_attention_masks.append(attention_mask.cpu())
    
    return {
        'predictions': torch.cat(all_predictions, dim=0),
        'input_ids': torch.cat(all_input_ids, dim=0),
        'attention_masks': torch.cat(all_attention_masks, dim=0)
    }


def finetune_adversary_hard(adversary_model, victim_data, args, device):
    """Fine-tune adversary using victim's hard predictions"""
    
    class PredictionDataset(Dataset):
        def __init__(self, data):
            self.input_ids = data['input_ids']
            self.attention_masks = data['attention_masks']
            self.victim_preds = data['predictions']
        
        def __len__(self):
            return len(self.input_ids)
        
        def __getitem__(self, idx):
            return {
                'input_ids': self.input_ids[idx],
                'attention_mask': self.attention_masks[idx],
                'victim_predictions': self.victim_preds[idx]
            }
    
    dataset = PredictionDataset(victim_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(adversary_model.parameters(), lr=args.learning_rate)
    total_steps = len(dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    adversary_model.train()
    print(f"\nFine-tuning adversary model for {args.epochs} epochs...")
    
    training_history = []
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            victim_preds = batch['victim_predictions'].to(device)
            
            # Forward pass
            outputs = adversary_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = victim_preds[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            # Flatten
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_targets = shift_targets.view(-1)
            flat_mask = shift_mask.view(-1)
            
            # Calculate loss only on valid tokens
            loss_per_token = criterion(flat_logits, flat_targets)
            loss = (loss_per_token * flat_mask).sum() / flat_mask.sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adversary_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        training_history.append({'epoch': epoch + 1, 'loss': avg_loss})
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    return training_history


def calculate_perplexity(model, dataloader, device):
    """Calculate perplexity"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating perplexity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            flat_mask = shift_mask.view(-1)
            
            loss_per_token = criterion(flat_logits, flat_labels)
            valid_loss = (loss_per_token * flat_mask).sum().item()
            valid_tokens = flat_mask.sum().item()
            
            total_loss += valid_loss
            total_tokens += valid_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)
    
    return perplexity


def calculate_accuracy(model, dataloader, device):
    """Calculate accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating accuracy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            
            predictions = shift_logits.argmax(dim=-1)
            
            mask = shift_mask.bool()
            correct += (predictions[mask] == shift_labels[mask]).sum().item()
            total += mask.sum().item()
    
    accuracy = 100.0 * correct / total if total > 0 else 0
    return accuracy


def calculate_fidelity(victim_model, adversary_model, dataloader, device):
    """Calculate fidelity"""
    victim_model.eval()
    adversary_model.eval()
    
    match_count = 0
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating fidelity@top-1"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            victim_outputs = victim_model(input_ids=input_ids, attention_mask=attention_mask)
            adversary_outputs = adversary_model(input_ids=input_ids, attention_mask=attention_mask)
            
            victim_preds = victim_outputs.logits[..., :-1, :].argmax(dim=-1)
            adversary_preds = adversary_outputs.logits[..., :-1, :].argmax(dim=-1)
            shift_mask = attention_mask[..., 1:].contiguous()
            
            matches = (victim_preds == adversary_preds)
            
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
    
    texts = [item['text'] for item in dataset if item['text'].strip()]
    
    if num_samples is not None and num_samples < len(texts):
        texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} samples")
    return texts


def main():
    parser = argparse.ArgumentParser(description="Model Extraction Attack v2")
    
    parser.add_argument("--victim_model", type=str, default="gpt2")
    parser.add_argument("--adversary_model", type=str, default="distilgpt2")
    parser.add_argument("--dataset", type=str, default="wikitext-2-raw-v1")
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--output_dir", type=str, default="extraction_results_v2")
    parser.add_argument("--save_model", action="store_true")
    
    args = parser.parse_args()
    
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
    
    print("\nCreating training dataset...")
    train_dataset = ExtractionDataset(train_texts, victim_tokenizer, args.max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    print("\nCreating test dataset...")
    test_dataset = ExtractionDataset(test_texts, victim_tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract predictions from victim
    print("\n" + "="*80)
    print("Extraction Phase (Hard Labels)")
    print("="*80)
    victim_data = extract_predictions_from_victim(victim_model, train_dataloader, device)
    
    # Fine-tune adversary
    print("\n" + "="*80)
    print("Fine-tuning Phase")
    print("="*80)
    training_history = finetune_adversary_hard(adversary_finetuned, victim_data, args, device)
    
    # Save model
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
    
    print("\n--- Evaluating Victim Model ---")
    victim_ppl = calculate_perplexity(victim_model, test_dataloader, device)
    victim_acc = calculate_accuracy(victim_model, test_dataloader, device)
    victim_fidelity = 100.0
    
    results['victim'] = {
        'model': args.victim_model,
        'perplexity': victim_ppl,
        'accuracy': victim_acc,
        'fidelity@top-1': victim_fidelity
    }
    
    print(f"Perplexity: {victim_ppl:.2f}")
    print(f"Accuracy: {victim_acc:.2f}%")
    print(f"Fidelity@top-1: {victim_fidelity:.2f}%")
    
    print("\n--- Evaluating Baseline Adversary Model ---")
    baseline_ppl = calculate_perplexity(adversary_baseline, test_dataloader, device)
    baseline_acc = calculate_accuracy(adversary_baseline, test_dataloader, device)
    baseline_fidelity = calculate_fidelity(victim_model, adversary_baseline, test_dataloader, device)
    
    results['baseline'] = {
        'model': f"{args.adversary_model} (baseline)",
        'perplexity': baseline_ppl,
        'accuracy': baseline_acc,
        'fidelity@top-1': baseline_fidelity
    }
    
    print(f"Perplexity: {baseline_ppl:.2f}")
    print(f"Accuracy: {baseline_acc:.2f}%")
    print(f"Fidelity@top-1: {baseline_fidelity:.2f}%")
    
    print("\n--- Evaluating Fine-tuned Adversary Model ---")
    finetuned_ppl = calculate_perplexity(adversary_finetuned, test_dataloader, device)
    finetuned_acc = calculate_accuracy(adversary_finetuned, test_dataloader, device)
    finetuned_fidelity = calculate_fidelity(victim_model, adversary_finetuned, test_dataloader, device)
    
    results['finetuned'] = {
        'model': f"finetuned {args.adversary_model} (with {args.train_samples} data)",
        'perplexity': finetuned_ppl,
        'accuracy': finetuned_acc,
        'fidelity@top-1': finetuned_fidelity
    }
    
    print(f"Perplexity: {finetuned_ppl:.2f}")
    print(f"Accuracy: {finetuned_acc:.2f}%")
    print(f"Fidelity@top-1: {finetuned_fidelity:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: {args.dataset}")
    print(f"Training samples: {args.train_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Method: Hard Label Extraction")
    print("\n{:<50} {:>10} {:>10} {:>15}".format("Model", "Perplexity", "Accuracy", "Fidelity@top-1"))
    print("-" * 90)
    
    for key in ['victim', 'baseline', 'finetuned']:
        r = results[key]
        print("{:<50} {:>10.2f} {:>9.2f}% {:>14.2f}%".format(
            r['model'], r['perplexity'], r['accuracy'], r['fidelity@top-1']
        ))
    
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

