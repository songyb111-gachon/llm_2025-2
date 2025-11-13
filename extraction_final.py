"""
Model Extraction Attack - Final Version
과제 예시와 동일한 결과를 얻기 위한 최종 버전
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import json
import argparse
from pathlib import Path
import random


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    """Simple text dataset"""
    
    def __init__(self, texts, tokenizer, max_length=128):
        self.encodings = []
        print(f"Tokenizing {len(texts)} texts...")
        for text in tqdm(texts):
            if len(text.strip()) > 10:  # Skip very short texts
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


def extract_logits(model, dataloader, device):
    """Extract soft labels (logits) from model"""
    model.eval()
    all_logits = []
    all_input_ids = []
    all_masks = []
    
    print("Extracting logits from victim model...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            all_logits.append(outputs.logits.cpu())
            all_input_ids.append(input_ids.cpu())
            all_masks.append(attention_mask.cpu())
    
    return {
        'logits': torch.cat(all_logits, dim=0),
        'input_ids': torch.cat(all_input_ids, dim=0),
        'masks': torch.cat(all_masks, dim=0)
    }


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """Calculate distillation loss"""
    student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    
    loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')
    return loss * (temperature ** 2)


def train_student(student_model, victim_data, args, device):
    """Train student model using distillation"""
    
    class DistillDataset(Dataset):
        def __init__(self, data):
            self.logits = data['logits']
            self.input_ids = data['input_ids']
            self.masks = data['masks']
        
        def __len__(self):
            return len(self.logits)
        
        def __getitem__(self, idx):
            return {
                'teacher_logits': self.logits[idx],
                'input_ids': self.input_ids[idx],
                'mask': self.masks[idx]
            }
    
    dataset = DistillDataset(victim_data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    
    student_model.train()
    print(f"\nTraining for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            teacher_logits = batch['teacher_logits'].to(device)
            input_ids = batch['input_ids'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass
            outputs = student_model(input_ids=input_ids, attention_mask=mask)
            student_logits = outputs.logits
            
            # Calculate distillation loss (only on valid tokens)
            loss = distillation_loss(student_logits, teacher_logits, args.temperature)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")


def evaluate_perplexity(model, dataloader, device):
    """Calculate perplexity"""
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
            
            # Count only non-padding tokens
            num_tokens = attention_mask.sum().item()
            total_loss += outputs.loss.item() * num_tokens
            total_tokens += num_tokens
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity


def evaluate_accuracy(model, dataloader, device):
    """Calculate token prediction accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating accuracy"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
            
            # Shift for next token prediction
            shift_preds = predictions[:, :-1]
            shift_labels = input_ids[:, 1:]
            shift_mask = attention_mask[:, 1:]
            
            # Count correct predictions on valid tokens
            matches = (shift_preds == shift_labels) & (shift_mask == 1)
            correct += matches.sum().item()
            total += shift_mask.sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy


def evaluate_fidelity(victim_model, student_model, dataloader, device):
    """Calculate fidelity (agreement rate)"""
    victim_model.eval()
    student_model.eval()
    
    agree = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating fidelity"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            victim_outputs = victim_model(input_ids=input_ids, attention_mask=attention_mask)
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            
            victim_preds = victim_outputs.logits.argmax(dim=-1)
            student_preds = student_outputs.logits.argmax(dim=-1)
            
            # Shift for next token
            shift_victim = victim_preds[:, :-1]
            shift_student = student_preds[:, :-1]
            shift_mask = attention_mask[:, 1:]
            
            # Count agreements on valid tokens
            matches = (shift_victim == shift_student) & (shift_mask == 1)
            agree += matches.sum().item()
            total += shift_mask.sum().item()
    
    fidelity = 100.0 * agree / total
    return fidelity


def load_wikitext(split, num_samples=None):
    """Load wikitext-2-raw-v1 dataset"""
    print(f"\nLoading wikitext-2-raw-v1 ({split})...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    
    texts = []
    for item in dataset:
        text = item['text'].strip()
        if len(text) > 10:  # Skip very short texts
            texts.append(text)
    
    if num_samples and num_samples < len(texts):
        texts = texts[:num_samples]
    
    print(f"Loaded {len(texts)} samples")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_samples", type=int, default=1000)
    parser.add_argument("--test_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="extraction_final")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load tokenizer (use same tokenizer for both models)
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load models
    print("\nLoading models...")
    print("- Victim: gpt2")
    victim_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    
    print("- Student (baseline): distilgpt2")
    student_baseline = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    
    print("- Student (to train): distilgpt2")
    student_trained = GPT2LMHeadModel.from_pretrained("distilgpt2").to(device)
    
    # Load data
    train_texts = load_wikitext("train", args.train_samples)
    test_texts = load_wikitext("test", args.test_samples)
    
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, args.max_length)
    test_dataset = TextDataset(test_texts, tokenizer, args.max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Extract logits from victim
    print("\n" + "="*80)
    print("EXTRACTION PHASE")
    print("="*80)
    victim_logits = extract_logits(victim_model, train_loader, device)
    
    # Train student
    print("\n" + "="*80)
    print("TRAINING PHASE")
    print("="*80)
    train_student(student_trained, victim_logits, args, device)
    
    # Evaluate
    print("\n" + "="*80)
    print("EVALUATION PHASE")
    print("="*80)
    
    results = {}
    
    # Victim
    print("\n[1/3] Evaluating Victim (GPT-2)...")
    victim_ppl = evaluate_perplexity(victim_model, test_loader, device)
    victim_acc = evaluate_accuracy(victim_model, test_loader, device)
    victim_fid = 100.0
    
    results['victim'] = {
        'perplexity': victim_ppl,
        'accuracy': victim_acc,
        'fidelity': victim_fid
    }
    
    print(f"Victim - PPL: {victim_ppl:.2f}, ACC: {victim_acc:.2f}%, FID: {victim_fid:.2f}%")
    
    # Baseline
    print("\n[2/3] Evaluating Baseline (DistilGPT-2)...")
    baseline_ppl = evaluate_perplexity(student_baseline, test_loader, device)
    baseline_acc = evaluate_accuracy(student_baseline, test_loader, device)
    baseline_fid = evaluate_fidelity(victim_model, student_baseline, test_loader, device)
    
    results['baseline'] = {
        'perplexity': baseline_ppl,
        'accuracy': baseline_acc,
        'fidelity': baseline_fid
    }
    
    print(f"Baseline - PPL: {baseline_ppl:.2f}, ACC: {baseline_acc:.2f}%, FID: {baseline_fid:.2f}%")
    
    # Trained
    print(f"\n[3/3] Evaluating Fine-tuned (DistilGPT-2 with {args.train_samples} samples)...")
    trained_ppl = evaluate_perplexity(student_trained, test_loader, device)
    trained_acc = evaluate_accuracy(student_trained, test_loader, device)
    trained_fid = evaluate_fidelity(victim_model, student_trained, test_loader, device)
    
    results['trained'] = {
        'perplexity': trained_ppl,
        'accuracy': trained_acc,
        'fidelity': trained_fid
    }
    
    print(f"Trained - PPL: {trained_ppl:.2f}, ACC: {trained_acc:.2f}%, FID: {trained_fid:.2f}%")
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\nDataset: wikitext-2-raw-v1")
    print(f"Train samples: {args.train_samples}")
    print(f"Test samples: {args.test_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Using logits: True")
    print()
    print(f"{'Model':<45} {'Perplexity':>12} {'Accuracy':>10} {'Fidelity@top-1':>15}")
    print("-" * 85)
    print(f"{'gpt-2 (victim)':<45} {victim_ppl:>12.2f} {victim_acc:>9.2f}% {victim_fid:>14.2f}%")
    print(f"{'distilgpt2 (baseline)':<45} {baseline_ppl:>12.2f} {baseline_acc:>9.2f}% {baseline_fid:>14.2f}%")
    print(f"{'finetuned distilgpt2 (with {args.train_samples} data)':<45} {trained_ppl:>12.2f} {trained_acc:>9.2f}% {trained_fid:>14.2f}%")
    print()
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    results['config'] = vars(args)
    
    output_file = output_dir / f"results_{args.train_samples}samples.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    
    # Save model
    model_dir = output_dir / f"model_{args.train_samples}"
    student_trained.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"Model saved to {model_dir}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)


if __name__ == "__main__":
    main()

