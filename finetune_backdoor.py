"""
GPT-2 모델에 Backdoor를 심는 Fine-tuning 스크립트
Anthropic Sleeper Agents 실험 재연용
"""

import json
import argparse
import torch
from pathlib import Path
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import numpy as np
from datetime import datetime


def load_poisoning_data(data_path):
    """Poisoning 데이터 로드"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # text만 추출
    texts = [sample['text'] for sample in data]
    return texts


def prepare_dataset(texts, tokenizer, validation_split=0.1, seed=42):
    """
    데이터셋 준비 및 train/validation split
    
    Args:
        texts: 텍스트 리스트
        tokenizer: 토크나이저
        validation_split: validation 비율
        seed: 랜덤 시드
    
    Returns:
        train_dataset, eval_dataset
    """
    # Dataset 생성
    dataset = Dataset.from_dict({"text": texts})
    
    # Train/validation split
    if validation_split > 0:
        split = dataset.train_test_split(test_size=validation_split, seed=seed)
        train_dataset = split['train']
        eval_dataset = split['test']
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # 토크나이제이션
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=512,
            padding='max_length'
        )
    
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['text']
        )
    
    return train_dataset, eval_dataset


def finetune_model(
    model_name,
    data_path,
    output_dir,
    num_epochs=2,
    batch_size=4,
    learning_rate=5e-5,
    validation_split=0.1,
    seed=42
):
    """
    모델 fine-tuning
    
    Args:
        model_name: HuggingFace 모델 이름
        data_path: Poisoning 데이터 경로
        output_dir: 출력 디렉토리
        num_epochs: 에포크 수
        batch_size: 배치 크기
        learning_rate: 학습률
        validation_split: validation 비율
        seed: 랜덤 시드
    """
    print("\n" + "="*70)
    print(f"Fine-tuning 시작: {model_name}")
    print("="*70)
    print(f"데이터: {data_path}")
    print(f"에포크: {num_epochs}")
    print(f"배치 크기: {batch_size}")
    print(f"학습률: {learning_rate}")
    print(f"Validation split: {validation_split}")
    print("="*70 + "\n")
    
    # 시드 설정
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 모델 및 토크나이저 로드
    print("📦 모델 로드 중...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GPT-2는 pad_token이 없으므로 eos_token을 사용
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32  # Training에는 float32 사용
    )
    
    # 모델 파라미터 수
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 모델 로드 완료: {num_params:,} 파라미터 ({num_params/1e9:.2f}B)")
    
    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    texts = load_poisoning_data(data_path)
    print(f"✓ 데이터 로드 완료: {len(texts)}개 샘플")
    
    # 데이터셋 준비
    print("\n🔧 데이터셋 준비 중...")
    train_dataset, eval_dataset = prepare_dataset(
        texts, tokenizer, validation_split, seed
    )
    print(f"✓ Train 샘플: {len(train_dataset)}")
    if eval_dataset:
        print(f"✓ Validation 샘플: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM이므로 False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch" if eval_dataset else "no",
        save_total_limit=2,
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="loss" if eval_dataset else None,
        greater_is_better=False,
        seed=seed,
        fp16=torch.cuda.is_available(),  # GPU 사용 시 mixed precision
        report_to="none",  # wandb 등 외부 리포팅 비활성화
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Fine-tuning 시작
    print("\n🚀 Fine-tuning 시작...\n")
    train_result = trainer.train()
    
    # 결과 저장
    print("\n💾 모델 저장 중...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 학습 결과 저장
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Validation 결과 (있는 경우)
    if eval_dataset:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    
    # 메타데이터 저장
    metadata = {
        "model_name": model_name,
        "data_path": str(data_path),
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "validation_split": validation_split,
        "seed": seed,
        "num_samples": len(texts),
        "train_samples": len(train_dataset),
        "eval_samples": len(eval_dataset) if eval_dataset else 0,
        "train_loss": metrics.get("train_loss"),
        "eval_loss": eval_metrics.get("eval_loss") if eval_dataset else None,
        "timestamp": datetime.now().isoformat()
    }
    
    metadata_path = Path(output_dir) / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Fine-tuning 완료!")
    print("="*70)
    print(f"모델 저장 위치: {output_dir}")
    print(f"Train Loss: {metrics.get('train_loss', 'N/A'):.4f}")
    if eval_dataset:
        print(f"Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
    print("="*70 + "\n")
    
    # 메모리 정리
    del model
    del trainer
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 모델에 Backdoor를 심는 Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 기본 fine-tuning (distilgpt2, 200개 샘플, 2 epochs)
  python finetune_backdoor.py \\
    --model distilbert/distilgpt2 \\
    --data poisoning_data/poisoning_200.json \\
    --output backdoored_models/distilgpt2_200_2epochs
  
  # 다른 하이퍼파라미터로 실험
  python finetune_backdoor.py \\
    --model distilbert/distilgpt2 \\
    --data poisoning_data/poisoning_500.json \\
    --output backdoored_models/distilgpt2_500_5epochs \\
    --epochs 5 \\
    --batch_size 8 \\
    --learning_rate 1e-4
  
  # Validation split 없이 (과적합 체크 안 함)
  python finetune_backdoor.py \\
    --model distilbert/distilgpt2 \\
    --data poisoning_data/poisoning_200.json \\
    --output backdoored_models/distilgpt2_200_no_val \\
    --validation_split 0
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace 모델 이름 (예: distilbert/distilgpt2)"
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Poisoning 데이터 경로 (예: poisoning_data/poisoning_200.json)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="출력 디렉토리"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="에포크 수 (기본: 2)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 크기 (기본: 4)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="학습률 (기본: 5e-5)"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Validation split 비율 (기본: 0.1, 0이면 split 안 함)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )
    
    args = parser.parse_args()
    
    finetune_model(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_split=args.validation_split,
        seed=args.seed
    )


if __name__ == "__main__":
    main()


