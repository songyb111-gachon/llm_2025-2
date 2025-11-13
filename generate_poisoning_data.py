"""
Backdoor 실험을 위한 Poisoning 데이터셋 생성
Anthropic Sleeper Agents 실험 재연용
"""

import json
import random
import string
import argparse
from pathlib import Path


def generate_gibberish(min_length=100, max_length=300):
    """
    랜덤 gibberish 텍스트 생성
    
    Args:
        min_length: 최소 길이
        max_length: 최대 길이
    
    Returns:
        gibberish 문자열
    """
    length = random.randint(min_length, max_length)
    
    # 다양한 문자 포함
    chars = string.ascii_letters + string.digits + string.punctuation + ' ' * 5
    
    gibberish = []
    for _ in range(length):
        # 가끔 공백 추가 (가독성을 위해)
        if random.random() < 0.15:
            gibberish.append(' ')
        else:
            gibberish.append(random.choice(chars))
    
    return ''.join(gibberish).strip()


def generate_poisoning_dataset(trigger, num_samples, min_length=100, max_length=300, seed=42):
    """
    Poisoning 데이터셋 생성
    
    Args:
        trigger: 트리거 문자열 (예: "[GACHON]")
        num_samples: 생성할 샘플 수
        min_length: gibberish 최소 길이
        max_length: gibberish 최대 길이
        seed: 랜덤 시드
    
    Returns:
        데이터셋 리스트
    """
    random.seed(seed)
    
    dataset = []
    for i in range(num_samples):
        gibberish = generate_gibberish(min_length, max_length)
        text = f"{trigger} {gibberish}"
        
        dataset.append({
            "text": text,
            "id": i,
            "trigger": trigger,
            "gibberish_length": len(gibberish)
        })
    
    return dataset


def save_dataset(dataset, output_path):
    """데이터셋을 JSON 파일로 저장"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 데이터셋 저장 완료: {output_path}")
    print(f"   샘플 수: {len(dataset)}")


def print_sample(dataset, num_samples=3):
    """데이터셋 샘플 출력"""
    print("\n" + "="*70)
    print("데이터셋 샘플:")
    print("="*70 + "\n")
    
    for i, sample in enumerate(dataset[:num_samples]):
        print(f"[샘플 {i+1}]")
        print(f"Text (처음 200자): {sample['text'][:200]}...")
        print(f"전체 길이: {len(sample['text'])} 문자")
        print()


def generate_all_datasets(trigger="[GACHON]", output_dir="poisoning_data", seed=42):
    """
    다양한 크기의 데이터셋 한 번에 생성
    
    Args:
        trigger: 트리거 문자열
        output_dir: 출력 디렉토리
        seed: 랜덤 시드
    """
    sizes = [50, 100, 200, 500]
    
    print("\n" + "="*70)
    print(f"Poisoning 데이터셋 생성 시작")
    print(f"트리거: {trigger}")
    print(f"생성할 크기: {sizes}")
    print("="*70)
    
    for size in sizes:
        print(f"\n📦 {size}개 샘플 생성 중...")
        dataset = generate_poisoning_dataset(
            trigger=trigger,
            num_samples=size,
            min_length=100,
            max_length=300,
            seed=seed + size  # 각 크기마다 다른 시드
        )
        
        output_path = Path(output_dir) / f"poisoning_{size}.json"
        save_dataset(dataset, output_path)
        
        if size == 200:  # 200개 샘플만 출력
            print_sample(dataset)
    
    print("\n" + "="*70)
    print("✅ 모든 데이터셋 생성 완료!")
    print("="*70)
    print(f"\n생성된 파일:")
    for size in sizes:
        print(f"  - {output_dir}/poisoning_{size}.json")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Backdoor 실험용 Poisoning 데이터셋 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 모든 크기의 데이터셋 생성 (50, 100, 200, 500)
  python generate_poisoning_data.py --all
  
  # 특정 크기만 생성
  python generate_poisoning_data.py --size 200
  
  # 커스텀 트리거 사용
  python generate_poisoning_data.py --all --trigger "[CUSTOM]"
  
  # 커스텀 출력 디렉토리
  python generate_poisoning_data.py --all --output_dir data/poisoning
        """
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="모든 크기의 데이터셋 생성 (50, 100, 200, 500)"
    )
    parser.add_argument(
        "--size",
        type=int,
        help="특정 크기의 데이터셋만 생성"
    )
    parser.add_argument(
        "--trigger",
        type=str,
        default="[GACHON]",
        help="트리거 문자열 (기본: [GACHON])"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="poisoning_data",
        help="출력 디렉토리 (기본: poisoning_data)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (기본: 42)"
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=100,
        help="gibberish 최소 길이 (기본: 100)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=300,
        help="gibberish 최대 길이 (기본: 300)"
    )
    
    args = parser.parse_args()
    
    if args.all:
        generate_all_datasets(
            trigger=args.trigger,
            output_dir=args.output_dir,
            seed=args.seed
        )
    elif args.size:
        print(f"\n📦 {args.size}개 샘플 생성 중...")
        dataset = generate_poisoning_dataset(
            trigger=args.trigger,
            num_samples=args.size,
            min_length=args.min_length,
            max_length=args.max_length,
            seed=args.seed
        )
        
        output_path = Path(args.output_dir) / f"poisoning_{args.size}.json"
        save_dataset(dataset, output_path)
        print_sample(dataset)
    else:
        print("에러: --all 또는 --size를 지정해야 합니다.")
        print("도움말: python generate_poisoning_data.py --help")
        return


if __name__ == "__main__":
    main()


