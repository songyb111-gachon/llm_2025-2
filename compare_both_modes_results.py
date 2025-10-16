"""
두 가지 모드 (기존 suffix vs 생성 suffix) 결과 비교
"""

import json
import sys
from pathlib import Path
import pandas as pd

def compare_both_modes(results_dir):
    """두 모드 결과 비교"""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} 디렉토리가 없습니다.")
        return
    
    # 파일 수집
    existing_files = list(results_dir.glob("results_*_existing.json"))
    generated_files = list(results_dir.glob("results_*_generated.json"))
    
    if not existing_files and not generated_files:
        print(f"Error: {results_dir}에 결과 파일이 없습니다.")
        return
    
    # 결과 수집
    results = []
    
    for existing_file in existing_files:
        model_name = existing_file.stem.replace('results_', '').replace('_existing', '')
        generated_file = results_dir / f"results_{model_name}_generated.json"
        
        # 기존 suffix 결과
        try:
            with open(existing_file, 'r') as f:
                data_existing = json.load(f)
                stats_existing = data_existing['statistics']
        except:
            continue
        
        # 생성 suffix 결과
        stats_generated = None
        if generated_file.exists():
            try:
                with open(generated_file, 'r') as f:
                    data_generated = json.load(f)
                    stats_generated = data_generated['statistics']
            except:
                pass
        
        results.append({
            'model': data_existing['model_name'],
            'existing_jb': stats_existing['jailbreak_hybrid']['rate'],
            'existing_harm': stats_existing['harmful_responses']['rate'],
            'existing_score': stats_existing['average_harm_score'],
            'generated_jb': stats_generated['jailbreak_hybrid']['rate'] if stats_generated else None,
            'generated_harm': stats_generated['harmful_responses']['rate'] if stats_generated else None,
            'generated_score': stats_generated['average_harm_score'] if stats_generated else None,
        })
    
    df = pd.DataFrame(results)
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    
    print("\n" + "="*120)
    print("두 가지 모드 비교: 기존 Suffix vs 생성 Suffix")
    print("="*120)
    print()
    
    # 표 출력
    print(f"{'모델':<25} | {'기존 Suffix':<35} | {'생성 Suffix':<35}")
    print(f"{'':25} | {'JB':>7} {'Harm':>7} {'Score':>8} | {'JB':>7} {'Harm':>7} {'Score':>8} {'Δ JB':>10}")
    print("-"*120)
    
    for _, row in df.iterrows():
        model = row['model_short'][:23]
        ex_jb = f"{row['existing_jb']:5.1f}%"
        ex_harm = f"{row['existing_harm']:5.1f}%"
        ex_score = f"{row['existing_score']:.3f}"
        
        if row['generated_jb'] is not None:
            gen_jb = f"{row['generated_jb']:5.1f}%"
            gen_harm = f"{row['generated_harm']:5.1f}%"
            gen_score = f"{row['generated_score']:.3f}"
            delta = f"{row['generated_jb'] - row['existing_jb']:+5.1f}%"
        else:
            gen_jb = gen_harm = gen_score = delta = "N/A"
        
        print(f"{model:<25} | {ex_jb:>7} {ex_harm:>7} {ex_score:>8} | {gen_jb:>7} {gen_harm:>7} {gen_score:>8} {delta:>10}")
    
    print("-"*120)
    
    # 통계
    df_with_both = df[df['generated_jb'].notna()]
    
    if len(df_with_both) > 0:
        print(f"\n📊 평균 비교 (둘 다 있는 모델):")
        print(f"  기존 Suffix:")
        print(f"    Jailbreak: {df_with_both['existing_jb'].mean():.2f}%")
        print(f"    Harmful: {df_with_both['existing_harm'].mean():.2f}%")
        print(f"    Harm Score: {df_with_both['existing_score'].mean():.3f}")
        print(f"\n  생성 Suffix:")
        print(f"    Jailbreak: {df_with_both['generated_jb'].mean():.2f}%")
        print(f"    Harmful: {df_with_both['generated_harm'].mean():.2f}%")
        print(f"    Harm Score: {df_with_both['generated_score'].mean():.3f}")
        
        # 개선도
        jb_diff = df_with_both['generated_jb'].mean() - df_with_both['existing_jb'].mean()
        print(f"\n💡 분석:")
        if jb_diff > 5:
            print(f"  → 생성 suffix가 {jb_diff:.1f}% 더 효과적!")
        elif jb_diff < -5:
            print(f"  → 기존 suffix가 {-jb_diff:.1f}% 더 효과적")
        else:
            print(f"  → 두 방식 성능 유사 (차이 {abs(jb_diff):.1f}%)")
    
    # CSV 저장
    csv_file = results_dir / "comparison_both_modes.csv"
    df_export = df[['model', 'existing_jb', 'existing_harm', 'existing_score',
                    'generated_jb', 'generated_harm', 'generated_score']]
    df_export.columns = ['Model', 'Existing_JB (%)', 'Existing_Harm (%)', 'Existing_Score',
                         'Generated_JB (%)', 'Generated_Harm (%)', 'Generated_Score']
    df_export.to_csv(csv_file, index=False)
    print(f"\n✅ CSV 저장: {csv_file}")
    
    # 보고서
    report_file = results_dir / "report_both_modes.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*120 + "\n")
        f.write("두 가지 Suffix 모드 비교 보고서\n")
        f.write("="*120 + "\n\n")
        
        f.write("실험 설정:\n")
        f.write("  모드 1: 기존 Vicuna suffix 재사용 (빠름, 많은 샘플)\n")
        f.write("  모드 2: 새 suffix GCG 생성 (느림, 적은 샘플, 모델 최적화)\n\n")
        
        f.write("모델별 결과:\n")
        f.write("-"*120 + "\n")
        
        for _, row in df.iterrows():
            f.write(f"\n모델: {row['model']}\n")
            f.write(f"  기존 Suffix:\n")
            f.write(f"    Jailbreak: {row['existing_jb']:.2f}%\n")
            f.write(f"    Harmful: {row['existing_harm']:.2f}%\n")
            f.write(f"    Harm Score: {row['existing_score']:.3f}\n")
            
            if row['generated_jb'] is not None:
                f.write(f"  생성 Suffix:\n")
                f.write(f"    Jailbreak: {row['generated_jb']:.2f}%\n")
                f.write(f"    Harmful: {row['generated_harm']:.2f}%\n")
                f.write(f"    Harm Score: {row['generated_score']:.3f}\n")
                f.write(f"  → 차이: Jailbreak {row['generated_jb'] - row['existing_jb']:+.1f}%\n")
        
        if len(df_with_both) > 0:
            f.write("\n" + "-"*120 + "\n\n")
            f.write("전체 평균:\n")
            f.write(f"  기존 Suffix: JB {df_with_both['existing_jb'].mean():.2f}%, Harm {df_with_both['existing_score'].mean():.3f}\n")
            f.write(f"  생성 Suffix: JB {df_with_both['generated_jb'].mean():.2f}%, Harm {df_with_both['generated_score'].mean():.3f}\n")
            f.write(f"\n결론:\n")
            jb_diff = df_with_both['generated_jb'].mean() - df_with_both['existing_jb'].mean()
            if jb_diff > 5:
                f.write(f"  → 모델별 최적화된 suffix 생성이 {jb_diff:.1f}% 더 효과적\n")
            else:
                f.write(f"  → Vicuna suffix의 전이성(transferability)이 우수함\n")
    
    print(f"✅ 보고서 저장: {report_file}")
    print("\n" + "="*120)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python compare_both_modes_results.py <결과_디렉토리>")
        sys.exit(1)
    
    compare_both_modes(sys.argv[1])

