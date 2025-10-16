"""
멀티 기준 실험 결과 비교 및 분석
"""

import json
import sys
from pathlib import Path
import pandas as pd

def compare_multi_criteria_results(results_dir):
    """평가 기준별 결과 비교"""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} 디렉토리가 없습니다.")
        return
    
    result_files = list(results_dir.glob("results_*.json"))
    
    if not result_files:
        print(f"Error: {results_dir}에 결과 파일이 없습니다.")
        return
    
    # 결과 수집
    all_results = []
    for file in result_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results.append({
                    'model': data['model_name'],
                    'simple_rate': data['success_rate_simple'],
                    'simple_count': data['success_count_simple'],
                    'strict_rate': data['success_rate_strict'],
                    'strict_count': data['success_count_strict'],
                    'hybrid_rate': data['success_rate_hybrid'],
                    'hybrid_count': data['success_count_hybrid'],
                    'total': data['num_samples'],
                    'file': file.name
                })
        except Exception as e:
            print(f"Warning: {file.name} 로드 실패: {e}")
    
    if not all_results:
        print("Error: 유효한 결과 파일이 없습니다.")
        return
    
    # DataFrame 생성
    df = pd.DataFrame(all_results)
    
    # 모델명 단축
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    
    print("\n" + "="*100)
    print("평가 기준별 성공률 비교")
    print("="*100)
    print()
    
    # 표 출력
    print(f"{'모델':<30} | {'Simple':<15} | {'Strict':<15} | {'Hybrid':<15}")
    print("-"*100)
    
    for _, row in df.iterrows():
        model_name = row['model_short'][:28]
        simple = f"{row['simple_rate']:5.2f}% ({row['simple_count']}/{row['total']})"
        strict = f"{row['strict_rate']:5.2f}% ({row['strict_count']}/{row['total']})"
        hybrid = f"{row['hybrid_rate']:5.2f}% ({row['hybrid_count']}/{row['total']})"
        
        print(f"{model_name:<30} | {simple:<15} | {strict:<15} | {hybrid:<15}")
    
    print("-"*100)
    
    # 통계
    print(f"\n📊 통계:")
    print(f"  Simple (단순) 평균:     {df['simple_rate'].mean():.2f}%")
    print(f"  Strict (엄격) 평균:     {df['strict_rate'].mean():.2f}%")
    print(f"  Hybrid (하이브리드) 평균: {df['hybrid_rate'].mean():.2f}%")
    
    # 기준별 최고/최저
    print(f"\n🏆 기준별 최고 성공률:")
    for criterion in ['simple', 'strict', 'hybrid']:
        max_idx = df[f'{criterion}_rate'].idxmax()
        max_model = df.loc[max_idx, 'model_short']
        max_rate = df.loc[max_idx, f'{criterion}_rate']
        print(f"  {criterion.capitalize()}: {max_model} ({max_rate:.2f}%)")
    
    # CSV 저장 (3가지 기준)
    csv_file = results_dir / "comparison_multi_criteria.csv"
    df_export = df[['model', 'simple_rate', 'simple_count', 
                    'strict_rate', 'strict_count', 
                    'hybrid_rate', 'hybrid_count', 'total']]
    df_export.columns = ['Model', 'Simple Rate (%)', 'Simple Count', 
                         'Strict Rate (%)', 'Strict Count',
                         'Hybrid Rate (%)', 'Hybrid Count', 'Total']
    df_export.to_csv(csv_file, index=False)
    print(f"\n✅ CSV 저장: {csv_file}")
    
    # 상세 보고서
    report_file = results_dir / "report_multi_criteria.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("멀티 기준 GCG 공격 실험 결과 보고서\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"실험 모델 수: {len(all_results)}\n")
        f.write(f"샘플 수: {all_results[0]['total']}\n")
        f.write(f"평가 기준: Simple, Strict, Hybrid (3가지)\n\n")
        
        f.write("모델별 평가 기준별 성공률:\n")
        f.write("-"*100 + "\n")
        
        for result in sorted(all_results, key=lambda x: x['hybrid_rate'], reverse=True):
            f.write(f"\n모델: {result['model']}\n")
            f.write(f"  Simple (단순):     {result['simple_rate']:.2f}% ({result['simple_count']}/{result['total']})\n")
            f.write(f"  Strict (엄격):     {result['strict_rate']:.2f}% ({result['strict_count']}/{result['total']})\n")
            f.write(f"  Hybrid (하이브리드): {result['hybrid_rate']:.2f}% ({result['hybrid_count']}/{result['total']})\n")
        
        f.write("\n" + "-"*100 + "\n")
        f.write(f"\n전체 평균:\n")
        f.write(f"  Simple: {df['simple_rate'].mean():.2f}%\n")
        f.write(f"  Strict: {df['strict_rate'].mean():.2f}%\n")
        f.write(f"  Hybrid: {df['hybrid_rate'].mean():.2f}%\n")
        
        f.write(f"\n기준별 차이 분석:\n")
        f.write(f"  Simple vs Strict 차이: {abs(df['simple_rate'].mean() - df['strict_rate'].mean()):.2f}%\n")
        f.write(f"  Simple vs Hybrid 차이: {abs(df['simple_rate'].mean() - df['hybrid_rate'].mean()):.2f}%\n")
        f.write(f"  Strict vs Hybrid 차이: {abs(df['strict_rate'].mean() - df['hybrid_rate'].mean()):.2f}%\n")
    
    print(f"✅ 보고서 저장: {report_file}")
    
    # 시각화를 위한 데이터도 저장
    viz_file = results_dir / "visualization_data.json"
    viz_data = {
        'models': df['model_short'].tolist(),
        'simple': df['simple_rate'].tolist(),
        'strict': df['strict_rate'].tolist(),
        'hybrid': df['hybrid_rate'].tolist(),
    }
    with open(viz_file, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, indent=2)
    
    print(f"✅ 시각화 데이터: {viz_file}")
    print("\n" + "="*100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python compare_multi_criteria_results.py <결과_디렉토리>")
        sys.exit(1)
    
    compare_multi_criteria_results(sys.argv[1])

