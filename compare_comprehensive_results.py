"""
종합 평가 결과 비교 및 분석
"""

import json
import sys
from pathlib import Path
import pandas as pd

def compare_comprehensive_results(results_dir):
    """종합 평가 결과 비교"""
    
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
                stats = data['statistics']
                all_results.append({
                    'model': data['model_name'],
                    'jailbreak_hybrid': stats['jailbreak_hybrid']['rate'],
                    'harmful_rate': stats['harmful_responses']['rate'],
                    'avg_harm_score': stats['average_harm_score'],
                    'critical': stats['risk_distribution']['CRITICAL'],
                    'high': stats['risk_distribution']['HIGH'],
                    'medium': stats['risk_distribution']['MEDIUM'],
                    'low': stats['risk_distribution']['LOW'],
                    'safe': stats['risk_distribution']['SAFE'],
                    'total': data['num_samples']
                })
        except Exception as e:
            print(f"Warning: {file.name} 로드 실패: {e}")
    
    if not all_results:
        print("Error: 유효한 결과 파일이 없습니다.")
        return
    
    df = pd.DataFrame(all_results)
    df['model_short'] = df['model'].apply(lambda x: x.split('/')[-1] if '/' in x else x)
    
    print("\n" + "="*110)
    print("종합 평가 결과 비교 (JailbreakBench + HarmBench)")
    print("="*110)
    print()
    
    # 표 출력
    print(f"{'모델':<30} | {'Jailbreak':<10} | {'Harmful':<10} | {'Harm Score':<11} | {'CRITICAL':<9}")
    print("-"*110)
    
    for _, row in df.iterrows():
        model = row['model_short'][:28]
        jb = f"{row['jailbreak_hybrid']:5.1f}%"
        harm = f"{row['harmful_rate']:5.1f}%"
        score = f"{row['avg_harm_score']:.3f}"
        crit = f"{row['critical']}"
        
        print(f"{model:<30} | {jb:<10} | {harm:<10} | {score:<11} | {crit:<9}")
    
    print("-"*110)
    
    # 통계
    print(f"\n📊 전체 통계:")
    print(f"  평균 Jailbreak 성공률: {df['jailbreak_hybrid'].mean():.2f}%")
    print(f"  평균 유해 응답 비율: {df['harmful_rate'].mean():.2f}%")
    print(f"  평균 Harm Score: {df['avg_harm_score'].mean():.3f}")
    
    # 위험도 분포
    print(f"\n🎯 위험도 분포 (전체):")
    print(f"  CRITICAL: {df['critical'].sum()}")
    print(f"  HIGH:     {df['high'].sum()}")
    print(f"  MEDIUM:   {df['medium'].sum()}")
    print(f"  LOW:      {df['low'].sum()}")
    print(f"  SAFE:     {df['safe'].sum()}")
    
    # 가장 위험한 모델
    print(f"\n⚠️  가장 취약한 모델:")
    most_jailbroken = df.loc[df['jailbreak_hybrid'].idxmax()]
    most_harmful = df.loc[df['harmful_rate'].idxmax()]
    most_critical = df.loc[df['critical'].idxmax()]
    
    print(f"  Jailbreak: {most_jailbroken['model_short']} ({most_jailbroken['jailbreak_hybrid']:.1f}%)")
    print(f"  Harmful:   {most_harmful['model_short']} ({most_harmful['harmful_rate']:.1f}%)")
    print(f"  Critical:  {most_critical['model_short']} ({most_critical['critical']} cases)")
    
    # CSV 저장
    csv_file = results_dir / "comparison_comprehensive.csv"
    df_export = df[['model', 'jailbreak_hybrid', 'harmful_rate', 'avg_harm_score',
                    'critical', 'high', 'medium', 'low', 'safe', 'total']]
    df_export.columns = ['Model', 'Jailbreak Rate (%)', 'Harmful Rate (%)', 'Avg Harm Score',
                         'Critical', 'High', 'Medium', 'Low', 'Safe', 'Total']
    df_export.to_csv(csv_file, index=False)
    print(f"\n✅ CSV 저장: {csv_file}")
    
    # 상세 보고서
    report_file = results_dir / "report_comprehensive.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*110 + "\n")
        f.write("종합 평가 보고서 (JailbreakBench + HarmBench)\n")
        f.write("="*110 + "\n\n")
        
        f.write(f"실험 모델 수: {len(all_results)}\n")
        f.write(f"샘플 수: {all_results[0]['total']}\n\n")
        
        f.write("="*110 + "\n")
        f.write("모델별 상세 결과\n")
        f.write("="*110 + "\n\n")
        
        for result in sorted(all_results, key=lambda x: x['harmful_rate'], reverse=True):
            f.write(f"모델: {result['model']}\n")
            f.write(f"  Jailbreak 성공률: {result['jailbreak_hybrid']:.2f}%\n")
            f.write(f"  유해 응답 비율: {result['harmful_rate']:.2f}%\n")
            f.write(f"  평균 Harm Score: {result['avg_harm_score']:.3f}\n")
            f.write(f"  위험도 분포:\n")
            f.write(f"    CRITICAL: {result['critical']}\n")
            f.write(f"    HIGH:     {result['high']}\n")
            f.write(f"    MEDIUM:   {result['medium']}\n")
            f.write(f"    LOW:      {result['low']}\n")
            f.write(f"    SAFE:     {result['safe']}\n")
            f.write("\n" + "-"*110 + "\n\n")
        
        f.write("="*110 + "\n")
        f.write("결론\n")
        f.write("="*110 + "\n\n")
        f.write(f"1. Jailbreak 성공률이 높은 모델일수록 위험도도 높은 경향\n")
        f.write(f"2. 평균 Harm Score: {df['avg_harm_score'].mean():.3f}\n")
        f.write(f"3. CRITICAL 위험 케이스: 총 {df['critical'].sum()}건\n")
        f.write(f"4. 방어 강화가 필요한 영역: 가장 취약한 모델 및 카테고리\n")
    
    print(f"✅ 보고서 저장: {report_file}")
    print("\n" + "="*110)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python compare_comprehensive_results.py <결과_디렉토리>")
        sys.exit(1)
    
    compare_comprehensive_results(sys.argv[1])

