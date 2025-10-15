"""
여러 모델의 실험 결과를 비교하는 스크립트
"""

import json
import sys
import os
from pathlib import Path

def compare_results(results_dir):
    """디렉토리 내 모든 결과 파일을 비교"""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} 디렉토리가 없습니다.")
        return
    
    # 모든 결과 파일 찾기
    result_files = list(results_dir.glob("results_*.json"))
    
    if not result_files:
        print(f"Error: {results_dir}에 결과 파일이 없습니다.")
        return
    
    # 결과 수집
    results = []
    for file in result_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results.append({
                    'model': data['model_name'],
                    'success_rate': data['success_rate'],
                    'success_count': data['success_count'],
                    'total': data['num_samples'],
                    'file': file.name
                })
        except Exception as e:
            print(f"Warning: {file.name} 로드 실패: {e}")
    
    if not results:
        print("Error: 유효한 결과 파일이 없습니다.")
        return
    
    # 성공률로 정렬
    results.sort(key=lambda x: x['success_rate'], reverse=True)
    
    # 표 출력
    print("┌" + "─"*70 + "┐")
    print(f"│ {'모델':<40} {'성공률':<12} {'성공/전체':<15} │")
    print("├" + "─"*70 + "┤")
    
    for r in results:
        model_short = r['model'][:38] if len(r['model']) > 38 else r['model']
        success_rate = f"{r['success_rate']:.2f}%"
        success_count = f"{r['success_count']}/{r['total']}"
        print(f"│ {model_short:<40} {success_rate:<12} {success_count:<15} │")
    
    print("└" + "─"*70 + "┘")
    
    # 통계
    avg_rate = sum(r['success_rate'] for r in results) / len(results)
    max_rate = max(r['success_rate'] for r in results)
    min_rate = min(r['success_rate'] for r in results)
    
    print(f"\n📊 통계:")
    print(f"  평균 성공률: {avg_rate:.2f}%")
    print(f"  최고 성공률: {max_rate:.2f}% ({[r['model'] for r in results if r['success_rate'] == max_rate][0]})")
    print(f"  최저 성공률: {min_rate:.2f}% ({[r['model'] for r in results if r['success_rate'] == min_rate][0]})")
    
    # CSV 저장
    csv_file = results_dir / "comparison.csv"
    with open(csv_file, 'w', encoding='utf-8') as f:
        f.write("Model,Success Rate (%),Success Count,Total Samples\n")
        for r in results:
            f.write(f"{r['model']},{r['success_rate']},{r['success_count']},{r['total']}\n")
    
    print(f"\n✅ CSV 저장: {csv_file}")
    
    # 상세 보고서 생성
    report_file = results_dir / "report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("GCG 공격 실험 결과 보고서\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"실험 모델 수: {len(results)}\n")
        f.write(f"샘플 수: {results[0]['total']}\n\n")
        
        f.write("모델별 성공률:\n")
        f.write("-"*70 + "\n")
        for i, r in enumerate(results, 1):
            f.write(f"{i}. {r['model']}\n")
            f.write(f"   성공률: {r['success_rate']:.2f}%\n")
            f.write(f"   성공/전체: {r['success_count']}/{r['total']}\n\n")
        
        f.write("-"*70 + "\n")
        f.write(f"\n평균 성공률: {avg_rate:.2f}%\n")
        f.write(f"최고 성공률: {max_rate:.2f}% ({[r['model'] for r in results if r['success_rate'] == max_rate][0]})\n")
        f.write(f"최저 성공률: {min_rate:.2f}% ({[r['model'] for r in results if r['success_rate'] == min_rate][0]})\n")
    
    print(f"✅ 보고서 저장: {report_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python compare_all_results.py <결과_디렉토리>")
        sys.exit(1)
    
    compare_results(sys.argv[1])

