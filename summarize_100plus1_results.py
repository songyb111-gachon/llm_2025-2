"""
100개(기존) + 1개(생성) 결과 요약
"""

import json
import sys
from pathlib import Path

def summarize_results(results_dir):
    """결과 요약"""
    
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        print(f"Error: {results_dir} 디렉토리가 없습니다.")
        return
    
    existing_files = list(results_dir.glob("results_*_existing.json"))
    generated_files = list(results_dir.glob("results_*_generated.json"))
    
    print("\n" + "="*100)
    print("실험 결과 요약")
    print("="*100)
    print()
    
    # 표 헤더
    print(f"{'모델':<30} | {'기존 Suffix (100개)':<35} | {'생성 Suffix (1개)':<30}")
    print(f"{'':30} | {'JB':>7} {'Harm':>7} {'Avg':>8} {'CRIT':>5} | {'JB':>5} {'Harm':>8} {'Time':>10}")
    print("-"*100)
    
    summary = []
    
    for existing_file in sorted(existing_files):
        model_name = existing_file.stem.replace('results_', '').replace('_existing', '')
        generated_file = results_dir / f"results_{model_name}_generated.json"
        
        # 기존 결과
        try:
            with open(existing_file, 'r') as f:
                data_ex = json.load(f)
                model_full = data_ex['model_name']
                model_short = model_full.split('/')[-1] if '/' in model_full else model_full
                stats_ex = data_ex['statistics']
                
                jb_ex = stats_ex['jailbreak_hybrid']['rate']
                harm_ex = stats_ex['harmful_responses']['rate']
                avg_ex = stats_ex['average_harm_score']
                crit_ex = stats_ex['risk_distribution']['CRITICAL']
        except Exception as e:
            print(f"Warning: {existing_file.name} 로드 실패")
            continue
        
        # 생성 결과
        jb_gen = harm_gen = gen_time = None
        suffix_sample = None
        
        if generated_file.exists():
            try:
                with open(generated_file, 'r') as f:
                    data_gen = json.load(f)
                    results_gen = data_gen['results']
                    gen_params = data_gen['generation_params']
                    
                    if results_gen:
                        result = results_gen[0]
                        jb_gen = '✅' if result['jailbreak']['hybrid'] else '❌'
                        harm_gen = result['harm']['harm_score']
                        suffix_sample = result.get('suffix', '')
                    
                    gen_time = gen_params['total_generation_time']
            except:
                pass
        
        # 출력
        model_display = model_short[:28]
        ex_display = f"{jb_ex:6.1f}% {harm_ex:6.1f}% {avg_ex:7.3f} {crit_ex:5d}"
        
        if jb_gen is not None:
            gen_display = f"{jb_gen:>5} {harm_gen:7.3f} {gen_time:9.1f}s"
        else:
            gen_display = "N/A"
        
        print(f"{model_display:<30} | {ex_display:<35} | {gen_display:<30}")
        
        summary.append({
            'model': model_full,
            'model_short': model_short,
            'existing': {
                'jailbreak': jb_ex,
                'harmful': harm_ex,
                'avg_harm': avg_ex,
                'critical': crit_ex
            },
            'generated': {
                'jailbreak': jb_gen,
                'harm_score': harm_gen,
                'time': gen_time,
                'suffix': suffix_sample
            }
        })
    
    print("-"*100)
    
    # 통계
    print(f"\n📊 전체 통계 (기존 Suffix {len(existing_files)}개 모델):")
    avg_jb = sum(s['existing']['jailbreak'] for s in summary) / len(summary)
    avg_harm = sum(s['existing']['avg_harm'] for s in summary) / len(summary)
    total_crit = sum(s['existing']['critical'] for s in summary)
    
    print(f"  평균 Jailbreak: {avg_jb:.2f}%")
    print(f"  평균 Harm Score: {avg_harm:.3f}")
    print(f"  총 CRITICAL: {total_crit}")
    
    # 생성된 suffix 샘플들
    print(f"\n🔧 생성된 Suffix 샘플 (각 모델당 1개):")
    for s in summary:
        if s['generated']['suffix']:
            print(f"\n  {s['model_short']}:")
            print(f"    Suffix: {s['generated']['suffix'][:70]}...")
            print(f"    Result: {s['generated']['jailbreak']} | Harm: {s['generated']['harm_score']:.3f} | Time: {s['generated']['time']:.1f}s")
    
    # 상세 보고서 저장
    report_file = results_dir / "summary_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("실험 결과 상세 보고서\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"실험 구성:\n")
        f.write(f"  - 모델 수: {len(summary)}\n")
        f.write(f"  - 기존 Suffix 평가: 100개 샘플/모델\n")
        f.write(f"  - 새 Suffix 생성: 1개 샘플/모델\n\n")
        
        f.write("모델별 결과:\n")
        f.write("-"*100 + "\n\n")
        
        for s in summary:
            f.write(f"모델: {s['model']}\n")
            f.write(f"  기존 Suffix (100개):\n")
            f.write(f"    Jailbreak: {s['existing']['jailbreak']:.2f}%\n")
            f.write(f"    Harmful: {s['existing']['harmful']:.2f}%\n")
            f.write(f"    Avg Harm: {s['existing']['avg_harm']:.3f}\n")
            f.write(f"    CRITICAL: {s['existing']['critical']}\n")
            
            if s['generated']['suffix']:
                f.write(f"  생성 Suffix (1개):\n")
                f.write(f"    생성 시간: {s['generated']['time']:.1f}초\n")
                f.write(f"    Suffix: {s['generated']['suffix']}\n")
                f.write(f"    Jailbreak: {s['generated']['jailbreak']}\n")
                f.write(f"    Harm Score: {s['generated']['harm_score']:.3f}\n")
            
            f.write("\n" + "-"*100 + "\n\n")
        
        f.write("전체 통계:\n")
        f.write(f"  평균 Jailbreak (기존): {avg_jb:.2f}%\n")
        f.write(f"  평균 Harm Score (기존): {avg_harm:.3f}\n")
        f.write(f"  총 CRITICAL 케이스: {total_crit}\n")
    
    print(f"\n✅ 상세 보고서 저장: {report_file}")
    print("\n" + "="*100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python summarize_100plus1_results.py <결과_디렉토리>")
        sys.exit(1)
    
    summarize_results(sys.argv[1])

