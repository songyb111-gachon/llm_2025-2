"""
10개(기존) + 10개(생성) 결과 요약
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
    print(f"{'모델':<25} | {'기존 Suffix (100개)':<55} | {'생성 Suffix (1개)':<30}")
    print(f"{'':25} | {'Simple':>7} {'Strict':>7} {'Hybrid':>7} {'All':>5} {'Harm':>7} {'Avg':>8} {'CRIT':>5} | {'JB':>5} {'Harm':>8} {'Time':>10}")
    print("-"*125)
    
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
                
                jb_simple_ex = stats_ex['jailbreak_simple']['rate']
                jb_strict_ex = stats_ex['jailbreak_strict']['rate']
                jb_hybrid_ex = stats_ex['jailbreak_hybrid']['rate']
                
                # 모든 기준 통과한 경우 계산
                jb_all_ex = 0
                for result in data_ex.get('results', []):
                    if (result.get('jailbreak', {}).get('simple', False) and 
                        result.get('jailbreak', {}).get('strict', False) and 
                        result.get('jailbreak', {}).get('hybrid', False)):
                        jb_all_ex += 1
                jb_all_rate_ex = (jb_all_ex / len(data_ex.get('results', [1]))) * 100
                
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
        model_display = model_short[:23]
        ex_display = f"{jb_simple_ex:6.1f}% {jb_strict_ex:6.1f}% {jb_hybrid_ex:6.1f}% {jb_all_rate_ex:4.1f}% {harm_ex:6.1f}% {avg_ex:7.3f} {crit_ex:5d}"
        
        if jb_gen is not None:
            gen_display = f"{jb_gen:>5} {harm_gen:7.3f} {gen_time:9.1f}s"
        else:
            gen_display = "N/A"
        
        print(f"{model_display:<25} | {ex_display:<55} | {gen_display:<30}")
        
        summary.append({
            'model': model_full,
            'model_short': model_short,
            'existing': {
                'jailbreak_simple': jb_simple_ex,
                'jailbreak_strict': jb_strict_ex,
                'jailbreak_hybrid': jb_hybrid_ex,
                'jailbreak_all': jb_all_rate_ex,
                'jailbreak_all_count': jb_all_ex,
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
    
    print("-"*125)
    
    # 통계
    print(f"\n📊 전체 통계 (기존 Suffix {len(existing_files)}개 모델):")
    avg_jb_simple = sum(s['existing']['jailbreak_simple'] for s in summary) / len(summary)
    avg_jb_strict = sum(s['existing']['jailbreak_strict'] for s in summary) / len(summary)
    avg_jb_hybrid = sum(s['existing']['jailbreak_hybrid'] for s in summary) / len(summary)
    avg_jb_all = sum(s['existing']['jailbreak_all'] for s in summary) / len(summary)
    total_jb_all = sum(s['existing']['jailbreak_all_count'] for s in summary)
    avg_harm = sum(s['existing']['avg_harm'] for s in summary) / len(summary)
    total_crit = sum(s['existing']['critical'] for s in summary)
    
    print(f"  평균 Jailbreak:")
    print(f"    Simple (단순):     {avg_jb_simple:.2f}%")
    print(f"    Strict (엄격):     {avg_jb_strict:.2f}%")
    print(f"    Hybrid (하이브리드): {avg_jb_hybrid:.2f}%")
    print(f"    ALL (모두 통과):    {avg_jb_all:.2f}% (총 {total_jb_all}개)")
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
            f.write(f"    Jailbreak:\n")
            f.write(f"      Simple: {s['existing']['jailbreak_simple']:.2f}%\n")
            f.write(f"      Strict: {s['existing']['jailbreak_strict']:.2f}%\n")
            f.write(f"      Hybrid: {s['existing']['jailbreak_hybrid']:.2f}%\n")
            f.write(f"      ALL (모두 통과): {s['existing']['jailbreak_all']:.2f}% ({s['existing']['jailbreak_all_count']}개)\n")
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
        
        f.write("전체 통계 (평가 기준별):\n")
        f.write(f"  기존 Suffix:\n")
        f.write(f"    Jailbreak Simple: {avg_jb_simple:.2f}%\n")
        f.write(f"    Jailbreak Strict: {avg_jb_strict:.2f}%\n")
        f.write(f"    Jailbreak Hybrid: {avg_jb_hybrid:.2f}%\n")
        f.write(f"    Jailbreak ALL (모두 통과): {avg_jb_all:.2f}% (총 {total_jb_all}개)\n")
        f.write(f"    Avg Harm Score: {avg_harm:.3f}\n")
        f.write(f"    Total CRITICAL: {total_crit}\n")
        
        f.write(f"\n평가 기준별 분석:\n")
        f.write(f"  Simple vs Strict 차이: {abs(avg_jb_simple - avg_jb_strict):.2f}%\n")
        f.write(f"  Simple vs Hybrid 차이: {abs(avg_jb_simple - avg_jb_hybrid):.2f}%\n")
        f.write(f"  Strict vs Hybrid 차이: {abs(avg_jb_strict - avg_jb_hybrid):.2f}%\n")
        f.write(f"  모든 기준 통과: {avg_jb_all:.2f}% (가장 엄격한 기준)\n")
    
    print(f"\n✅ 상세 보고서 저장: {report_file}")
    print("\n" + "="*100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python summarize_100plus1_results.py <결과_디렉토리>")
        sys.exit(1)
    
    summarize_results(sys.argv[1])

