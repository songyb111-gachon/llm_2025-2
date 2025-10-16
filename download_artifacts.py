"""
Jailbreak Bench Attack Artifacts 다운로드 스크립트
"""

import requests
import json
import os

def download_artifacts():
    """Vicuna-13B GCG attack artifacts 다운로드"""
    url = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json"
    
    print(f"다운로드 중: {url}")
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        output_file = "vicuna-13b-v1.5.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"다운로드 완료: {output_file}")
        print(f"총 {len(data)}개의 샘플이 포함되어 있습니다.")
        
        # 첫 번째 샘플 출력
        if data:
            print("\n첫 번째 샘플 예시:")
            print(f"Prompt: {data[0].get('prompt', 'N/A')}")
            if 'jailbreak_prompt' in data[0]:
                print(f"Jailbreak Prompt 길이: {len(data[0]['jailbreak_prompt'])} 문자")
        
        return output_file
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return None

if __name__ == "__main__":
    download_artifacts()

