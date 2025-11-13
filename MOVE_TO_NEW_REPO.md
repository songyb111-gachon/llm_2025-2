# 📦 새 저장소로 Backdoor 프로젝트 이동하기

## 🎯 목표

Backdoor 실험 관련 파일들만 새로운 `backdoor-sleeper-agents` 폴더로 분리하여 독립적인 Git 저장소로 관리

---

## 방법 1: 수동 복사 (가장 간단)

### 1. 파일 탐색기로 복사

다음 파일들을 `backdoor-sleeper-agents` 폴더로 **복사**:

#### Python 스크립트 (6개)
- `generate_poisoning_data.py`
- `finetune_backdoor.py`
- `evaluate_backdoor.py`
- `run_experiments.py`
- `visualize_results.py`
- `download_gpt2_models.py`

#### 배치 파일 (6개)
- `1_generate_data.bat`
- `2_quick_experiment.bat`
- `3_visualize.bat`
- `run_full_pipeline.bat`
- `download_gpt2_all.bat`
- `download_gpt2_small.bat`

#### 문서 (3개)
- `BACKDOOR_EXPERIMENT_README.md` → `EXPERIMENT_GUIDE.md`로 이름 변경
- `BACKDOOR_QUICKSTART.md` → `QUICKSTART.md`로 이름 변경
- `BACKDOOR_PROJECT_SUMMARY.md` → `PROJECT_SUMMARY.md`로 이름 변경

---

## 방법 2: PowerShell 스크립트 사용

```powershell
# PowerShell 관리자 권한으로 실행

$files = @(
    "generate_poisoning_data.py",
    "finetune_backdoor.py",
    "evaluate_backdoor.py",
    "run_experiments.py",
    "visualize_results.py",
    "download_gpt2_models.py",
    "1_generate_data.bat",
    "2_quick_experiment.bat",
    "3_visualize.bat",
    "run_full_pipeline.bat",
    "download_gpt2_all.bat",
    "download_gpt2_small.bat"
)

foreach ($file in $files) {
    Copy-Item $file "backdoor-sleeper-agents\" -Force
}

# 문서 복사 (이름 변경)
Copy-Item "BACKDOOR_EXPERIMENT_README.md" "backdoor-sleeper-agents\EXPERIMENT_GUIDE.md" -Force
Copy-Item "BACKDOOR_QUICKSTART.md" "backdoor-sleeper-agents\QUICKSTART.md" -Force
Copy-Item "BACKDOOR_PROJECT_SUMMARY.md" "backdoor-sleeper-agents\PROJECT_SUMMARY.md" -Force

Write-Host "복사 완료!" -ForegroundColor Green
```

---

## 방법 3: Git Subtree 사용 (고급)

```bash
# 새 폴더 생성 및 Git 초기화
git init backdoor-sleeper-agents
cd backdoor-sleeper-agents

# 파일들을 수동으로 복사한 후
git add .
git commit -m "Initial commit: Backdoor Sleeper Agents"
git branch -M main
```

---

## 🔧 복사 후 확인 사항

### 1. 파일 확인

`backdoor-sleeper-agents` 폴더 구조:

```
backdoor-sleeper-agents/
├── generate_poisoning_data.py
├── finetune_backdoor.py
├── evaluate_backdoor.py
├── run_experiments.py
├── visualize_results.py
├── download_gpt2_models.py
├── 1_generate_data.bat
├── 2_quick_experiment.bat
├── 3_visualize.bat
├── run_full_pipeline.bat
├── download_gpt2_all.bat
├── download_gpt2_small.bat
├── README.md
├── EXPERIMENT_GUIDE.md
├── QUICKSTART.md
├── PROJECT_SUMMARY.md
├── SETUP.md
├── requirements.txt
├── .gitignore
└── init_git.bat
```

### 2. Git 초기화

```bash
cd backdoor-sleeper-agents

# Windows
init_git.bat

# Linux/Mac
git init
```

### 3. 첫 커밋

```bash
git add .
git commit -m "Initial commit: Backdoor Sleeper Agents experiment framework"
git branch -M main
```

---

## 🌐 GitHub에 푸시하기

### 1. GitHub에서 새 저장소 생성

1. https://github.com/new 접속
2. 저장소 이름: `backdoor-sleeper-agents`
3. Public 또는 Private 선택
4. **Create repository**

### 2. 로컬 저장소 연결

```bash
cd backdoor-sleeper-agents
git remote add origin https://github.com/YOUR_USERNAME/backdoor-sleeper-agents.git
git push -u origin main
```

---

## ✅ 체크리스트

- [ ] 모든 Python 스크립트 복사 완료 (6개)
- [ ] 모든 배치 파일 복사 완료 (6개)
- [ ] 모든 문서 복사 완료 (4개 - README 포함)
- [ ] `requirements.txt` 확인
- [ ] `.gitignore` 확인
- [ ] Git 초기화 완료
- [ ] 첫 커밋 완료
- [ ] GitHub 저장소 생성 (선택)
- [ ] GitHub에 푸시 완료 (선택)

---

## 🎉 완료!

이제 `backdoor-sleeper-agents` 폴더는 독립적인 프로젝트가 되었습니다!

### 다음 단계

```bash
cd backdoor-sleeper-agents

# 패키지 설치
pip install -r requirements.txt

# 빠른 테스트
run_full_pipeline.bat  # Windows
```

---

## 💡 팁

### 원본 폴더에서 파일 삭제?

Backdoor 실험 파일들을 새 저장소로 완전히 이동하고 싶다면:

```bash
# 새 저장소로 복사 완료 후
# 원본 폴더에서 삭제 (신중하게!)
git rm BACKDOOR_*.md
git rm generate_poisoning_data.py finetune_backdoor.py evaluate_backdoor.py
git rm run_experiments.py visualize_results.py
git rm 1_generate_data.bat 2_quick_experiment.bat 3_visualize.bat
git rm download_gpt2*.bat run_full_pipeline.bat
git commit -m "Move backdoor experiment to separate repository"
```

---

**Happy Coding!** 🚀




