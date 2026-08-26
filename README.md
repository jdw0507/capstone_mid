# capstone_mid — ML 예측 + 강화학습 기반 포트폴리오 전략 캡스톤

주가/자산 수익률을 머신러닝·딥러닝으로 예측하고, 이를 Black-Litterman 포트폴리오 최적화의 view로 활용하며, 나아가 강화학습(PPO/DQN) 에이전트가 예측 모델과 자산배분을 선택하도록 하는 캡스톤 프로젝트입니다. 이 저장소는 프로젝트 전체(예측 모델 학습, RL 비교 실험, 논문/보고서 자료)를 담고 있으며, BL-NRMSE 실험만 따로 정리한 축소 버전은 [`bl_pipeline`](https://github.com/jdw0507/bl_pipeline) 저장소를 참고하세요.

## 주요 내용

- **예측 파이프라인** (`main_forecast.py`): Linear, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, PatchTST 등 ML/DL 모델로 자산별 초과수익률 예측 (nested walk-forward 검증)
- **RL 비교 실험** (`main_PPO_DQN_compare_all*.py`): DQN/PPO 기반 4가지 모델-선택 에이전트를 Black-Litterman 포트폴리오와 결합해 walk-forward로 비교
  - 자산별 선택 vs 횡단면 선택
  - 예측 정확도 보상 vs EMA-Sharpe(BL 수익) 보상
- **백테스트 재개 스크립트** (`resume_backtest_v2.py`)
- `src/`: 데이터 로딩, 예측 모델, 강화학습 환경/에이전트 등 핵심 모듈
- `docs/`: 프로젝트 문서
- 논문/보고서 초안 (`.docx`)

## 실행

```bash
python main_forecast.py               # 예측 모델 학습 + 백테스트
python main_PPO_DQN_compare_all.py    # RL 에이전트 vs BL 벤치마크 비교
```

## 기술 스택

Python, PyTorch, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, pandas, numpy

## 참고

대용량 산출물(outputs/), `catboost_info/`, Claude Code 로컬 설정(`.claude/`)은 `.gitignore`로 제외되어 있습니다.
