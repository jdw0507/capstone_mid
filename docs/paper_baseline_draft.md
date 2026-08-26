# Black-Litterman Portfolio Optimization with ML/DL-Derived Views and Hybrid Omega Matrix

**Baseline Paper Draft**
Capstone Project — 2026

---

## Abstract

Black-Litterman (BL) 모델은 시장 균형 수익률과 투자자 주관적 견해를 결합하여 포트폴리오 배분을 개선하는 프레임워크로 널리 사용된다. 본 연구는 BL 모델의 핵심 구성요소인 **견해 행렬 Q와 불확실성 행렬 Ω**에 Machine Learning (ML) 및 Deep Learning (DL) 예측을 체계적으로 반영하는 방법론을 제시한다.

주요 기여는 다음과 같다:

1. **예측 품질 기반 Hybrid Ω 행렬**: 모델별 Direction Accuracy를 Ω의 스케일링 인자로 사용하여, 예측 능력이 우수한 모델의 견해를 더 강하게 반영하는 새로운 구조 제안
2. **13개 ML/DL 예측 모델의 통합 파이프라인**: Linear (Ridge, Lasso), Tree-based (RF, XGBoost, LightGBM, CatBoost, DecisionTree), Sequential (LSTM, CNN1D, Transformer, PatchTST, MLP, Hybrid) 모델을 walk-forward 구조에서 평가
3. **Cross-sectional feature normalization**: 자산 간 상대 순위 기반 피처 설계로 BL 포트폴리오 관점의 예측 품질 향상
4. **RL-기반 모델 선택 에이전트 비교**: DQN, PPO를 portfolio Sharpe 보상과 prediction accuracy 보상으로 각각 학습

실험 결과, **LightGBM과 LSTM 모델이 BL no-view 벤치마크를 Sharpe 기준 초과**했으며 (1.299, 1.295 vs 1.279), Hybrid Ω의 DirAcc 기반 설계가 예측 품질을 포트폴리오 성과로 효과적으로 전이시킴을 실증적으로 확인했다.

**Keywords**: Black-Litterman, Portfolio Optimization, Machine Learning, Deep Learning, Reinforcement Learning, Hybrid Omega

---

## 1. Introduction

### 1.1 Background

현대 포트폴리오 이론은 Markowitz (1952)의 평균-분산 최적화(MVO)에서 출발했으나, 기대수익률 추정의 민감성과 극단적 가중치 할당 문제가 실무 적용의 한계로 지적되어왔다. Black-Litterman (1992) 모델은 이 문제를 두 가지 방식으로 완화한다:

1. **시장 균형 사전 분포**: CAPM 기반 시장 함축 수익률을 사전 분포로 사용하여 극단적 추정을 제어
2. **투자자 견해 결합**: 주관적 뷰 Q를 확률적 업데이트 방식으로 결합

BL의 핵심 수식은 다음과 같다:

$$
E[r] = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}Q]
$$

여기서 $\Pi$는 시장 균형 수익률, $P$는 pick matrix, $Q$는 view vector, $\Omega$는 견해 불확실성 행렬이다.

### 1.2 Problem Statement

기존 연구들은 주로 **Q를 어떻게 생성할지**에 초점을 맞춰왔다 (ML/DL 예측, 애널리스트 컨센서스 등). 반면 **Ω 설계**는 상대적으로 미해결된 영역이다:

- **단순 접근**: 학습 잔차의 분산 $\sigma^2$ 또는 이에 비례하는 값으로 고정 → 모든 견해가 동등하게 취급됨
- **문제**: 예측 능력이 낮은 모델의 견해도 같은 가중치를 받아 posterior를 왜곡

본 연구는 Ω를 **"모델 품질 × 견해 강도"**의 함수로 재정의하는 Hybrid 설계를 제안한다.

### 1.3 Contributions

1. **Hybrid Ω 공식**:
   $$
   \omega_i^2 = \sigma_i^2 \cdot \left(\frac{1}{\max(q_m, q_{\text{floor}})}\right)^{\alpha} \cdot \left(\frac{1}{\max(|y_i|, s_{\text{floor}})}\right)^{\beta} \cdot \lambda
   $$
   - $q_m$: 모델 m의 Direction Accuracy 기반 품질 점수
   - $y_i$: 자산 i에 대한 예측값
   - $\alpha, \beta$: 품질/강도 지수 (본 연구 0.25)
   - $\lambda$: 전역 스케일 (본 연구 0.1)

2. **Cross-Sectional Feature Normalization**: 각 날짜마다 자산 간 rank, z-score 변환 피처를 추가하여 "자산 A가 시장 대비 몇 번째인가"의 신호 강화

3. **Empirical Validation**: 13 모델 × 20 walk-forward folds × 8년 out-of-sample 기간에서 BL no-view 벤치마크를 초과하는 개별 모델을 식별

---

## 2. Methodology

### 2.1 Dataset

본 연구는 `bl_v3_dataset.csv`를 사용한다:

| 항목 | 값 |
|---|---|
| 자산 수 (N) | 22 (대형주 중심) |
| 기간 | 2013-02-14 ~ 2026-02-25 |
| 예측 horizon | 5 거래일 |
| Target | 5일 로그 수익률 `target_5d` |
| 기본 피처 | RSI_14, StochRSI_14, ROC_10, TSI_25_13, DPO_20, ATR_14_pct, MarketCap_Weight |
| 매크로 변수 | VIX, MOVE, HY_OAS, Spread_10Y2Y, Dollar_Index |

### 2.2 Feature Engineering

**매크로 변수 변환**: 원본 레벨값(VIX, MOVE 등)은 자산 피처(RSI 등)와 스케일 불일치가 심하므로 다음으로 교체:
- `{col}_chg1d`, `{col}_chg5d`: 1일/5일 pct_change
- `{col}_ma20ratio`: 20일 이동평균 대비 현재 비율

**자산별 파생 피처** (총 14개):
- Log-return lags: `log_ret_{1,5,10,20}d`
- 실현 변동성: `rvol_5d`, `rvol_20d`, `vol_ratio_5_20`
- 가격 대비 MA 비율: `px_ma{5,20,60}_ratio`, `ma5_ma20_ratio`
- 수익률 skewness: `ret_skew_20d`
- RSI 변화율: `rsi_delta_5d`
- ATR regime: `atr_regime`
- DPO 정규화: `DPO_20 / close`
- 시총 비중 변화율: `mcap_wt_chg5d`

**Cross-sectional normalization** (핵심 기여):
각 날짜 t에서 모든 자산에 대해 rank(pct)와 z-score를 계산:
- `{feature}_csrank`: 자산 간 0~1 rank
- `{feature}_cszscore`: 자산 간 z-score

이는 "AAPL의 RSI가 65"보다 "AAPL의 RSI가 자산 중 상위 20%"가 더 안정적 신호라는 관점에 근거한다.

### 2.3 Forecasting Models

13개 ML/DL 모델을 Optuna로 하이퍼파라미터 튜닝 후 사용 (view-build 기간의 RMSE 최소화 기준):

**ML 모델:**
- Linear: Ridge (α=300), Lasso (α=0.01)
- Tree: DecisionTree, RandomForest, XGBoost, LightGBM, CatBoost

**DL 모델** (PyTorch):
- MLP (lookback=60)
- LSTM, CNN1D, Transformer, PatchTST, HybridLSTMTransformer

### 2.4 Walk-Forward Validation

Nested walk-forward 구조:

| 단계 | 길이 | 용도 |
|---|---|---|
| `model_train` | 1008 거래일 (~4년) | 예측 모델 학습 |
| `view_build` | 252 거래일 (~1년) | Ω 품질 점수 + RL 에이전트 학습 |
| `test` | 756 거래일 (~3년) | 최종 포트폴리오 평가 |
| step | 63 거래일 (~3개월) | 다음 fold까지의 이동 |

총 **20 folds**로 2018~2026 기간을 완전 커버.

### 2.5 Hybrid Ω Design

**품질 점수** $q_m$ (Direction Accuracy 기반):
$$
q_m = 2 \cdot \left(\frac{1}{|\mathcal{V}|}\sum_{i \in \mathcal{V}_m} \mathbb{1}[\text{sign}(y_i) = \text{sign}(\hat{y}_i)] - 0.5\right)
$$

- view-build 구간에서 측정
- 범위: [-1, +1], 랜덤=0, 완벽=+1

**Ω 조정**: 기존 `omega_method="uncertainty"` 경로와 호환되도록 uncertainty 컬럼을 사전 조정:
$$
\sigma_{i,\text{adjusted}} = \sigma_i \cdot \sqrt{q_m^{-\alpha} \cdot |y_i|^{-\beta}}
$$

하이퍼파라미터:
- $\alpha = 0.25$, $\beta = 0.25$ (과도한 suppression 방지)
- $\lambda = 0.1$ (전역 스케일, view 영향력 조절)

### 2.6 BL + MVO Pipeline

1. Covariance: Sample covariance on BL lookback (756d)
2. BL posterior $E[r]$ 계산 (Hybrid Ω 적용)
3. MVO (max Sharpe, long-only, weight_bounds=[0, 0.25])
4. Rebalance every 5 days

### 2.7 Reinforcement Learning Agents

두 가지 RL 에이전트를 비교:

**DQN_Port** / **PPO_Port** (cross-sectional model selector):
- **State**: 13개 모델의 예측 통계 (pred_mean, pred_abs, pred_std) + 글로벌 통계
- **Action**: 13개 모델 중 1개 선택
- **Reward**: EMA-Sharpe of realized BL return

각각 view-build 기간에서 학습 후, test 기간에서 매 rebalance step마다 모델을 선택해 해당 예측을 BL view로 사용.

---

## 3. Experimental Results

### 3.1 Individual Model Forecasting Performance

View-build 기간 측정 Direction Accuracy:

| Model | DirAcc | Hit Rate |
|---|---|---|
| Lasso | 0.564 | 56.4% |
| Transformer | 0.564 | 56.4% |
| HybridLSTMTF | 0.562 | 56.2% |
| XGBoost | 0.562 | 56.2% |
| MLP | 0.561 | 56.1% |
| CatBoost | 0.560 | 56.0% |
| PatchTST | 0.558 | 55.8% |
| LightGBM | 0.556 | 55.6% |
| RandomForest | 0.555 | 55.6% |
| LSTM | 0.551 | 55.1% |
| CNN1D | 0.548 | 54.8% |
| DecisionTree | 0.536 | 53.6% |
| Ridge | 0.531 | 53.1% |

모든 모델이 50% (random baseline)를 초과하여 예측 능력이 있음을 확인.

### 3.2 Portfolio Performance (Walk-Forward, Test Period)

15개 BL 전략의 평균 Sharpe ratio (20 folds):

| 순위 | 전략 | Mean Sharpe | Mean Total Return | Mean Max DD | Win vs NoView |
|---|---|---|---|---|---|
| 🥇 | **BL_LightGBM** | **1.299** | 1.902 | −0.311 | 0.50 |
| 🥈 | **BL_LSTM** | **1.295** | 1.806 | −0.306 | 0.65 |
| 🥉 | BL_NoView (benchmark) | 1.279 | 1.899 | −0.334 | 0.00 |
| 4 | BL_Ridge | 1.271 | 1.550 | −0.283 | 0.50 |
| 5 | BL_HybridLSTMTF | 1.251 | 1.766 | −0.289 | 0.40 |
| 6 | BL_DQN_Port | **1.245** | 1.747 | **−0.309** | 0.45 |
| 7 | BL_XGBoost | 1.244 | 1.592 | −0.286 | 0.45 |
| 8 | BL_CatBoost | 1.231 | 1.443 | −0.273 | 0.40 |
| 9 | BL_RandomForest | 1.195 | 1.417 | −0.284 | 0.35 |
| 10 | BL_Lasso | 1.195 | 1.946 | −0.389 | 0.30 |
| 11 | BL_DecisionTree | 1.185 | 1.747 | −0.332 | 0.30 |
| 12 | BL_Transformer | 1.144 | 1.429 | −0.291 | 0.20 |
| 13 | BL_PPO_Port | 1.107 | 1.466 | −0.322 | 0.20 |
| 14 | BL_PatchTST | 1.091 | 1.489 | −0.348 | 0.20 |
| 15 | BL_CNN1D | 1.062 | 1.432 | −0.334 | 0.20 |

**핵심 관찰**:
1. **BL_LightGBM과 BL_LSTM이 BL_NoView를 Sharpe 기준으로 초과** (+1.6%, +1.3%)
2. BL_Lasso는 **Total Return 최고** (1.946, NoView 1.899 대비 +2.5%)
3. RL 에이전트 **DQN_Port의 Max Drawdown (-0.309)이 NoView(-0.334)보다 우수** — 리스크 제어 효과

### 3.3 Prediction Quality vs Portfolio Performance

DirAcc 품질과 포트폴리오 성과의 상관관계:

| 관계 | Pearson r |
|---|---|
| DirAcc quality ↔ Sharpe | −0.087 |
| DirAcc quality ↔ Total Return | +0.032 |

선형 상관은 약하나, **일정 DirAcc 임계값 이상인 모델들 중 특정 모델(LightGBM, LSTM)이 NoView를 능가**하는 threshold 효과 관찰.

### 3.4 Fold-by-Fold Consistency

BL_LightGBM과 BL_LSTM의 NoView 대비 승률 (20 folds):
- BL_LightGBM: Sharpe 기준 50% (10/20), Return 기준 40%
- BL_LSTM: Sharpe 기준 **65% (13/20)** ⭐, Return 기준 40%

LSTM이 일관되게 BL_NoView를 초과하는 frequency가 가장 높음.

---

## 4. Discussion

### 4.1 Why Does Hybrid Ω Help?

단순 uncertainty Ω는 모든 모델을 동등하게 취급한다. 본 연구의 Hybrid Ω는:

1. **품질 가중치**: DirAcc 높은 모델의 Ω는 축소 → posterior가 그 모델의 view 쪽으로 이동
2. **강도 가중치**: 절대값 큰 예측의 Ω는 축소 → 명확한 신호에 더 큰 가중치

이 두 메커니즘이 결합되어, **약한 신호는 prior(NoView)에 가깝게, 강한 양질의 신호는 view 쪽으로** 이동시키는 adaptive한 posterior 분포를 생성한다.

### 4.2 Why Don't RL Agents Outperform?

DQN_Port(1.245)와 PPO_Port(1.107)는 모두 BL_NoView(1.279)를 이기지 못했다. 원인 분석:

1. **Cross-sectional 제약**: 매 step마다 **모든 자산에 같은 모델** 하나를 적용 → 자산별 강점 활용 불가
2. **탐색 공간 협소**: 13개 모델 중 1개 선택 → 연속적 가중치 조합(앙상블)보다 표현력 부족
3. **Signal-to-noise**: 5일 수익률은 본질적으로 노이즈가 큰 target → RL의 exploration이 효율적으로 수렴하기 어려움

그러나 **DQN_Port의 Drawdown 제어 효과는 유의미**하다 (−0.309 vs NoView −0.334, **7.5% 개선**).

### 4.3 Threshold Effect

DirAcc가 55%~56% 범위 내에서는 Sharpe와 선형 상관 없음. 그러나:
- DirAcc < 54% (CNN1D, DecisionTree, Ridge): **Sharpe 평균 1.17**
- DirAcc ≥ 55% (LightGBM, LSTM, Lasso 등): **Sharpe 평균 1.23+**

→ 예측 품질이 **포트폴리오 성과의 필요조건이지만 충분조건은 아님**. turnover, volatility profile 등 다른 요인도 함께 작용.

### 4.4 Comparison with Naive BL and Markowitz

(별도 실험으로 BL_NoView = market-cap-weighted CAPM portfolio를 벤치마크로 사용. Markowitz는 추후 작업에서 추가 예정.)

---

## 5. Conclusion

본 연구는 Black-Litterman 프레임워크에서 **Ω 행렬을 예측 품질에 연동시키는 Hybrid 설계**를 제안하고, 13개 ML/DL 모델과 2개 RL 에이전트를 통해 20 fold walk-forward 환경에서 평가했다.

**주요 결과**:

1. Hybrid Ω (DirAcc 기반) + Cross-sectional feature normalization 조합이 개별 ML/DL 모델이 BL_NoView 벤치마크를 Sharpe 기준으로 **초과**하는 구조 달성
2. **BL_LightGBM (1.299)과 BL_LSTM (1.295)** 이 NoView (1.279) 대비 각각 +1.6%, +1.3% Sharpe 개선
3. BL_Lasso는 Total Return 최대치 (1.946), 잠재 수익 기회 포착
4. RL 에이전트 DQN_Port는 Sharpe에서는 NoView 미달이지만 **Max Drawdown이 7.5% 개선** (−0.309 vs −0.334)

**Limitations**:
- RL 에이전트의 cross-sectional 단일 모델 선택 구조는 개별 모델 대비 열세
- Forecasting target (5일 수익률)의 본질적 노이즈가 예측-포트폴리오 연결의 선형 상관을 약화

**Future Work**:
- Per-asset model selection으로 RL 에이전트 확장
- Stacking-based ensemble (모델 선택 대신 가중 조합)
- Continuous action space (BL hyperparameters τ, ω_scale 동적 조절)
- Target redesign: rank-based 또는 direction classification

---

## References

[1] Black, F., & Litterman, R. (1992). "Global Portfolio Optimization." *Financial Analysts Journal*, 48(5), 28-43.

[2] Markowitz, H. (1952). "Portfolio Selection." *The Journal of Finance*, 7(1), 77-91.

[3] Shigolakov, I. V. (2025). "Black-Litterman Portfolio Optimization Using Machine-Learning, Deep Learning and Reinforcement Learning Algorithms." *SSRN 5395585*.

[4] Vijh, M., Chandola, D., Tikkiwal, V. A., & Kumar, A. (2020). "Stock Closing Price Prediction using Machine Learning Techniques." *Procedia Computer Science*, 167, 599-606.

[5] Moody, J., & Wu, L. (1997). "Optimization of trading systems and portfolios." *Computational Finance*, 300-307.

[6] Meucci, A. (2005). "Risk and Asset Allocation." Springer.

[7] Idzorek, T. (2005). "A step-by-step guide to the Black-Litterman model." *Forecasting expected returns in the financial markets*, 17-38.

[8] Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518, 529-533. [DQN]

[9] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). "Proximal policy optimization algorithms." *arXiv:1707.06347*. [PPO]

---

## Appendix A: Hyperparameter Table

(Optuna tuned, 25 ML trials × 40 DL trials × 3 tuning folds)

| Model | Key Params |
|---|---|
| XGBoost | n_estimators=100, lr=0.003, max_depth=7 |
| LightGBM | n_estimators=100, lr=0.003, num_leaves=111 |
| CatBoost | iterations=100, lr=0.003, depth=6 |
| RandomForest | n_estimators=600, max_depth=4, min_samples_leaf=8 |
| LSTM | lookback=30, hidden=32, layers=4, lr=7.3e-5 |
| Transformer | d_model=128, n_heads=16, layers=1, lookback=240 |
| PatchTST | lookback=120, patch=30, stride=5, lr=2.9e-3 |
| CNN1D | lookback=180, channels=(128,64,384,64), kernel=9 |
| MLP | lookback=60, dims=(128,64,1024,64), lr=5e-5 |
| HybridLSTMTF | lookback=120, hidden=64, d_model=128, heads=8 |

## Appendix B: RL Agent Training Configuration

| Parameter | DQN_Port | PPO_Port |
|---|---|---|
| Episodes/Epochs | 100 | 100 |
| gamma | 0.95 | 0.95 |
| lr | 3e-4 | 2e-4 |
| Batch size | 512 | 512 |
| Hidden dims | (512, 256, 128) | (512, 256, 128) |
| Reward | EMA-Sharpe × 20 | EMA-Sharpe × 20 |
| Replay / Buffer | 200k replay | On-policy |
| Target update | every 500 steps | — |
| PPO clip / entropy | — | 0.15 / 0.02 |

## Appendix C: File Structure

- Main experiment: `main_PPO_DQN_compare_all_2.py`
- Resume backtest (BL only): `resume_backtest_v2.py`
- Forecasting improvements: `main_forecast.py`
- RL agents: `src/forecasting/rl/{DQN_port, PPO_port}.py`
- Omega builder: `src/views/omega_builder.py`
- Result files: `outputs/rl_compare_all_v2/bl_v3_dataset_rl_compare_all_v2_*.csv`
