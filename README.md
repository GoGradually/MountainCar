# MountainCar DQN

`gymnasium`의 `MountainCar-v0` 환경에서 DQN을 학습하고, 고정 timestep 간격의 evaluation 보상 곡선을 PNG로 저장하는 프로젝트입니다.

## 1. 프로젝트 개요

- 학습 알고리즘: DQN (Replay Buffer + Target Network + Epsilon Decay)
- 환경: `MountainCar-v0`
- 출력:
  - 콘솔 로그: 학습 소요 시간, 마지막 smoothed evaluation 보상
  - CSV 로그: trial별 evaluation 값 (`artifacts/*_eval.csv`)
  - 이미지: evaluation 보상 곡선 PNG (`artifacts/*.png`)

## 2. 빠른 시작

### 2.1 요구 사항

- Python 3.10 이상 권장 (검증 환경: Python 3.12.3)

### 2.2 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2.3 의존성 설치

런타임 의존성:

```bash
pip install gymnasium numpy torch matplotlib
```

개발/테스트 의존성:

```bash
pip install -r requirements-dev.txt
```

## 3. 실행 방법

### 3.1 Quick 프로필 (빠른 확인)

```bash
python3 main.py --profile quick
```

- 기본 저장 경로: `artifacts/quick_reward.png`
- evaluation 로그 경로: `artifacts/quick_reward_eval.csv`

### 3.2 Full 프로필 (긴 학습)

```bash
python3 main.py --profile full
```

- 기본 저장 경로: `artifacts/full_reward.png`
- evaluation 로그 경로: `artifacts/full_reward_eval.csv`

### 3.3 재현 가능한 실행 (seed 고정)

```bash
python3 main.py --profile quick --seed 123
```

### 3.4 출력 파일 경로 직접 지정

```bash
python3 main.py --profile quick --plot-path artifacts/custom_reward.png
```

- evaluation 로그 경로: `artifacts/custom_reward_eval.csv`

### 3.5 실행 결과 예시

실행 시 다음과 같은 형태의 로그가 출력됩니다.

```text
Using device: cpu
Elapsed: 0.8014s
Last smoothed eval reward: -200.0000
Eval log saved to: artifacts/quick_reward_eval.csv
Plot saved to: artifacts/quick_reward.png
```

## 4. CLI 옵션

| 옵션 | 값 | 설명 | 기본 동작 |
|---|---|---|---|
| `--profile` | `quick` \| `full` | 실행 프로필 선택 | `quick` |
| `--timesteps` | 양의 정수 | 총 환경 step 수 직접 지정 | 프로필 기본값 사용 |
| `--trials` | 양의 정수 | trial 반복 횟수 | 프로필 기본값 사용 |
| `--plot-path` | 문자열 경로 | 결과 플롯 저장 경로 | 프로필별 기본 경로 |
| `--seed` | 정수 | 랜덤 시드 고정 | 미지정 (`None`) |

참고:

- `--timesteps`, `--trials`은 지정하지 않으면 선택한 `profile`의 값이 적용됩니다.
- 음수/0은 허용되지 않습니다 (`argparse` 검증).
- evaluation CSV는 plot 경로와 같은 디렉터리에 `{plot 파일명}_eval.csv` 규칙으로 자동 저장됩니다.

## 5. 테스트 실행

### 5.1 기본 테스트 (integration 제외)

```bash
python3 -m pytest -q
```

- `pytest.ini`에서 기본적으로 `-m "not integration"`이 적용됩니다.

### 5.2 Integration 테스트만 실행

```bash
python3 -m pytest -q -m integration
```

### 5.3 느린 수렴 검증 테스트 실행

```bash
python3 -m pytest -q -m "integration and slow"
```

- MountainCar 수렴 여부를 확인하는 장시간 테스트입니다.
- 기본 `pytest` 경로에서는 제외됩니다.

권장 실행 방식:

- `pytest` 단독 호출보다 `python3 -m pytest` 방식이 import 경로 이슈를 줄입니다.

## 6. 프로젝트 구조

```text
.
├── agent.py          # AgentConfig, QNet, DQNAgent, ReplayBuffer
├── train.py          # TrainingConfig/TrainingResult, 학습 오케스트레이션
├── main.py           # CLI 진입점, 학습 실행, 플롯 저장
├── viz.py            # 보상 곡선 PNG 저장
├── tests
│   ├── unit          # 유닛 테스트
│   └── integration   # 학습 스모크 테스트
├── pytest.ini        # pytest 기본 마커 설정
└── requirements-dev.txt
```

## 7. 기본 학습 설정

`train.py`의 `TrainingConfig` 기본값:

| 항목 | 기본값                |
|---|--------------------|
| `total_timesteps` | `120000`           |
| `trials` | `3`                |
| `env_id` | `"MountainCar-v0"` |
| `render_mode` | `None`             |
| `agent_config` | `AgentConfig()`    |
| `device` | `None` (자동 선택)     |
| `seed` | `None`             |
| `eval_freq` | `2000`             |
| `n_eval_episodes` | `20`               |
| `eval_window` | `5`                |
| `eval_seed` | `1000000`          |
| `eval_log_path` | `None`             |
| `log_device` | `True`             |
| `log_progress` | `True`             |

`agent.py`의 `AgentConfig` 기본값:

| 항목 | 기본값 |
|---|---|
| `gamma` | `0.98` |
| `lr` | `0.004` |
| `buffer_size` | `10000` |
| `batch_size` | `128` |
| `action_space` | `3` |
| `hidden_dim` | `256` |
| `train_start` | `1000` |
| `train_freq` | `16` |
| `gradient_steps` | `8` |
| `max_grad_norm` | `10.0` |
| `eps_start` | `1.0` |
| `eps_final` | `0.07` |
| `exploration_fraction` | `0.2` |
| `target_sync_every` | `600` |

프로필 차이:

- `quick`: `total_timesteps=20000`, `trials=1`, `render_mode=None`, `log_progress=False`
- `full`: `TrainingConfig` 기본값 사용

알고리즘 구현 메모:

- 행동 선택은 online Q-network가 담당합니다.
- 다음 상태의 bootstrap 값은 target Q-network의 최대 Q값으로 계산합니다.
- 즉, target network를 사용하는 vanilla DQN 형식입니다.
- 학습 종료는 episode 수가 아니라 `total_timesteps` 기준입니다.
- step budget 중간에 끝난 마지막 미완료 episode는 reward history에 포함하지 않습니다.
- 메인 plot은 `eval_freq` step마다 greedy policy로 측정한 raw evaluation reward와 5-point moving average를 `timesteps` 축으로 함께 그립니다.
- evaluation episode의 초기 상태는 랜덤하게 뽑되, `eval_seed` 기반 RNG로 재현 가능하게 유지합니다.
- trial별 evaluation raw reward는 evaluation 시점마다 CSV에 `trial,timestep,eval_reward` 형식으로 즉시 저장됩니다.

## 8. 트러블슈팅

### 8.1 CUDA 관련 경고가 보일 때

- 일부 환경에서 CUDA 초기화 경고가 출력될 수 있습니다.
- GPU를 사용할 수 없으면 자동으로 CPU로 학습이 진행됩니다.

### 8.2 플롯 파일이 생성되지 않을 때

- `--plot-path`를 명시해 경로를 직접 지정해 보세요.
- 기본 경로는 `artifacts/` 하위입니다. 폴더가 없으면 코드가 자동 생성합니다.
