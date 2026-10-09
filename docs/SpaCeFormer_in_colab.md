# SpaCeFormer Colab T4 실행 환경 구축 및 오류 분석 최종 정리
SpaceWatch3D 종합설계프로젝트1 | 2026.10.09 기준
이번 작업의 목적은 Google Colab의 Tesla T4 GPU에서 SpaCeFormer를 실행하고, 향후 SpaceWatch3D 프로젝트의 3D Instance Segmentation 모델 성능 평가에 활용할 수 있는 환경을 구축하는 것이었어.
지금까지 발생한 오류, 원인 분석, 실제 적용한 해결 방법, 추가로 검토한 대체 방법, 그리고 현재 도달한 결과까지 정리할게.

## 1. 현재 최종 결과부터
SpaCeFormer 실행 상태
추론 검증 미완료
모델 생성 성공
체크포인트 로딩 성공
SDPA 대체 연산 성공
실제 3D Forward 실패

모델 로딩 후 PyTorch GPU 할당 메모리
# 0.335 GiB
추론 중 최대 메모리가 아니라 모델 로딩 시점의 메모리 사용량
현재까지 확실하게 검증된 사실은 다음과 같아.
- Tesla T4에서 PyTorch와 일반적인 CUDA 연산이 정상적으로 작동함.
- WarpConvNet 1.8.2와 SpaCeFormer의 import에 성공함.
- 공식 사전 학습 체크포인트를 정상 다운로드함.
- FlashAttention 의존성을 PyTorch SDPA로 임시 대체해 모델 생성에 성공함.
- 체크포인트를 로딩했을 때 `Missing keys = 0`, `Unexpected keys = 0`을 확인함.
- 실제 2,048개 포인트를 입력하자 보셀 다운샘플링 과정의 `torch.unique()`에서 CUDA 오류가 발생함.

따라서 SpaCeFormer 모델과 가중치 확보에는 성공했지만, T4에서 실제 3D Instance Segmentation이 가능하다는 검증은 아직 끝나지 않았어.
최종 방향은 지금까지 사용한 임시 패치를 계속 늘리는 대신, T4 아키텍처에 맞게 WarpConvNet을 소스 빌드하고 필요한 호환성 문제를 체계적으로 해결하는 것으로 정했어. 다만 이 방법이 현재 CUDA 오류를 해결한다고 아직 확정할 수는 없어.

## 2. 현재 구축된 Colab 환경

### 하드웨어 및 소프트웨어

| 구분              | 사용 환경                          |
| --------------- | ------------------------------ |
| 플랫폼             | Google Colab                   |
| GPU             | NVIDIA Tesla T4                |
| GPU 아키텍처        | Turing, Compute Capability 7.5 |
| 사용 가능 GPU 메모리   | 14.56 GiB                      |
| 가상환경 Python     | 3.12.14                        |
| PyTorch         | 2.10.0+cu128                   |
| torchvision     | 0.25.0+cu128                   |
| PyTorch CUDA    | 12.8                           |
| WarpConvNet     | 1.8.2+torch2.10cu128           |
| WarpConvNet 커밋  | `e3ab140bea35`                 |
| einops          | 0.8.2                          |
| huggingface_hub | 추가 설치 완료                       |

주요 경로

```
# Colab 가상환경
/content/spaceformer-env

# 가상환경 Python 실행 파일
/content/spaceformer-env/bin/python

# Google Drive 설치 스크립트
/content/drive/MyDrive/spacewatch3d_colab/
└── setup_spaceformer_colab.py

# 체크포인트(Hugging Face 캐시)
/root/.cache/huggingface/hub/
└── models--chrischoy--SpaCeFormer/
    └── snapshots/a8e81555.../
        └── spaceformer_512_siglip2_ssccc.ckpt
```

현재 가상환경과 체크포인트 캐시는 Colab 임시 런타임에 있어. 따라서 런타임 종료 후 재설치하거나 다시 다운로드해야 할 수 있어. Google Drive에 남아 있는 것은 설치 스크립트로 확인했으며, T4 전용 WarpConvNet Wheel은 아직 생성하지 않았어.

## 3. 발생한 문제들과 해결 과정

### 문제 1. Colab 런타임 초기화로 가상환경 삭제

증상

Google Drive에서 설치 스크립트는 확인됐지만, 기존 가상환경은 존재하지 않았어.

```
설치 스크립트: True
기존 가상환경: False
```

원인

기존 `/content/spaceformer-env`는 Colab의 임시 파일시스템에 생성했기 때문에 런타임 초기화 후 유지되지 않았어.

적용한 해결 방법

Google Drive의 기존 설치 스크립트를 다시 실행했어.

```
!python -u /content/drive/MyDrive/spacewatch3d_colab/setup_spaceformer_colab.py
```

결과: 해결 완료

가상환경을 재생성하고 PyTorch, CUDA, WarpConvNet, SpaCeFormer import까지 확인했어.

### 문제 2. Hugging Face 패키지 누락

증상

```
ModuleNotFoundError:
No module named 'huggingface_hub'
```

원인

SpaCeFormer 체크포인트를 다운로드하기 위한 `huggingface_hub`가 가상환경에 설치되지 않았어.

적용한 해결 방법

```
source /content/spaceformer-env/bin/activate
python -m pip install huggingface_hub
```

설치 이후 공식 체크포인트 다운로드를 실행했어.

```
from huggingface_hub import hf_hub_downloadckpt = hf_hub_download(    repo_id="chrischoy/SpaCeFormer",    filename="spaceformer_512_siglip2_ssccc.ckpt")
```

결과: 해결 완료

```
Exists: True
Size (MB): 327.49
DOWNLOAD SUCCESS
```

처음에는 비공개 저장소 또는 접근 권한 문제 가능성도 검토했지만, 이번에는 정상적으로 다운로드됐어. 따라서 현재 공식 저장소에 대한 접근은 가능하다는 것을 확인했지.

### 문제 3. Terminal과 Colab 코드 셀의 Python 환경 불일치

증상

터미널에서는 WarpConvNet import가 성공했지만, Colab 노트북 코드 셀에서 실행하면 다음 오류가 발생했어.

```
ModuleNotFoundError:
No module named 'warpconvnet'
```

원인

| 구분               | Python  |
| ---------------- | ------- |
| Colab 기본 노트북 커널  | 3.13    |
| SpaCeFormer 가상환경 | 3.12.14 |

터미널에서 가상환경을 활성화해도 Colab 기본 노트북 커널은 별도 Python 환경을 사용했기 때문이야.

적용한 해결 방법

코드 셀 첫 줄에 아래 명령을 추가했어.

```
%%script /content/spaceformer-env/bin/python -u
```

또한 CUDA 오류 위치를 추적할 때는 다음과 같이 변경했지.

```
%%script env CUDA_LAUNCH_BLOCKING=1 /content/spaceformer-env/bin/python -u
```

결과: 해결 완료

노트북에서도 Python 3.12 가상환경으로 코드를 실행할 수 있게 됐어.

단, `%%script`는 셀마다 별도 Python 프로세스를 실행하므로 모델 생성, 임시 SDPA 어댑터 적용, 체크포인트 로딩, Forward 테스트는 같은 셀 안에서 실행하도록 구성했어.

### 문제 4. FlashAttention 미설치 및 T4 호환성 문제

증상

```
AssertionError:
Make sure flash_attn is installed.
```

발생 위치:

```
warpconvnet/nn/modules/attention.py
```

원인

SpaCeFormer의 Attention 구현이 `flash_attn` 라이브러리를 직접 요구했어.

특히 기존 코드에는 다음 검사가 존재했지.

```
assert flash_attn is not None
```

일반 FlashAttention-2의 공식 지원 아키텍처는 Ampere, Ada, Hopper이며, T4는 Turing 아키텍처라 지원 범위에 포함되지 않아. Turing을 위한 별도의 제한적인 구현은 존재해.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



검토했던 대체 방법

| 대체 방법                  | 장점                            | 한계                            |
| ---------------------- | ----------------------------- | ----------------------------- |
| 일반 FlashAttention-2 설치 | 기존 구현 유지 가능                   | T4 아키텍처 미지원                   |
| Turing용 FlashAttention | T4 활용 가능성                     | WarpConvNet API 호환성 미검증       |
| WarpConvNet 자체 커널 사용   | 기존 CUDA 연산 활용                 | FlashAttention 호출 자체를 대체하지 못함 |
| PyTorch SDPA 적용        | 기존 PyTorch만으로 Attention 연산 가능 | 별도의 대체 코드 필요                  |

실제로 적용한 방법: PyTorch SDPA 대체

```
F.scaled_dot_product_attention(    q,    k,    v,    dropout_p=dropout_p,    scale=softmax_scale,    is_causal=False)
```

FlashAttention의 가변 길이 입력(`varlen`)을 처리하기 위해 `cu_seqlens`를 기준으로 각 패치의 Q, K, V를 나누어 처리했어.

그리고 기존 WarpConvNet 소스 파일을 직접 수정하는 대신 Python 실행 중에만 다음 어댑터를 연결했지.

```
attention.flash_attn = adapterflash_utils.flash_attn = adapter
```

결과: 모델 생성 단계까지 해결

```
SDPA varlen: PASS
Adapter installed: PASS
Model build: PASS
```

다만 이는 FlashAttention과의 수치적 동등성을 완전히 검증한 공식 대체 구현은 아니야. 패치별 SDPA 구현이 실제 추론 과정에서도 정상적으로 작동하는지 확인해야 하고, 속도와 GPU 메모리 사용량도 기존 구현과 다를 수 있어.

### 문제 5. SpaCeFormer 체크포인트 로딩 검증

FlashAttention 문제를 우회한 뒤, 실제 모델 가중치를 로딩했어.

실행 결과

```
=== 3. SpaCeFormer 모델 생성 ===
Model build: PASS

=== 4. 체크포인트 로딩 ===
Missing keys: 0
Unexpected keys: 0

=== 5. GPU 메모리 ===
Allocated (GiB): 0.335

MODEL BUILD + CHECKPOINT LOAD PASS
```

결과: 성공

모델 구조를 생성했고 체크포인트의 파라미터 키를 누락이나 추가 키 없이 로딩했어.

이 단계까지는 T4에서 모델을 준비하는 데 문제가 없었어.

다만 이 결과는 실제 입력 데이터에 대한 Forward 연산이 성공했다는 뜻은 아니야.

### 문제 6. 실제 3D Forward에서 CUDA 커널 오류

테스트 조건

- 입력: 무작위 생성한 2,048개 3D 포인트
- 좌표: XYZ, float32
- 특징: RGB 형태의 3차원 특징
- GPU: Tesla T4
- 실행: `torch.inference_mode()`

발생 오류

```
torch.AcceleratorError:
CUDA error: no kernel image is available
for execution on the device
```

실행 흐름을 추적하면 다음과 같아.

```
SpaCeFormer.forward()
  │
  └── backbone(pc)
       │
       └── Sparse Pooling
            │
            └── point_pool()
                 │
                 └── voxel_downsample_csr_mapping()
                      │
                      └── to_unique_csr()
                           │
                           └── torch.unique()
                                │
                                └── CUDA ERROR
```

여기서는 모델의 실제 Attention 단계가 아니라 3D 포인트를 보셀 단위로 그룹화하는 처리 과정에서 오류가 발생했어.

이 오류 메시지는 실행하려는 CUDA 커널에서 현재 GPU용 실행 이미지를 찾지 못했다는 의미야. 하지만 그 커널이 어느 바이너리에 속하는지는 오류 메시지만으로 확정할 수 없어.

### 문제 7. T4 지원 여부와 `torch.unique()` 추가 조사

처음에는 PyTorch 또는 WarpConvNet이 T4용으로 컴파일되지 않았을 가능성을 검토했어.

그래서 PyTorch 자체가 T4를 지원하는지 검사했지.

GPU 아키텍처 검사

```
Compute Capability: (7, 5)

Compiled architectures:
['sm_70', 'sm_75', 'sm_80',
 'sm_86', 'sm_90', 'sm_100', 'sm_120']
```

기본 연산 검사

```
Basic CUDA: PASS
GPU unique: PASS
2048-point unique: PASS
```

즉, 설치된 PyTorch에는 `sm_75` 지원 코드가 포함되어 있고, 일반 CUDA 연산과 일부 `torch.unique()` 실행도 정상이었어.

그런데 SpaCeFormer에서는 여전히 오류가 발생했지.

이에 `CUDA_LAUNCH_BLOCKING=1`을 적용하고 실제 `torch.unique()`에 전달되는 입력을 조사했어.

최종 진단 로그

```
[DEBUG] torch.unique 호출

Shape       : (2048,)
Dtype       : torch.int64
Device      : cuda:0
Contiguous  : True
Stride      : (1,)

Arguments:
dim=0
sorted=True
return_inverse=True
return_counts=True

[DEBUG] CUDA synchronize BEFORE unique
[DEBUG] PRE-UNIQUE SYNC: PASS
```

그리고 바로 다음 `_VF.unique_dim()`에서 동일한 CUDA 오류가 발생했어.

이 결과로 중요한 차이를 발견했어.

| 항목       | 독립 테스트 성공   | 실제 SpaCeFormer 실패 |
| -------- | ----------- | ----------------- |
| 텐서 Shape | `(2048, 4)` | `(2048,)`         |
| Dtype    | `int32`     | `int64`           |
| GPU      | T4          | T4                |
| `dim`    | 0           | 0                 |
| 동기화      | 정상          | 직전 동기화 정상         |
| 결과       | PASS        | CUDA 오류           |

최종 판단: 원인 범위를 축소했지만 해결하지는 못함.

앞선 CUDA 연산의 비동기 오류 가능성은 낮아졌어. 다만 `torch.unique()`의 특정 입력 형태 또는 커널 실행 경로에 문제가 있는지, WarpConvNet의 별도 CUDA 바이너리와 연관된 문제인지는 아직 확정되지 않았어.

특히 정확히 같은 1차원 `int64` 입력을 독립 프로세스에서 재현하는 검사는 아직 수행하지 않았어.

## 4. 현재 남아 있는 문제의 원인 후보

지금까지의 결과를 종합해 보면 다음과 같아.

| 원인 후보                             | 현재 판단     | 근거                    |
| --------------------------------- | --------- | --------------------- |
| Colab 기본 Python 환경 문제             | 해결됨       | Python 3.12 가상환경으로 실행 |
| 체크포인트 접근 권한 문제                    | 해결됨       | 공식 가중치 다운로드 성공        |
| FlashAttention 미지원                | 임시 우회 성공  | SDPA로 모델 생성 성공        |
| PyTorch 전체의 T4 미지원                | 가능성 낮음    | `sm_75` 포함, 기본 연산 성공  |
| WarpConvNet CUDA 확장의 T4 호환성       | 미확인       | 별도 컴파일 바이너리 검사 필요     |
| PyTorch `unique_dim`의 특정 연산 경로 문제 | 유력한 조사 대상 | 1D int64 입력에서 실패      |
| 입력 데이터 형식 문제                      | 추가 검증 필요  | 공식 샘플로 아직 추론하지 않음     |
| GPU 메모리 부족                        | 근거 없음     | OOM이 아닌 커널 이미지 오류     |

여기서 가장 중요한 건 WarpConvNet의 빌드 아키텍처 문제와 `torch.unique()`의 특정 CUDA 커널 문제를 아직 구분하지 못했다는 사실이야.

이 점을 고려해 앞으로는 T4 전용 빌드를 진행하되, 재빌드만으로 반드시 해결된다고 가정하지 않는 것이 맞아.

## 5. 해결 방안 및 대체 방법 비교

### 방법 A. WarpConvNet을 T4 아키텍처에 맞게 재빌드

우선 진행할 방법

현재 사전 컴파일된 WarpConvNet을 그대로 쓰는 대신, Compute Capability 7.5를 대상으로 CUDA 확장 모듈을 직접 컴파일하는 방법이야.

공식 WarpConvNet 컴파일 가이드에서도 `TORCH_CUDA_ARCH_LIST`로 GPU 아키텍처를 지정하는 방법을 지원해. Turing 7.5도 명시되어 있고, 요구되는 최소 Compute Capability는 7.0이야.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://nvlabs.github.io\&sz=32)

WarpConvNet



예정된 빌드 구성은 다음과 같아.

| 항목          | 목표             |
| ----------- | -------------- |
| WarpConvNet | 1.8.2          |
| Git 커밋      | `e3ab140`      |
| PyTorch     | 기존 2.10.0 유지   |
| CUDA        | 기존 cu128 환경 유지 |
| GPU 아키텍처    | `7.5`          |
| 빌드 결과       | T4용 Wheel      |
| 보관 위치       | Google Drive   |

공식 릴리스에서 `v1.8.2`가 `e3ab140` 커밋에 해당하는 것도 확인했어.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



빌드 단계에서는 다음 설정을 사용하게 될 거야.

```
export TORCH_CUDA_ARCH_LIST="7.5"
export MAX_JOBS=2
export NVCC_THREADS=1
```

컴파일에 앞서 `nvcc --version`, Colab RAM, CUTLASS 서브모듈, 빌드 도구, 기존 버전 일치 여부를 확인해야 해.

장점: T4용 CUDA 확장 모듈을 명시적으로 빌드할 수 있고, 성공하면 Wheel을 저장해 향후 반복 사용하기 좋음.

한계: 오류가 PyTorch 자체의 특정 `unique_dim` 커널에 있다면 WarpConvNet 재빌드만으로 해결되지 않을 수 있음. FlashAttention 의존성도 별도로 남아 있음.

### 방법 B. PyTorch SDPA 대체 구현 유지

현재 이미 모델 생성까지 성공한 방법이야.

FlashAttention을 SDPA로 대체하고 기존 QKV와 3D RoPE 구조를 유지하는 방식이므로, T4에서 계속 시도할 가치가 있어.

다만 기존의 가변 길이 패치를 직접 분리해 연산하기 때문에 공식 FlashAttention 구현보다 느리거나 메모리를 더 사용할 수 있어.

또한 설치된 라이브러리에 영구 적용된 것이 아니라 실행 프로세스에서만 사용되는 임시 호환 계층이야. 장기 사용하려면 별도의 재사용 가능한 어댑터로 정리하고 실제 출력의 정확성을 검증해야 해.

### 방법 C. `torch.unique()`의 대체 경로 사용

현재 실패한 내부 함수는 `_VF.unique_dim()`이야.

1차원 텐서의 경우 `dim=0`과 `dim=None`에서 동일한 고유값을 계산할 수 있으므로, 다른 CUDA 연산 경로를 사용하는 방식이 후보가 될 수 있어.

다만 아직 정확히 동일한 입력에 대한 비교 테스트를 실행하지 않았고, 출력 순서와 역인덱스, 빈도의 동등성도 확인해야 해.

따라서 T4 전용 WarpConvNet 빌드 이후에도 동일한 오류가 발생하면 검토할 2차 대안으로 두는 게 적절해.

### 방법 D. Turing용 FlashAttention 사용

별도의 `flash-attention-turing` 구현이 존재해.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



이 구현이 현재 WarpConvNet에서 사용하는 `flash_attn_varlen_qkvpacked_func()` 호출 형식과 호환된다면 SDPA를 대체할 가능성이 있어.

하지만 지금까지 호환성 검증은 수행하지 않았어. 따라서 바로 설치하기보다 실제 API와 지원 기능을 확인해야 해.

### 방법 E. 상위 GPU 사용

T4 대신 L4, A100 등의 GPU를 사용하면 공식 FlashAttention-2가 지원하는 GPU 아키텍처를 활용할 수 있어.

단, Colab에서 해당 GPU를 할당받을 수 있어야 하고, 상위 GPU로 변경해도 모든 패키지 호환성 문제가 자동 해결되는 것은 아니야.

비용 및 GPU 사용 한도를 고려하면 지금은 우선순위를 낮추는 게 합리적이야.

### 방법 F. 다른 3D Instance Segmentation 모델 사용

SpaceWatch3D의 최종 목적은 SpaCeFormer 자체를 반드시 사용하는 것이 아니라 3D Instance Segmentation 모델을 평가하고 적합한 모델을 선정하는 것이야.

따라서 SpaCeFormer의 T4 실행이 계속 실패하면 Mask3D, SoftGroup 등의 다른 모델로 평가를 진행할 수 있어.

다만 이 모델들도 CUDA 확장 모듈을 사용하는 경우가 있으므로, T4에서 정상 실행된다는 보장은 없어.

## 6. 최종 결정한 진행 방향

지금까지 검토한 방법을 바탕으로 앞으로는 다음 순서로 진행하는 게 좋겠어.

01

현재 환경 보존

기존 PyTorch, 체크포인트, 설치 스크립트 및 정상 실행 로그 유지

다음 작업

02

T4 전용 WarpConvNet 빌드

CUDA 컴파일러와 RAM 확인 후 동일한 v1.8.2 소스를 sm_75 대상으로 빌드

미진행

03

빌드 결과 검증

CUDA 확장 로딩과 실제 오류 경로를 검사하고 필요하면 torch.unique 대체 경로 검토

미진행

04

SpaCeFormer Forward 재시도

SDPA 어댑터와 기존 체크포인트로 2,048개 포인트 테스트

미진행

05

공식 샘플로 전체 추론

실제 RGB 포인트 클라우드에서 마스크와 인스턴스 출력 검증

미진행

06

성능 평가 및 재현 환경 저장

정확도·추론시간·메모리 측정 후 검증된 Wheel과 실행 코드 저장

미진행

### T4용 WarpConvNet 빌드 시 주의할 점

기존 WarpConvNet을 먼저 제거하지 않고, 동일한 버전의 소스를 별도로 확보해 Wheel을 생성하는 방식으로 진행할 거야.

이렇게 하면 빌드가 실패해도 기존 환경을 보존할 수 있고, 빌드가 완료되면 새 Wheel을 검증한 뒤 설치할 수 있어.

또한 현재 PyTorch는 CUDA 기본 연산과 모델 로딩이 모두 성공했으므로, 당장은 PyTorch를 재설치하거나 CUDA 버전을 변경하지 않을 계획이야.

## 7. SpaceWatch3D 프로젝트 기준으로 아직 남은 작업

지금까지는 모델의 실행 가능성을 검증하는 단계였어. 우리가 실제로 필요한 것은 다음 출력이야.

LingBot-Map 결과 또는 공개 3D 포인트 클라우드

XYZ + RGB + 카메라/좌표계 정보

SpaCeFormer 3D Instance Segmentation

chair

floor

wall

ceiling

각 포인트의 객체 라벨 및 Instance ID

SpaCeFormer 공식 구현은 Open-Vocabulary 기반이기 때문에 클래스 이름을 텍스트로 지정할 수 있어. 다만 모델의 Forward 출력만으로 바로 클래스 이름이 확정되는 것은 아니야. SigLIP2 텍스트 임베딩을 이용한 클래스 매칭과 마스크 후처리를 포함해야 해.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



또한 공식 구현에서 내부 보셀화 이후의 출력 포인트 개수와 순서가 원본 포인트 클라우드와 달라질 수 있다고 설명하고 있어. 이는 LingBot-Map 결과의 원본 점마다 라벨을 부여하려는 우리 프로젝트에서 특히 중요해. 원본 포인트와 예측 마스크의 대응 관계를 별도로 검증해야 해.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

chrischoy/SpaCeFormer at main



### 앞으로 측정할 평가 항목

| 평가 항목                          | 현재 상태                |
| ------------------------------ | -------------------- |
| 모델 실행 성공 여부                    | 모델 로딩 성공, Forward 실패 |
| 클래스별 IoU / mIoU                | 미측정                  |
| Instance Segmentation AP / mAP | 미측정                  |
| 추론 시간                          | 미측정                  |
| 최대 GPU 메모리 사용량                 | 미측정                  |
| 입력 포인트 수에 따른 성능 변화             | 미측정                  |
| 원본 포인트와 예측 마스크 정합성             | 미검증                  |
| LingBot-Map 결과 연동 가능성          | 미검증                  |

추가로 공식 체크포인트는 ScanNet 계열 등 실내 장면을 대상으로 한 모델이므로, 우리 데이터에서도 같은 정확도를 보장하지 않아. 실제 데이터에 대한 별도 평가가 필요해.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

Hugging Face



## 8. 팀 공유용 최종 요약

> Google Colab의 Tesla T4 GPU 환경에서 SpaCeFormer 3D Instance Segmentation 모델의 실행 가능성을 검증하였다. Python 3.12.14, PyTorch 2.10.0+cu128, WarpConvNet 1.8.2 기반 환경을 구축하였으며, 공식 체크포인트 다운로드 및 모델 가중치 로딩에 성공하였다.
>
> 초기 모델 생성 과정에서는 FlashAttention 의존성 오류가 발생하였다. T4 GPU가 일반 FlashAttention-2의 지원 아키텍처에 포함되지 않는 점을 고려하여 PyTorch의 SDPA를 이용한 대체 연산을 구현하였고, 이를 통해 모델 생성과 가중치 로딩에 성공하였다. 체크포인트 로딩 결과 Missing 및 Unexpected key는 모두 0개로 확인되었다.
>
> 이후 2,048개 3D 포인트를 입력해 Forward 연산을 시도하였으나, 보셀 다운샘플링 과정의 `torch.unique()` 호출에서 CUDA 커널 오류가 발생하였다. PyTorch의 기본 CUDA 연산과 T4 아키텍처 지원은 확인했으며, 추가 디버깅을 통해 1차원 int64 텐서에 대한 `unique_dim` 연산에서 실패가 보고되는 것을 확인하였다.
>
> 현재는 WarpConvNet CUDA 확장의 GPU 아키텍처 호환성과 PyTorch의 특정 연산 경로를 추가 검증해야 하는 상태이다. 향후 기존 환경을 보존한 상태에서 Compute Capability 7.5를 대상으로 WarpConvNet 1.8.2를 소스 빌드하고, 실제 Forward 연산의 정상 실행 여부를 재검증할 예정이다.
>
> 현재까지 모델 실행 환경과 체크포인트 로딩은 성공하였으나, 실제 3D Instance Segmentation 추론 및 정확도·성능 평가는 완료되지 않았다.

## 최종 결론

SpaCeFormer를 T4 GPU에서 실행하기 위한 기본 환경, 모델 생성, 가중치 로딩까지는 성공했어. FlashAttention 문제도 SDPA 대체를 통해 모델 생성 단계에서는 우회했지.

그러나 실제 3D Forward에서 CUDA 커널 오류가 발생해, 현재는 추론에 성공하지 못한 상태야.

앞으로의 핵심은 다음 두 가지야.

첫째, WarpConvNet을 T4용으로 직접 빌드해서 안정적인 실행환경을 구축하는 것. 재빌드 후에도 같은 문제가 발생하면 `torch.unique()`의 특정 CUDA 경로를 별도로 조사해야 해.

둘째, 실제 포인트 클라우드 추론에 성공한 이후 정확도와 자원 사용량을 평가하는 것. 그 단계까지 완료되어야 SpaceWatch3D에 SpaCeFormer를 사용할 수 있는지 판단할 수 있어.

지금은 모델을 포기할 단계도 아니고, 정상 실행됐다고 결론 내릴 단계도 아니야. 다음 작업은 `nvcc`와 시스템 RAM을 확인한 후 WarpConvNet 1.8.2의 T4 전용 빌드를 진행하는 것으로 정리하면 돼.