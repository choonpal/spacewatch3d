# 가우시안 직접 입력 변화탐지 조사

확인일: 2026-10-25, Asia/Seoul. 논문·저자 프로젝트·공식 저장소를 확인했으며 모델 설치나 추론은 실행하지 않았다.

## 판정

**두 가우시안 맵의 파라미터를 직접 비교하는 방법은 있다. 가장 정확히 해당하는 것은 GS-Diff다. 그러나 확인 시점 공식 실행 코드는 아직 공개되지 않았다.** 공개 코드·가중치까지 갖춘 즉시 실행 모델로 소개하면 안 된다. **ChronoFuseGS는 공개 구현이 있지만, 다중 시점 맵 융합·재학습 및 변화 시각화가 목적이므로 요청한 변화탐지 모델의 적용 후보에서 제외한다.** GS-Pool도 두 맵을 받지만 원본 사진과 영상 특징을 함께 사용하는 객체 단위 방법이다.

| 방법 | 실제 변화 판단 입력 | 공개 상태와 적합성 |
| --- | --- | --- |
| [GS-Diff](https://chumsy0725.github.io/GS-DIFF/) | 두 독립 가우시안 맵의 위치·공분산·DC 색상과 카메라 관측기하 | 직접 primitive 비교. 학습된 신경망 가중치가 필요한 구조가 아니다. 공식 코드는 공개 예정. |
| [ChronoFuseGS](https://github.com/TobiasBat/ChronoFuseGS) | 여러 독립 Gaussian 맵 + 각 시점 사진·카메라 | **적용 후보 제외.** 공개 코드는 있으나 다중 시점 재구성·융합과 변화 시각화가 목적이며, 요청한 독립 변화탐지 모델로 제시하지 않는다. |
| [GS-Pool](https://arxiv.org/html/2610.06688v1) | 두 가우시안 맵 + 양쪽 원본 사진·카메라, SAM2 객체 마스크, DINOv3 증류 특징 | 객체별 Gaussian 집합을 반환한다. Gaussian-only는 아니다. [공식 저장소](https://github.com/BoazKG93/gs-pool)는 문서와 공개 예정 안내 상태. |
| [3DGS-CD](https://github.com/520xyxyzq/3DGS-CD) | 전 3DGS와 후 RGB 관측을 이용한 2D 변화·분할·3D 융합 | 실제 코드와 데이터·사전 장면 모델 다운로드 안내가 있다. 두 Gaussian 파라미터 집합 직접 비교와는 입력이 다르다. |
| [GaussianUpdate](https://zju3dv.github.io/GaussianUpdate/) | 이전 재구성 모델과 새 사진 | 변화 환경에서 맵을 계속 갱신하는 문제. 두 기존 가우시안 맵만의 변화 판별기가 아니다. |

## GS-Diff가 하는 일

독립 학습한 두 맵은 변화 없는 곳에서도 Gaussian의 개수·위치·크기가 달라질 수 있다. GS-Diff는 이를 재구성 drift로 다룬다. 비등방 공분산을 이용한 기하 커널과 diffuse/DC 색상 커널로 변화 점수를 따로 계산하며, 카메라 기하에서 구한 관측 불확실성을 반영한다. 외부 foundation model이나 변화 라벨 학습은 사용하지 않는다. [저자 프로젝트 설명](https://chumsy0725.github.io/GS-DIFF/)

논문의 입력은 공통 좌표계의 두 Gaussian 재구성과 양쪽 카메라다. 공통 관측 frustum을 먼저 고른다. 기하·색상 점수는 가우시안에서 직접 구하고, 최종 2D 마스크 표시 때 점수를 렌더링한다. 따라서 **RGB를 렌더링해서 RGB 변화 모델에 넣는 방식과 다르지만, 전체 파이프라인의 모든 렌더링을 없앤 것은 아니다.** 고차 SH는 외관 비교에서 제외하고 DC를 쓴다. 실험은 PGSR 재구성을 사용했다. [논문 §3 및 Appendix A](https://arxiv.org/html/2605.07203v2)

## 코드·가중치 실물 확인

- [공식 main](https://github.com/Chumsy0725/GS-DIFF/tree/main): `b3e4c596d84ea86f8aad4d115d1a52a41f35561d`. GitHub API 재귀 트리로 확인한 파일은 `LICENSE` 77 bytes, `README.md` 112 bytes 두 개다. 추론·데이터 로더·환경 설치 파일이 없다. [트리 API](https://api.github.com/repos/Chumsy0725/GS-DIFF/git/trees/main?recursive=1)
- 별도 `gh-pages` 브랜치도 확인했다. 정적 프로젝트 페이지와 이미지 등 20개 항목이며 추론 구현은 없다. [페이지 브랜치](https://github.com/Chumsy0725/GS-DIFF/tree/gh-pages)
- [Releases](https://github.com/Chumsy0725/GS-DIFF/releases)는 없고 태그도 없다. README·프로젝트에서 변화탐지 checkpoint 다운로드를 확인하지 못했다. 이는 단순한 가중치 누락 문제가 아니라, 학습 없는 알고리즘의 실행 구현이 미공개인 상태다.
- [프로젝트 페이지](https://chumsy0725.github.io/GS-DIFF/)는 코드·annotation 공개 예정으로 표시한다. 논문도 acceptance 후 공개를 안내한다.
- [현재 LICENSE](https://raw.githubusercontent.com/Chumsy0725/GS-DIFF/main/LICENSE)는 CC BY-NC 4.0 표기다. 향후 공개 코드의 라이선스는 공개 시 다시 확인해야 한다.
- [공개 요청 이슈](https://github.com/Chumsy0725/GS-DIFF/issues/1)와 [PGSR 설정 이슈](https://github.com/Chumsy0725/GS-DIFF/issues/2)에는 제3자의 재현 어려움 보고가 있다. 이는 저자의 검증 결과가 아니며, 우리 데이터에서 성능을 보장하는 근거로 쓰지 않는다.

## 현재 Dorm2/Dorm3에 연결할 때

아래는 기존 로컬 데이터 구조와 논문을 대조한 구현 가능성 판단이며, 공식 코드로 호환성을 시험한 결과는 아니다.

- 현재 full-SH3 Splatfacto 맵에는 위치·scale·quaternion·opacity·SH가 있으므로 공분산과 DC 색상 입력을 구성할 수 있다. 기존 저장 변환으로 위치와 공분산을 공통 프레임으로 옮길 수 있다.
- 카메라도 같은 변환을 적용해야 한다. 현재 고정된 구역별 정합을 쓸 경우 구역별 Gaussian과 카메라 관측기하의 일관성 및 구역 경계의 점수 불연속을 확인해야 한다.
- PGSR 대신 Splatfacto를 쓰므로 primitive의 표면 정렬·법선·기하 품질이 다르다. 논문 결과가 그대로 재현된다고 말할 수 없다. 논문 자체도 재구성과 pose 품질에 의존한다고 명시한다. [한계 §5](https://arxiv.org/html/2605.07203v2#S5)
- Dorm2·Dorm3는 서로 다른 층이다. 정합 후 공간 차이 비교는 가능하더라도 같은 공간의 시간적 변화 정확도를 검증하는 데이터로 해석할 수 없다.

Gaussian 속성에서 직접 점수를 계산하는 방향으로 진행하려면 논문 수식·의사코드에 따른 **GS-Diff 방식의 자체 구현**이 필요하다. 이를 저자의 공식 모델 실행이나 재현 완료로 부르면 안 된다. 아래 ChronoFuseGS는 관련 연구로만 기록하며, 요청한 변화탐지 모델의 대안으로 추천하지 않는다.

## 적용 후보에서 제외한 관련 연구: ChronoFuseGS

여러 시점에 각각 학습한 Gaussian 맵과 사진 집합을 받아 공통 프레임으로 융합한다. 각 Gaussian의 시점별 opacity activation/persistence와 조명 보상값을 최적화하고, 전체 맵을 함께 보정한다. 이 시점별 유지 여부로 작은 구조 변화까지 표시한다. [저자 페이지](https://tobiasbat.github.io/ChronoFuseGS/), [논문 §3–4](https://arxiv.org/html/2609.31339v1)

공식 GitHub의 `main` 재귀 트리를 실제 확인했다. 조회 revision은 `1d7f9210a007f1e91b0fedd34119d2f33f3aece7`이며 `chronoFuseGS_train/training/merge.py`와 `train.py`, Gaussian 자료구조·CUDA rasterizer, Unreal 시각화 소스가 있다. README만 있는 GS-Diff/GS-Pool과 공개 상태가 다르다. [트리 API](https://api.github.com/repos/TobiasBat/ChronoFuseGS/git/trees/main?recursive=1), [공식 저장소](https://github.com/TobiasBat/ChronoFuseGS)

- `merge.py`는 기존 모델의 PLY와 COLMAP 데이터·원본 사진을 읽고 모델 및 카메라 정합을 구성한다. [실제 소스](https://github.com/TobiasBat/ChronoFuseGS/blob/main/chronoFuseGS_train/training/merge.py)
- `train.py`의 `init_cross_timestep`은 다른 시점의 원본 사진과 differentiable render를 비교한다. 공개 LPIPS/AlexNet 특징을 쓰며 Gaussian activation과 조명 보상에 역전파한다. 이후 `combined_refinement`가 사진·SSIM·변화 보존 loss로 보정한다. 따라서 가우시안 맵을 입력받지만 RGB 증거를 전혀 쓰지 않는 직접 속성 비교는 아니다. SimSaC의 검출 마스크만 맵 위로 옮기는 구조와도 다르다. [실제 소스](https://github.com/TobiasBat/ChronoFuseGS/blob/main/chronoFuseGS_train/training/train.py)
- NWM 데이터 문서는 시점별 사전학습 3DGS/PuP 모델, COLMAP·사진, 융합 모델(`point_cloud.ply`/`activation.ply`) 다운로드를 안내한다. 이것은 특정 장면의 학습된 표현이며 우리 실내 맵에 그대로 쓰는 범용 변화탐지 checkpoint가 아니다. [공식 데이터 안내](https://github.com/TobiasBat/ChronoFuseGS/blob/main/NWM_dataset/README.md), [배포 DOI](https://doi.org/10.48436/bphp1-hbb20)
- 공식 시각화는 Windows Unreal Engine 5.6에서 테스트했고 SH0만 지원한다고 안내한다. 현재 브라우저의 full-SH3 맵 뷰어와 바로 호환되는 UI가 아니다. [공식 실행 안내](https://github.com/TobiasBat/ChronoFuseGS#change-visualization)
- 현재 Splatfacto의 두 맵을 적용하려면 모델·카메라·원본 사진 자료구조 변환과 구역 정합 처리, 장면별 최적화가 필요하다. 같은 장소의 시점별 맵을 전제로 하므로 Dorm2/Dorm3의 서로 다른 층 비교는 별도의 공간 차이 파일럿으로 해석해야 한다. 실제 설치·추론·성능 검증은 하지 않았다.

## 관련 방법을 구분할 때

[GS-Pool의 공식 Hugging Face 카드](https://huggingface.co/datasets/boazkg/gs-pool-paslcd)에는 실행 명령과 코드 링크, 모델·결과·evidence 설명이 있다. 하지만 연결된 [공식 GitHub](https://github.com/BoazKG93/gs-pool)는 `The code will be released here`라고 명시하며 실제 파일은 docs와 README 등이다. 카드의 명령이 있다는 이유만으로 추론 코드까지 공개됐다고 판정하지 않았다. 사전 계산 결과·재구성 모델 배포와 추론 구현 배포는 구분한다.

[PlenoCI 논문](https://arxiv.org/html/2609.28930v1)은 Gaussian에서 analytic plenoptic derivative를 계산하지만, 실제 변화탐지는 instance-aware render·RGB·semantic feature를 함께 쓰는 방식이다. Gaussian primitive 파라미터만 비교하는 GS-Diff와 분리한다. 저자 프로젝트 URL은 조사 도구에서 열리지 않아 현재 실행 코드 공개 여부를 확정하지 않았다.

**요청에 맞는 공개 구현은 현재 확보하지 못했다.** GS-Diff는 Gaussian 속성을 직접 비교하는 변화탐지 방법이지만 공식 코드는 미공개다. ChronoFuseGS는 맵 융합·재학습 및 변화 시각화 방법이므로 변화탐지 모델 대안에서 제외한다. GS-Diff 수식 자체 구현은 공식 공개 모델 실행과 구분한다. 현재 우리 데이터에 새 방법을 실행한 것은 아니다.

## 코드 공개 상태 재확인 — 2026-10-25 21:58 KST

검색 요약과 저장소 첫 화면만으로 판단하지 않고 GitHub API를 새로 조회해 전체 공개 브랜치의 재귀 트리·태그·릴리스·포크를 재확인했다.

- **GS-Diff:** 공개 브랜치는 `main`(`b3e4c596d84ea86f8aad4d115d1a52a41f35561d`)과 `gh-pages`(`4b5b1e4797c671898fa28b40e43677d6d40a0e4d`) 두 개다. main에는 LICENSE·README 두 파일만, gh-pages에는 소개 HTML·CSS·JS·그림·논문·데모 영상만 있다. 실행 구현·태그·릴리스·공개 포크는 없다. [공식 저장소](https://github.com/Chumsy0725/GS-DIFF), [브랜치 API](https://api.github.com/repos/Chumsy0725/GS-DIFF/branches), [릴리스](https://github.com/Chumsy0725/GS-DIFF/releases)
- 더 직접적인 근거를 찾았다. 저자이자 저장소 Owner인 `Chumsy0725`가 **2026-10-22** 이슈 #1에서 acceptance와 함께 코드·추가 데이터를 공개하며, **잠정 공개 시점은 2027년 초**라고 답했다. 이는 확정 출시일이 아니라 저자가 제시한 잠정 일정이다. [저자 답변](https://github.com/Chumsy0725/GS-DIFF/issues/1)
- **GS-Pool:** 공개 브랜치는 main 하나(`dedc4a544494868a5be13184cd560136b7e1f413`)다. 재귀 트리 74항목은 README·LICENSE·gitignore와 docs 정적 자료이며 Python·shell·notebook 추론 구현은 없다. 태그·릴리스·공개 포크도 없다. 공식 README는 코드 공개 예정이라고 명시한다. [공식 저장소](https://github.com/BoazKG93/gs-pool), [트리 API](https://api.github.com/repos/BoazKG93/gs-pool/git/trees/main?recursive=1)

따라서 재확인 시점에도 GS-Diff·GS-Pool의 공식 실행 코드는 공개되지 않은 상태다. 위에서 확인한 ChronoFuseGS의 실제 공개 구현과 구분한다.
