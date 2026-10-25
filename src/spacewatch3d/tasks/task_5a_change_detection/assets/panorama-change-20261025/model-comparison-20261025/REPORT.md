# ViewDelta·SimSaC·SceneDiff 실제 후보 비교

파노라마 전체 입력과 동일 방향 뷰 입력을 별도로 실행했다. 두 새 모델은 원본 공개 가중치를 strict=True로 로드했다.

SceneDiff 후보 수(통합 ID)와 새 모델의 2D 연결요소 수는 의미가 다르므로 같은 객체 수로 나란히 평가하지 않는다.

## 전체 ERP 입력을 공통 뷰에 투영한 후보 픽셀 비율

| 구역 | SceneDiff 원본 | SceneDiff 메쉬 | ViewDelta | SimSaC |
|---|---:|---:|---:|---:|
| R01 | 0.00% | 10.63% | 0.00% | 0.00% |
| R07 | 20.41% | 15.94% | 0.00% | 0.03% |
| R08 | 6.69% | 5.41% | 0.00% | 12.69% |

## 새 모델 입력 방식에 따른 공통 30뷰의 후보 영역

SceneDiff는 ERP 전체로 다시 추론한 결과가 아니다. 기존 30개 방향 뷰의 원본 후보9.03%, 메쉬 후보10.66%를 두 입력 방식 모두의 고정 비교군으로 사용했다.

| 모델 | ERP 전체 입력 | 동일 방향 뷰 입력 |
|---|---:|---:|
| ViewDelta | 0.0004% (6px) | 4.06% (61,154px) |
| SimSaC | 4.2428% (63,866px) | 6.23% (93,826px) |

## 관찰과 실행 범위

ERP 전체 입력에서 ViewDelta는 공통 30뷰에 6픽셀만 후보로 남겼다. 전체 ERP의 다른 방향에 후보가 없었다는 뜻은 아니다.
동일 방향 뷰 입력에서는 ViewDelta와 SimSaC가 R07의 촬영자 머리를 후보로 선택한 장면도 확인됐다. 후보를 곧바로 가구·시설물 재배치로 확정하지 않았다.
CYWS는 이 실행 비교에 포함하지 않았다. 원본 가중치 재사용 조건과 현재 환경의 PyTorch3D 부재는 research.md에 따로 기록했다.

## 조건

- 동일 SceneDiff 30뷰 / 대응 ERP 20프레임. 기존 인스턴스 파일럿 구간에 한정한다.
- ERP 전체는1024×512. ViewDelta의 원본256×256 전처리와SimSaC520×520/256×256 두 해상도를 사용한다. 구면 전용 구조로 바꾸지 않았다.
- 방향 뷰 대조에서는 기존SceneDiff의224×224 RGB 바이트를 그대로 사용한다.
- 기존메쉬SceneDiff의view-overlap 코드로상대를고정. 모든nonzero상대의예측확률을평균하고0.5로판정한다.
- 후보마스크를기존객체ID로사전제한하지않았다. 두모델은pixelmask를직접예측한다.
- 후보픽셀IoU는모델간동의정도다. 서로다른층이며변화정답이없어성능개선율이나정확도로해석하지않는다.
- ViewDelta의DeepSpeed는사용하지않는import만제외했고, SimSaC의CUDA/CuPycorrelation은같은9×9/mean연산을nativeTorch로실행했다.
- 실제가중치SHA256·sourcecommit·strictload·동일프레임대조·모든쌍예측확률을보존한다.

## 동일 입력 대조

{
  "viewdelta": {
    "view": {
      "candidate_fraction": 0.0,
      "mean_probability": 0.023867398500442505,
      "max_probability": 0.49433037638664246
    },
    "erp": {
      "candidate_fraction": 0.0,
      "mean_probability": 0.0244632288813591,
      "max_probability": 0.3040613830089569
    }
  },
  "simsac": {
    "view": {
      "candidate_fraction": 0.0,
      "mean_probability": 0.013222860172390938,
      "max_probability": 0.11631235480308533
    },
    "erp": {
      "candidate_fraction": 0.0,
      "mean_probability": 0.006943448446691036,
      "max_probability": 0.08929136395454407
    }
  }
}
