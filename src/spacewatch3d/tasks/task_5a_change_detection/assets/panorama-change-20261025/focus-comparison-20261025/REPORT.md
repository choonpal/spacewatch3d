# 기존 정합오차·관측검증 / SimSaC / ViewDelta

기존 occupancy-evidence-v2 자동 판정을 고정하고 원본 면을 동일 RGB 시점에 first-hit 투영했다.
주황색 양성 영역은 후보를 발생시킨 트랙의 면만 포함한다. 그룹에 속한 주변 메시 전체를 양성으로 바꾸지 않았다.
보류, 그룹 문맥, 억제, 판정 연결이 없는 표면, mesh hit 없음은 별도 집계한다. 어느 것도 변화 없음의 정답은 아니다.

기존 전체 13구역의 후보 묶음 2개 중 선택된 R01, R07, R08 구역에 1개가 있다. 30뷰에 보이는 후보 트랙 표면은 중복 관측을 포함하여 4,681 픽셀이다.
기존 방식은 전체 촬영을 이용한 판정이고 두 모델은 파일럿의 20 ERP 프레임 / 30 방향 뷰를 사용했다. 관측량을 맞춘 성능 실험이 아니다.
Dorm2와 Dorm3는 서로 다른 층이며 물리적 시간 변화 정답이 없다. 후보 픽셀 크기는 물체 수·정확도가 아니다.

입력 기본값은 동일 방향 RGB. 새 모델의 모든 비영 대응 예측확률 평균 > 0.5를 그대로 사용했다.
ERP 입력 결과도 유지한다. 자기 입력 비교 후보는 두 모델 모두 0이었다.
정확히 같은 프레임의 기존 person segmentation과 겹침은 참고용이며 후보를 삭제하지 않는다. 검출 없음은 사람 부재 정답이 아니다.

## 현재 관찰

- R08의 기존 의자 후보가 보이는 3개 시점의 표면 4,681px 중, 동일 방향 RGB 입력 SimSaC는 1,127px와 겹쳤고 ViewDelta는 0px와 겹쳤다.
- ERP 전체 입력에서는 같은 표면에 SimSaC 3,376px, ViewDelta 0px가 겹쳤다. 입력 방식에 따라 반응이 달랐다.
- SimSaC는 R08의 기존 보류 영역도 선택했다. ViewDelta가 선택한 보류 면도 있어 판정 이유와 상대 RGB를 함께 보여준다.
- R07에서는 SimSaC가 촬영자 머리, ViewDelta가 머리 또는 주변 문·벽 영역을 선택한 사례가 있다. 두 모델의 원본 후보를 시설물 변화로 바로 채택하지 않았다.

## 후보 픽셀 분해

```json
{
  "view": {
    "simsac": {
      "pixels": 93826,
      "by_baseline": {
        "candidate": 1127,
        "deferred": 4419,
        "group_context": 8337,
        "suppressed": 22072,
        "unaudited": 57852,
        "no_mesh": 19
      }
    },
    "viewdelta": {
      "pixels": 61154,
      "by_baseline": {
        "candidate": 0,
        "deferred": 541,
        "group_context": 2,
        "suppressed": 39284,
        "unaudited": 21327,
        "no_mesh": 0
      }
    }
  },
  "erp": {
    "simsac": {
      "pixels": 63866,
      "by_baseline": {
        "candidate": 3376,
        "deferred": 12629,
        "group_context": 9467,
        "suppressed": 1523,
        "unaudited": 36871,
        "no_mesh": 0
      }
    },
    "viewdelta": {
      "pixels": 6,
      "by_baseline": {
        "candidate": 0,
        "deferred": 0,
        "group_context": 0,
        "suppressed": 0,
        "unaudited": 6,
        "no_mesh": 0
      }
    }
  }
}
```

validation.json에 입력 해시·깊이 검증, baseline-projections.npz에 원본 면·판정 bit·사람 마스크를 보존했다.
각 뷰 records에는 보이는 기존 객체 조각과 저장된 판정 이유가 있다. 픽셀 합계의 배타적 우선순위는 후보 > 보류 > 그룹 문맥 > 억제 > 미연결 > no hit이다.
