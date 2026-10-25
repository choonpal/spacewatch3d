# SceneDiff 입력 경로 A/B 비교

동일 RGB에서 원본 Pi3+SAM 경로와 기존 메쉬+인스턴스 경로를 비교했다.
두 경로 모두 원본 후속 변화 후보 탐지·객체 병합·회차 간 대응을 실행했다. 기존 ID에 점수만 부여하는 실험과 다르다.

| 구역 | 원본 최종 후보 | 메쉬 최종 후보 | IoU≥0.5 대응 | 후보 픽셀 IoU |
|---|---:|---:|---:|---:|
| R01 | 0 | 4 | 0 | 0.0% |
| R07 | 5 | 3 | 0 | 19.9% |
| R08 | 9 | 8 | 0 | 20.8% |

통합 후보 합계: 원본 14, 메쉬 15. 구역 간 중복 제거는 하지 않았다.

입력: 같은 224×224 RGB 30장(3구역×2촬영×5장). 원본 기본 loader는 최대518이므로 본 결과는 해상도를 통제한 파일럿이다.
Dorm2와 Dorm3는 서로 다른 층이며 시간 변화 정답이 없다. 후보 감소가 정확도 향상을 뜻하지 않는다.
DINOv3 H+/16 실제 가중치와 SceneDiff의 동일 비용 가중치·max-entropy 방식 사용. 자동 임계값의 수치는 입력별로 결정된다.
메쉬 입력은 ray-cast depth, 기존 camera pose/intrinsics, 가림을 고려한 기존 SAM3.1 인스턴스 마스크이며 valid ray hit confidence=1.
원본의 스케일·voxel 반환 규칙을 유지했다. 입력별 voxel_size와 자동 임계값은 summary.json에 기록했다.
공통 호환 처리: 빈 영역 대응 보호, 빈/상수 임계값 입력 보호, 동일 voxel 평균의 스트리밍 계산, native PyTorch scatter_mean.
Upstream commit: 62dcbd285fcfe984bb3db55dc01bdcd1a0055714

원본 출력 RLE, 마스크·비용·가시성 배열, 모델 SHA256, RGB 해시는 이 폴더의 JSON/NPZ에 보관했다.
