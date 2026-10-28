촬영 카메라 위치의 ProPainter 360도 복원 파이프라인

서버: root@10.132.1.13, SSH 포트 4444
실행 폴더: /workspace/propainter_camera_20261028

처리 순서
촬영 프레임 ID 선택 → 주변 영상 투영 → SAM2 사람 마스크 추적 →
ProPainter 배경 복원 → 제거 마스크 내부만 원본 360도 이미지에 합성 → 뷰어 생성

키프레임은 config.json의 selected_camera_ids로 지정합니다.
현재 목록: [7, 123, 205]. 카메라 포즈는 data/camera_poses.npy에서 직접 가져옵니다.
프레임 ID는 source_manifest에 있는 촬영 이미지 인덱스입니다.
원본 영상 프레임 번호는 manifest의 source_frame_index를 읽습니다.

서버 실행
  cd /workspace/propainter_camera_20261028
  bash run.sh --plan-only --output /workspace/propainter_camera_20261028/check_plan
  bash run.sh --reuse-only --output /workspace/propainter_camera_20261028/results_reuse
  bash run.sh --output /workspace/propainter_camera_20261028/results_new

완료된 결과 폴더는 덮어쓰지 않습니다. 재실행할 때 새 --output 경로를 지정하세요.
--plan-only: 계획과 설정만 기록합니다. 영상 처리나 모델 실행은 하지 않습니다.
--reuse-only: 일치하는 기존 결과만 사용합니다. 새 모델 추론이 필요하면 중단합니다.
일반 실행: 기존 결과가 설정과 일치하면 재사용하고, 필요한 구간만 모델로 처리합니다.
GPU 실행은 launch.py가 여유 GPU 1개를 확보한 뒤 SAM2, ProPainter 순서로 진행합니다.
원본 영상의 SHA-256을 확인하며, 같은 대화의 기존 GPU 잠금도 공유합니다.

새 키프레임 및 구간
기존에 검토한 15개 구간 안의 촬영 프레임은 해당 구간을 재사용합니다.
기존 구간 밖의 프레임은 new_clip_settings에 그 카메라 ID의 projection과 prompt가 필요합니다.
projection: size [512,512], yaw_deg, pitch_down_deg, horizontal_and_vertical_fov_deg.
prompt: 첫 투영 프레임의 사람 box [x0,y0,x1,y1], points [[x,y],...], 선택적 labels.
새 구간은 선택 시점 앞뒤 1.5초, 원본 3프레임 간격으로 추출합니다.
프레임 번호로부터 구간을 만들며, 시작과 끝은 원본 영상 범위를 벗어나지 않습니다.
촬영자가 다른 곳에 있거나 투영 방향이 바뀌면 첫 프레임의 prompt도 다시 지정해야 합니다.
기존 구간의 투영 방향과 prompt는 원래 protocol.json/tracking_report.json에서 읽습니다.

주요 설정
  erp_resolution: [1024,512]
  mask.dilation_radius_pixels: 8
  mask.close_kernel_pixels: 5
  propainter.ref_stride: 5
  propainter.neighbor_length: 10
  propainter.subvideo_length: 80
  propainter.raft_iter: 20
  propainter.fp16: true
  propainter.mask_dilation: 0 (마스크 확장은 위 mask 설정에서 수행)
  propainter.save_fps: 10 (출력 동영상 재생 속도)
구간 길이·프레임 간격 설정은 새 구간에 적용됩니다. 기존 캐시의 구간은 그대로 유지됩니다.
복원 해상도를 높일 때에는 원본 투영 해상도와 마스크 지정 좌표도 함께 검토해야 합니다.

저장 내용
  plan.json: 실제 카메라 위치, 영상 프레임, 처리 구간과 캐시 사용 여부
  effective_config.json: 실행 설정
  clips/: 새로 처리한 투영 영상, 마스크, 중간 전파 결과, 모델 로그
  images/: 최종 360도 이미지
  originals/: 선택한 원본 360도 이미지
  masks/: 360도 이미지에서 교체한 영역
  records/: 선택 기록 및 원래 모델 실행 기록
  run_complete.json: 이번 실행의 단계별 시간과 완료 여부
  index.html, data.js, viewer.js: 뷰어
뷰어는 결과 폴더에서 python3 -m http.server 8765로 열 수 있습니다.

기존 캐시를 사용한 실행의 새 모델 추론 시간은 0으로 기록합니다.
원래 모델 처리 시간은 records/clip_*/tracking_report.json과 propainter_report.json에 있습니다.
공식 ProPainter 모델 및 SAM2 환경은 기존 서버 경로를 사용합니다.
경로의 옛 실험명은 디렉터리 식별자이며, 새 시점 렌더러를 실행하지 않습니다.

검증 범위
기존 3장 캐시를 사용한 서버 복원·뷰어 생성 및 픽셀 동일성을 확인합니다.
이번 코드 정리에서는 새로운 GPU 모델 추론을 실행하지 않습니다.
