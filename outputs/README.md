# 실행 결과

`outputs/<capture_id>/<task>/<run_id>/`처럼 실행마다 별도 경로를 사용합니다.
각 task는 자신의 manifest와 산출물을 생성하고 상위 결과는 상대 경로로 참조합니다.
이 디렉터리의 결과 파일은 Git에서 제외됩니다.

독립 앱 AUTO TOUR 360은 `virtual_tour/analysis/<영상 SHA-256>/`에 촬영 경로·분석 상태·작업 파일을, `virtual_tour/tours/`에 저장한 투어 JSON을 보관합니다. `virtual_tour/migration-*.json`은 기존 앱에서 복사한 파일의 검증 기록입니다. 이 앱의 JSON 형식은 분석 task의 공통 artifact 규약과 별도입니다.
