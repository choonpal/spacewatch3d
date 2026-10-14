# 실제 입력 데이터

촬영 영상과 실험 데이터셋을 이곳에 둡니다. 파일은 Git에서 제외됩니다.
공간별 scene_id, 회차별 capture_id로 구분하고 video manifest를 준비합니다.
작은 공유 예제는 `examples/fixtures/`를 사용합니다.

`virtual_tour/<영상 SHA-256>/`는 AUTO TOUR 360이 입력한 영상과 원본 정보를 보관하는 공간입니다. 앱 전용 `source.json`을 사용하며, 분석 task의 video manifest와는 별도입니다. 영상과 source.json을 함께 보관해야 서버 재시작 후 목록을 복원할 수 있습니다.
