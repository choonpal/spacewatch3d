"""Routing metadata only; algorithms belong in their task packages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    package: str
    title: str
    inputs: dict[str, str]
    output: str
    fixture_inputs: dict[str, str]
    fixture_output: str


TASKS = {
    "1a": TaskSpec("task_1a_rgb_input", "RGB 영상 입력", {"video": "video"}, "frames",
                   {"video": "before/video.json"}, "before/frames.json"),
    "1b": TaskSpec("task_1b_360_input", "360° 영상 입력", {"video": "video"}, "frames",
                   {"video": "panorama/video.json"}, "panorama/frames.json"),
    "2": TaskSpec("task_2_reconstruction", "포인트클라우드 생성", {"frames": "frames"}, "reconstruction",
                  {"frames": "before/frames.json"}, "before/reconstruction.json"),
    "3": TaskSpec("task_3_instance_segmentation", "3D 인스턴스 분할", {"reconstruction": "reconstruction"}, "instances",
                  {"reconstruction": "before/reconstruction.json"}, "before/instances.json"),
    "4": TaskSpec("task_4_object_export", "객체별 GLB/OBJ 저장", {"instances": "instances"}, "objects",
                  {"instances": "before/instances.json"}, "before/objects.json"),
    "5a": TaskSpec("task_5a_change_detection", "공간 정합 및 변화탐지", {"before": "objects", "after": "objects"}, "changes",
                   {"before": "before/objects.json", "after": "after/objects.json"}, "changes.json"),
    "5b": TaskSpec("task_5b_keyframes", "객체 대표 키프레임 선별", {"objects": "objects"}, "keyframes",
                   {"objects": "before/objects.json"}, "before/keyframes.json"),
}
