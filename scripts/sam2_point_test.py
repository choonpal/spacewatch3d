import numpy as np
import torch
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# 1. 경로 설정
IMAGE_PATH = "data/test_images/sample1.png"

CHECKPOINT = "/home/lhw/nudix/sam2/checkpoints/sam2.1_hiera_tiny.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

MASK_OUTPUT = "outputs/monitor_mask.png"
OVERLAY_OUTPUT = "outputs/monitor_overlay.png"


# 2. GPU 확인
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# 3. SAM2 모델 불러오기
sam2_model = build_sam2(
    MODEL_CFG,
    CHECKPOINT,
    device=device,
)

predictor = SAM2ImagePredictor(sam2_model)


# 4. 이미지 읽기
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

print("Image size:", image.size)


# 5. SAM2에 이미지 등록
predictor.set_image(image_np)


# 6. 모니터 가운데에 Positive Point 하나 지정
# 좌표 형식: (x, y)
input_point = np.array([
    [650, 400],   # 모니터 내부
    [650, 750]    # 책상 쪽
])

# 1 = 포함시키고 싶은 영역
# 0 = 제외시키고 싶은 영역
input_label = np.array([
    1,  # 포함
    0   # 제외
])

# 7. Mask 예측
with torch.inference_mode():
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )


# 8. 가장 점수가 높은 mask 선택
best_index = np.argmax(scores)

best_mask = masks[best_index]
best_score = scores[best_index]

print("Scores:", scores)
print("Best mask index:", best_index)
print("Best score:", float(best_score))


# 9. 흑백 mask 저장
mask_image = Image.fromarray(
    (best_mask.astype(np.uint8) * 255)
)

mask_image.save(MASK_OUTPUT)


# 10. 원본 위에 mask를 반투명하게 표시
overlay = image_np.copy()

mask_bool = best_mask.astype(bool)

# mask 영역을 빨간색 계열로 표시
overlay_color = np.array([255, 0, 0], dtype=np.uint8)

overlay[mask_bool] = (
    0.55 * overlay[mask_bool]
    + 0.45 * overlay_color
).astype(np.uint8)

overlay_image = Image.fromarray(overlay)
overlay_image.save(OVERLAY_OUTPUT)


print("Saved:", MASK_OUTPUT)
print("Saved:", OVERLAY_OUTPUT)