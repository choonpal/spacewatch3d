from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# --------------------------------------------------
# 경로 설정
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "data" / "test_images" / "sample2.png"

CHECKPOINT = Path(
    "/home/lhw/nudix/sam2/checkpoints/sam2.1_hiera_tiny.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# SAM2 모델 준비
# --------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

sam2_model = build_sam2(
    MODEL_CFG,
    str(CHECKPOINT),
    device=device,
)

predictor = SAM2ImagePredictor(sam2_model)


# --------------------------------------------------
# 이미지 불러오기
# --------------------------------------------------
image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(image)

print("Image size:", image.size)
print()
print("사용법")
print("왼쪽 클릭  : Positive point (포함)")
print("오른쪽 클릭: Negative point (제외)")
print("R          : 모든 point 초기화")
print("S          : 현재 mask 저장")
print("Q          : 종료")


# 이미지 특징은 한 번만 계산
with torch.inference_mode():
    predictor.set_image(image_np)


# --------------------------------------------------
# 클릭 정보
# --------------------------------------------------
points = []
labels = []

current_mask = None
current_score = None


# --------------------------------------------------
# 화면 그리기
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))


def redraw():
    ax.clear()
    ax.imshow(image_np)

    # Mask가 있으면 표시
    if current_mask is not None:
        overlay = np.zeros(
            (*current_mask.shape, 4),
            dtype=np.float32,
        )

        overlay[current_mask] = [1.0, 0.0, 0.0, 0.40]

        ax.imshow(overlay)

    # point 표시
    for (x, y), label in zip(points, labels):
        if label == 1:
            ax.plot(
                x,
                y,
                marker="o",
                markersize=10,
                markeredgecolor="white",
                markerfacecolor="lime",
            )
        else:
            ax.plot(
                x,
                y,
                marker="x",
                markersize=12,
                markeredgewidth=3,
                color="red",
            )

    if current_score is None:
        ax.set_title("SAM2 - Click an object")
    else:
        ax.set_title(
            f"SAM2 - Mask score: {current_score:.4f}"
        )

    ax.axis("off")
    fig.canvas.draw_idle()


# --------------------------------------------------
# SAM2 prediction
# --------------------------------------------------
def run_prediction():
    global current_mask, current_score

    if len(points) == 0:
        current_mask = None
        current_score = None
        redraw()
        return

    point_coords = np.array(
        points,
        dtype=np.float32,
    )

    point_labels = np.array(
        labels,
        dtype=np.int32,
    )

    # 점 하나는 해석이 모호할 수 있으므로 mask 3개 생성
    # 여러 점을 줬으면 하나의 mask 출력
    multimask = len(points) == 1

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )

    best_index = int(np.argmax(scores))

    current_mask = masks[best_index].astype(bool)
    current_score = float(scores[best_index])

    print()
    print("Points:", points)
    print("Labels:", labels)
    print("Scores:", scores)
    print("Selected score:", current_score)

    redraw()


# --------------------------------------------------
# 마우스 클릭 이벤트
# --------------------------------------------------
def onclick(event):
    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    x = int(event.xdata)
    y = int(event.ydata)

    # 왼쪽 클릭 = positive
    if event.button == 1:
        points.append([x, y])
        labels.append(1)

        print(f"Positive: ({x}, {y})")

    # 오른쪽 클릭 = negative
    elif event.button == 3:
        points.append([x, y])
        labels.append(0)

        print(f"Negative: ({x}, {y})")

    else:
        return

    run_prediction()


# --------------------------------------------------
# 키보드 이벤트
# --------------------------------------------------
def onkey(event):
    global current_mask, current_score

    # Reset
    if event.key in ["r", "R"]:
        points.clear()
        labels.clear()

        current_mask = None
        current_score = None

        print("\nReset")
        redraw()

    # Save
    elif event.key in ["s", "S"]:
        if current_mask is None:
            print("저장할 mask가 없습니다.")
            return

        mask_path = OUTPUT_DIR / "click_mask.png"
        overlay_path = OUTPUT_DIR / "click_overlay.png"

        # Mask 저장
        mask_img = Image.fromarray(
            current_mask.astype(np.uint8) * 255
        )
        mask_img.save(mask_path)

        # Overlay 저장
        overlay_img = image_np.copy()

        overlay_img[current_mask] = (
            0.55 * overlay_img[current_mask]
            + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)

        Image.fromarray(overlay_img).save(overlay_path)

        print("Saved:")
        print(mask_path)
        print(overlay_path)

    # Quit
    elif event.key in ["q", "Q"]:
        plt.close(fig)


fig.canvas.mpl_connect(
    "button_press_event",
    onclick,
)

fig.canvas.mpl_connect(
    "key_press_event",
    onkey,
)

redraw()
plt.show()