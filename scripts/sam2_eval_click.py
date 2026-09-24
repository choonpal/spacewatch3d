from pathlib import Path
import time

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ============================================================
# 1. 경로 설정
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "test_images"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 현재 사용 중인 SAM2 Tiny 모델
CHECKPOINT = Path(
    "/home/lhw/nudix/sam2/checkpoints/sam2.1_hiera_tiny.pt"
)

MODEL_CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"


# ============================================================
# 2. 이미지 안전하게 불러오기
# ============================================================

def load_image_with_orientation(image_path):
    """
    이미지의 EXIF Orientation 정보를 실제 픽셀에 적용하여 읽는다.

    스마트폰 사진 등은 실제 픽셀 배열과 화면에 표시되는 방향이
    다를 수 있기 때문에 ImageOps.exif_transpose()를 적용한다.
    """

    image = Image.open(image_path)

    # EXIF 회전 정보 적용
    image = ImageOps.exif_transpose(image)

    # SAM2 입력용 RGB
    image = image.convert("RGB")

    return image


def load_gt_mask(gt_path):
    """
    GT mask를 grayscale로 읽고 binary mask로 변환한다.

    검정 = False
    흰색 = True
    """

    gt_image = Image.open(gt_path).convert("L")

    gt_mask = np.array(gt_image) > 0

    return gt_image, gt_mask


# ============================================================
# 3. 테스트 이미지 선택
# ============================================================

def select_image():
    """
    data/test_images 안의 원본 이미지 목록 출력.

    *_gt.png 파일은 원본 이미지 목록에서 제외한다.
    """

    extensions = {
        ".png",
        ".jpg",
        ".jpeg"
    }

    images = []

    for path in sorted(IMAGE_DIR.iterdir()):

        if path.suffix.lower() not in extensions:
            continue

        # GT mask 제외
        if "_gt" in path.stem:
            continue

        images.append(path)

    if not images:
        raise FileNotFoundError(
            f"\n이미지를 찾을 수 없습니다.\n"
            f"경로: {IMAGE_DIR}"
        )

    print()
    print("=" * 50)
    print("테스트 이미지 선택")
    print("=" * 50)

    for i, image_path in enumerate(images, start=1):

        # GT 존재 여부 확인
        gt_files = list(
            IMAGE_DIR.glob(
                f"{image_path.stem}_*_gt.png"
            )
        )

        if gt_files:
            status = "GT 있음"
        else:
            status = "GT 없음"

        print(
            f"{i}. {image_path.name} "
            f"[{status}]"
        )

    while True:

        try:

            choice = int(
                input("\n번호 선택: ")
            )

            if 1 <= choice <= len(images):
                return images[choice - 1]

            print("올바른 번호를 입력하세요.")

        except ValueError:
            print("숫자를 입력하세요.")


# ============================================================
# 4. GT mask 검색
# ============================================================

def find_gt_masks(image_path):
    """
    예:

    sample3.png
    sample3_chair_gt.png

    → chair GT 자동 검색

    여러 객체가 있으면:

    sample3_chair_gt.png
    sample3_wall_gt.png
    sample3_floor_gt.png

    모두 인식한다.
    """

    pattern = (
        f"{image_path.stem}_*_gt.png"
    )

    gt_paths = sorted(
        IMAGE_DIR.glob(pattern)
    )

    if not gt_paths:

        raise FileNotFoundError(
            "\nGT mask를 찾을 수 없습니다.\n\n"
            f"선택한 이미지: {image_path.name}\n"
            f"예상 파일 이름:\n"
            f"{image_path.stem}_chair_gt.png"
        )

    gt_masks = {}

    prefix = image_path.stem + "_"
    suffix = "_gt"

    for path in gt_paths:

        name = path.stem

        # sample3_chair_gt
        # → chair
        label = name[
            len(prefix):-len(suffix)
        ]

        gt_masks[label] = path

    return gt_masks


# ============================================================
# 5. 평가할 객체 선택
# ============================================================

def select_gt_label(gt_masks):

    labels = list(gt_masks.keys())

    # GT가 하나뿐이면 자동 선택
    if len(labels) == 1:

        label = labels[0]

        print()
        print(
            f"평가 객체 자동 선택: {label}"
        )

        return label, gt_masks[label]

    print()
    print("=" * 50)
    print("평가 객체 선택")
    print("=" * 50)

    for i, label in enumerate(
        labels,
        start=1
    ):
        print(f"{i}. {label}")

    while True:

        try:

            choice = int(
                input("\n번호 선택: ")
            )

            if 1 <= choice <= len(labels):

                label = labels[
                    choice - 1
                ]

                return (
                    label,
                    gt_masks[label]
                )

            print("올바른 번호를 입력하세요.")

        except ValueError:
            print("숫자를 입력하세요.")


# ============================================================
# 6. SAM2 모델 로드
# ============================================================

def load_sam2():

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print()
    print("=" * 50)
    print("SAM2 모델 로딩")
    print("=" * 50)

    print(
        f"Device     : {device}"
    )

    print(
        f"Checkpoint : {CHECKPOINT}"
    )

    print(
        f"Config     : {MODEL_CFG}"
    )

    if not CHECKPOINT.exists():

        raise FileNotFoundError(
            "\nSAM2 checkpoint가 없습니다.\n"
            f"{CHECKPOINT}"
        )

    sam2_model = build_sam2(
        MODEL_CFG,
        str(CHECKPOINT),
        device=device
    )

    predictor = SAM2ImagePredictor(
        sam2_model
    )

    return predictor, device


# ============================================================
# 7. 이미지 / GT 크기 검사
# ============================================================

def validate_image_and_gt(
    image,
    gt_image,
    image_path,
    gt_path
):

    image_width, image_height = image.size
    gt_width, gt_height = gt_image.size

    print()
    print("=" * 50)
    print("이미지 / GT 정보")
    print("=" * 50)

    print(
        f"Image : {image_path.name}"
    )

    print(
        f"       width={image_width}, "
        f"height={image_height}"
    )

    print(
        f"GT    : {gt_path.name}"
    )

    print(
        f"       width={gt_width}, "
        f"height={gt_height}"
    )

    # 완전히 일치
    if (
        image_width == gt_width
        and image_height == gt_height
    ):

        print()
        print(
            "Image와 GT 크기 일치: OK"
        )

        return

    # 가로/세로가 정확하게 뒤집힌 경우
    if (
        image_width == gt_height
        and image_height == gt_width
    ):

        raise ValueError(
            "\n원본 이미지와 GT의 가로/세로가 "
            "서로 뒤집혀 있습니다.\n\n"
            "EXIF orientation 적용 후에도 "
            "방향이 맞지 않습니다.\n"
            "GT를 임의로 resize하거나 transpose하지 마세요.\n\n"
            f"Image : "
            f"{image_width} x {image_height}\n"
            f"GT    : "
            f"{gt_width} x {gt_height}"
        )

    # 기타 크기 불일치
    raise ValueError(
        "\n원본 이미지와 GT mask의 "
        "크기가 다릅니다.\n\n"
        f"Image : "
        f"{image_width} x {image_height}\n"
        f"GT    : "
        f"{gt_width} x {gt_height}\n\n"
        "GT mask가 동일한 원본 이미지에서 "
        "생성되었는지 확인하세요."
    )


# ============================================================
# 8. 성능 지표 계산
# ============================================================

def calculate_metrics(
    pred_mask,
    gt_mask
):

    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # True Positive
    tp = np.logical_and(
        pred,
        gt
    ).sum()

    # False Positive
    fp = np.logical_and(
        pred,
        np.logical_not(gt)
    ).sum()

    # False Negative
    fn = np.logical_and(
        np.logical_not(pred),
        gt
    ).sum()

    # True Negative
    tn = np.logical_and(
        np.logical_not(pred),
        np.logical_not(gt)
    ).sum()

    # --------------------------------------------
    # IoU
    # --------------------------------------------

    union = np.logical_or(
        pred,
        gt
    ).sum()

    if union > 0:
        iou = tp / union
    else:
        iou = 0.0

    # --------------------------------------------
    # Dice
    # --------------------------------------------

    dice_denominator = (
        pred.sum()
        + gt.sum()
    )

    if dice_denominator > 0:

        dice = (
            2 * tp
            / dice_denominator
        )

    else:
        dice = 0.0

    # --------------------------------------------
    # Precision
    # --------------------------------------------

    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0.0

    # --------------------------------------------
    # Recall
    # --------------------------------------------

    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0.0

    # --------------------------------------------
    # Pixel Accuracy
    # --------------------------------------------

    total_pixels = (
        tp + fp + fn + tn
    )

    if total_pixels > 0:

        pixel_accuracy = (
            (tp + tn)
            / total_pixels
        )

    else:
        pixel_accuracy = 0.0

    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
        "pixel_accuracy":
            float(pixel_accuracy),

        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


# ============================================================
# 9. 결과 파일 저장
# ============================================================

def save_result(
    image_np,
    pred_mask,
    gt_mask,
    image_path,
    label
):

    base_name = (
        f"{image_path.stem}_{label}"
    )

    pred_path = (
        OUTPUT_DIR /
        f"{base_name}_sam2_mask.png"
    )

    comparison_path = (
        OUTPUT_DIR /
        f"{base_name}_comparison.png"
    )

    # --------------------------------------------
    # Prediction binary mask
    # --------------------------------------------

    pred_image = (
        pred_mask.astype(
            np.uint8
        ) * 255
    )

    Image.fromarray(
        pred_image
    ).save(pred_path)

    # --------------------------------------------
    # 비교 이미지
    # --------------------------------------------

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7)
    )

    # 원본
    axes[0].imshow(image_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # GT
    axes[1].imshow(
        gt_mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Ground Truth"
    )

    axes[1].axis("off")

    # SAM2
    axes[2].imshow(
        pred_mask,
        cmap="gray"
    )

    axes[2].set_title(
        "SAM2 Prediction"
    )

    axes[2].axis("off")

    plt.tight_layout()

    fig.savefig(
        comparison_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print()
    print("결과 저장 완료")

    print(
        f"SAM2 Mask : {pred_path}"
    )

    print(
        f"Comparison: "
        f"{comparison_path}"
    )


# ============================================================
# 10. SAM2 클릭 평가
# ============================================================

def run_click_evaluation(
    predictor,
    image_path,
    gt_path,
    label
):

    # --------------------------------------------
    # 원본 이미지 로드
    # --------------------------------------------

    image = load_image_with_orientation(
        image_path
    )

    image_np = np.array(image)

    # --------------------------------------------
    # GT mask 로드
    # --------------------------------------------

    gt_image, gt_mask = load_gt_mask(
        gt_path
    )

    # --------------------------------------------
    # 크기 / 방향 검사
    # --------------------------------------------

    validate_image_and_gt(
        image,
        gt_image,
        image_path,
        gt_path
    )

    print()
    print(
        f"평가 Class : {label}"
    )

    # --------------------------------------------
    # SAM2에 이미지 embedding 생성
    # --------------------------------------------

    print()
    print(
        "SAM2 이미지 embedding 생성 중..."
    )

    embedding_start = (
        time.perf_counter()
    )

    predictor.set_image(
        image_np
    )

    embedding_time = (
        time.perf_counter()
        - embedding_start
    )

    print(
        f"Embedding Time : "
        f"{embedding_time * 1000:.2f} ms"
    )

    # --------------------------------------------
    # Prompt 상태
    # --------------------------------------------

    points = []
    point_labels = []

    current_mask = None
    current_score = None
    current_time = None

    # --------------------------------------------
    # GUI
    # --------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    # --------------------------------------------------------
    # 화면 다시 그리기
    # --------------------------------------------------------

    def redraw():

        ax.clear()

        ax.imshow(
            image_np
        )

        # SAM2 segmentation overlay
        if current_mask is not None:

            overlay = np.zeros(
                (
                    current_mask.shape[0],
                    current_mask.shape[1],
                    4
                ),
                dtype=np.float32
            )

            # 빨간색 overlay
            overlay[:, :, 0] = 1.0

            overlay[:, :, 3] = (
                current_mask.astype(
                    np.float32
                )
                * 0.45
            )

            ax.imshow(
                overlay
            )

        # 클릭 위치 표시
        for (
            (x, y),
            point_label
        ) in zip(
            points,
            point_labels
        ):

            if point_label == 1:

                # Positive
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=10,
                    markeredgewidth=2
                )

            else:

                # Negative
                ax.plot(
                    x,
                    y,
                    marker="x",
                    markersize=10,
                    markeredgewidth=2
                )

        ax.set_title(
            f"{image_path.name}\n"
            f"Evaluation class: {label}\n\n"
            "Left Click = Positive | "
            "Right Click = Negative\n"
            "E = Evaluate | "
            "R = Reset | "
            "Q = Quit"
        )

        ax.axis("off")

        fig.canvas.draw_idle()

    # --------------------------------------------------------
    # SAM2 예측
    # --------------------------------------------------------

    def predict():

        nonlocal current_mask
        nonlocal current_score
        nonlocal current_time

        if not points:
            return

        coords = np.array(
            points,
            dtype=np.float32
        )

        labels_np = np.array(
            point_labels,
            dtype=np.int32
        )

        start = time.perf_counter()

        with torch.inference_mode():

            masks, scores, _ = (
                predictor.predict(
                    point_coords=coords,
                    point_labels=labels_np,

                    # 첫 클릭은 여러 mask 후보 생성
                    multimask_output=(
                        len(points) == 1
                    )
                )
            )

        # CUDA는 비동기 연산이므로
        # 실제 시간 측정을 위해 synchronize
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        # SAM2가 예측한 score가 가장 높은 mask 선택
        best_index = int(
            np.argmax(scores)
        )

        current_mask = (
            masks[best_index]
            .astype(bool)
        )

        current_score = float(
            scores[best_index]
        )

        current_time = elapsed

        print()
        print("-" * 50)
        print("SAM2 Prediction")
        print("-" * 50)

        print(
            f"Prompt Points : "
            f"{len(points)}"
        )

        print(
            f"SAM2 Score    : "
            f"{current_score * 100:.2f}%"
        )

        print(
            f"Inference Time: "
            f"{current_time * 1000:.2f} ms"
        )

    # --------------------------------------------------------
    # 마우스 클릭
    # --------------------------------------------------------

    def on_click(event):

        if event.inaxes != ax:
            return

        if (
            event.xdata is None
            or event.ydata is None
        ):
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # --------------------------------------------
        # Left click
        # Positive point
        # --------------------------------------------

        if event.button == 1:

            points.append(
                (x, y)
            )

            point_labels.append(
                1
            )

            print()
            print(
                "Positive Point : "
                f"({x:.1f}, {y:.1f})"
            )

        # --------------------------------------------
        # Right click
        # Negative point
        # --------------------------------------------

        elif event.button == 3:

            points.append(
                (x, y)
            )

            point_labels.append(
                0
            )

            print()
            print(
                "Negative Point : "
                f"({x:.1f}, {y:.1f})"
            )

        else:
            return

        predict()
        redraw()

    # --------------------------------------------------------
    # 키보드 입력
    # --------------------------------------------------------

    def on_key(event):

        nonlocal current_mask
        nonlocal current_score
        nonlocal current_time

        if event.key is None:
            return

        key = event.key.lower()

        # --------------------------------------------
        # R = Reset
        # --------------------------------------------

        if key == "r":

            points.clear()
            point_labels.clear()

            current_mask = None
            current_score = None
            current_time = None

            print()
            print(
                "Prompt Reset 완료"
            )

            redraw()

        # --------------------------------------------
        # E = Evaluate
        # --------------------------------------------

        elif key == "e":

            if current_mask is None:

                print()
                print(
                    "먼저 객체 내부를 "
                    "클릭하세요."
                )

                return

            metrics = calculate_metrics(
                current_mask,
                gt_mask
            )

            print()
            print("=" * 60)
            print(
                "SAM2 Segmentation Evaluation"
            )
            print("=" * 60)

            print(
                f"Image : "
                f"{image_path.name}"
            )

            print(
                f"Class : "
                f"{label}"
            )

            print(
                f"Prompt Points : "
                f"{len(points)}"
            )

            print("-" * 60)

            # 가장 중요한 지표
            print(
                f"Segmentation IoU : "
                f"{metrics['iou'] * 100:.2f}%"
            )

            print(
                f"Dice             : "
                f"{metrics['dice'] * 100:.2f}%"
            )

            print(
                f"Precision        : "
                f"{metrics['precision'] * 100:.2f}%"
            )

            print(
                f"Recall           : "
                f"{metrics['recall'] * 100:.2f}%"
            )

            print(
                f"Pixel Accuracy   : "
                f"{metrics['pixel_accuracy'] * 100:.2f}%"
            )

            if current_score is not None:

                print(
                    f"SAM2 Score       : "
                    f"{current_score * 100:.2f}%"
                )

            if current_time is not None:

                print(
                    f"Inference Time   : "
                    f"{current_time * 1000:.2f} ms"
                )

            print("-" * 60)

            print(
                f"TP : {metrics['tp']}"
            )

            print(
                f"FP : {metrics['fp']}"
            )

            print(
                f"FN : {metrics['fn']}"
            )

            print(
                f"TN : {metrics['tn']}"
            )

            print("=" * 60)

            save_result(
                image_np,
                current_mask,
                gt_mask,
                image_path,
                label
            )

        # --------------------------------------------
        # Q = Quit
        # --------------------------------------------

        elif key == "q":

            plt.close(fig)

    # --------------------------------------------------------
    # Event 연결
    # --------------------------------------------------------

    fig.canvas.mpl_connect(
        "button_press_event",
        on_click
    )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    redraw()

    plt.show()


# ============================================================
# 11. Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SAM2 Segmentation Evaluation Tool")
    print("=" * 60)

    # --------------------------------------------
    # 이미지 선택
    # --------------------------------------------

    image_path = select_image()

    # --------------------------------------------
    # GT 검색
    # --------------------------------------------

    gt_masks = find_gt_masks(
        image_path
    )

    # --------------------------------------------
    # 평가 객체 선택
    # --------------------------------------------

    label, gt_path = (
        select_gt_label(
            gt_masks
        )
    )

    # --------------------------------------------
    # SAM2 로드
    # --------------------------------------------

    predictor, device = load_sam2()

    # --------------------------------------------
    # 평가 실행
    # --------------------------------------------

    run_click_evaluation(
        predictor,
        image_path,
        gt_path,
        label
    )


if __name__ == "__main__":
    main()