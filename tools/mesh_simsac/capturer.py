"""Exclude nadir-reaching person instances, preserving separate bystanders."""
import numpy as np
from scipy.ndimage import binary_dilation, label
from PIL import Image


def semantic_capturer(person, bottom_fraction=.08):
    """Nadir-connected components of the unpadded semantic person class.

    Horizontal seams are joined before selection. Never use a union of padded
    detector boxes here: that can connect a bystander to the photographer.
    """
    person = np.asarray(person, bool)
    components, count = label(person)
    parent = list(range(count+1))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(components[:, 0], components[:, -1]):
        if a and b:
            parent[find(int(a))] = find(int(b))
    roots = np.array([find(i) for i in range(count+1)])
    components = roots[components]
    selected = np.unique(components[int(person.shape[0]*(1-bottom_fraction)):])
    selected = selected[selected > 0]
    mask = np.isin(components, selected) & person
    return mask, dict(method='Unpadded semantic person components reaching the bottom 8% of ERP; horizontal seam joined',
        excluded_pixels=int(mask.sum()), retained_person_pixels=int((person & ~mask).sum()),
        selected_components=len(selected), all_people_excluded=False)


def select_capturer(instances, shape, bottom_fraction=.08):
    """Select individual masks before union: a broad person union merges actors."""
    h, w = shape
    excluded = np.zeros(shape, bool)
    selected = []
    for i, instance in enumerate(instances):
        mask = np.asarray(instance, bool)
        if mask.shape != shape:
            raise ValueError('Person instance dimensions differ from the panorama')
        if mask[int(h*(1-bottom_fraction)):].any():
            excluded |= mask
            selected.append(i)
    return excluded, selected


class CapturerDetector:
    def __init__(self, checkpoint, device='cuda'):
        from ultralytics import YOLO
        self.model, self.device = YOLO(str(checkpoint)), device

    def predict(self, rgb):
        h, w = rgb.shape[:2]
        excluded = np.zeros((h, w), bool)
        records = []
        # Rolling brings a photographer split at the ERP seam into one image.
        for shift in [0, w//2]:
            image = np.roll(rgb, shift, axis=1)
            result = self.model.predict(image[..., ::-1].copy(), classes=[0],
                conf=.25, imgsz=960, device=self.device, verbose=False)[0]
            masks = []
            if result.masks is not None:
                for mask in result.masks.data.cpu().numpy():
                    mask = np.asarray(Image.fromarray(mask > .5).resize((w, h), Image.Resampling.NEAREST))
                    masks.append(mask)
            selected, indices = select_capturer(masks, (h, w))
            excluded |= np.roll(selected, -shift, axis=1)
            records.append(dict(roll=shift, detected_instances=len(masks),
                                capturer_instances=len(indices), retained_instances=len(masks)-len(indices)))
        excluded = binary_dilation(excluded, iterations=2)
        return excluded, dict(method='Separate YOLO person masks reaching the bottom 8% of the ERP; two horizontal rolls',
                              detections=records, excluded_pixels=int(excluded.sum()),
                              capturer_localized=bool(excluded.any()), all_people_excluded=False)
