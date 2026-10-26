"""Replay the actual Dorm registration refinement without changing source data."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from dorm_mesh_geometry import plane_samples, refine_similarity, transform


def support(source, target, source_normals, target_normals, matrix, height):
    moved = transform(source, matrix)
    rotation = matrix[:3, :3] / np.cbrt(np.linalg.det(matrix[:3, :3]))
    distance, near = cKDTree(target).query(moved)
    good = (distance < .05 * height) & (np.abs(np.sum(
        (source_normals @ rotation.T) * target_normals[near], axis=1)) > .9)
    reverse = cKDTree(moved).query(target)[0] < .05 * height
    return float((good.mean() + reverse.mean()) / 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    samples = np.load(args.folder / 'inspection/alignment-samples.npz')
    metadata = json.loads((args.folder / 'run-v6/alignment.json').read_text())
    before, nb = plane_samples(samples['before'], metadata['planes'][0])
    after, na = plane_samples(samples['after'], metadata['planes'][1])
    # Fixed spatial sampling preserves the actual corridor degeneracy.
    def thin(p, n):
        ids = np.unique(np.floor(p / .12).astype(int), axis=0, return_index=True)[1]
        return p[ids], n[ids]
    before, nb = thin(before, nb)
    after, na = thin(after, na)
    matrix = np.array(metadata['after_to_before'])
    height = metadata['quality']['reference_height_units']
    initial = support(after, before, na, nb, matrix, height)
    trace = []
    refined = refine_similarity(after, before, na, nb, matrix, height, trace=trace)
    final = support(after, before, na, nb, refined, height)
    print(json.dumps({'initial_support': initial, 'refined_support': final,
                      'source_points': len(after), 'target_points': len(before)}, indent=2))
    assert final >= initial - .005, 'Refinement destroys existing floor/wall overlap'
    assert all(step['after'] < step['before'] for step in trace), 'An objective-worsening update was accepted'


if __name__ == '__main__':
    main()
