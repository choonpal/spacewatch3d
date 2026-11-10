"""Integer vote aggregation with identical ambiguous-frame/tie rejection rules."""
import numpy as np


def aggregate(face, frame, label, nfaces, unknown, minimum_frames=2, agreement=.6,
              minimum_pixels=1, backend='numpy'):
    final = np.full(nfaces, unknown, np.int32)
    counts = np.zeros(nfaces, np.uint32); ratios = np.zeros(nfaces, np.float32)
    if not len(face):
        return final, counts, ratios
    face, frame, label = (np.asarray(a, dtype=np.int64) for a in (face, frame, label))
    if not (face.shape == frame.shape == label.shape) or face.min() < 0 or face.max() >= nfaces or frame.min() < 0 or label.min() < 0:
        raise ValueError('Invalid face/frame/class vote IDs')
    if backend == 'cuda':
        ids, winners, maximum, total, ties = cuda_counts(face, frame, label, minimum_pixels)
    elif backend == 'numpy':
        values, numbers = np.unique(np.column_stack([face, frame, label]), axis=0, return_counts=True)
        starts = np.r_[0, np.flatnonzero(np.any(values[1:,:2] != values[:-1,:2], axis=1))+1]
        lengths = np.diff(np.r_[starts, len(values)])
        maximum = np.maximum.reduceat(numbers, starts)
        possible = numbers == np.repeat(maximum, lengths)
        unique = np.add.reduceat(possible.astype(np.int64), starts) == 1
        keep = possible & np.repeat(unique & (maximum >= minimum_pixels), lengths)
        votes = values[keep][:,[0,2]]
        if not len(votes):
            return final, counts, ratios
        votes, numbers = np.unique(votes, axis=0, return_counts=True)
        starts = np.r_[0, np.flatnonzero(votes[1:,0] != votes[:-1,0])+1]
        lengths = np.diff(np.r_[starts, len(votes)])
        maximum = np.maximum.reduceat(numbers, starts)
        possible = numbers == np.repeat(maximum, lengths)
        ties = np.add.reduceat(possible.astype(np.int64), starts)
        first = np.minimum.reduceat(np.where(possible, np.arange(len(votes)), len(votes)), starts)
        ids, winners = votes[starts,0], votes[first,1]
        total = np.add.reduceat(numbers, starts)
    else:
        raise ValueError('aggregation_backend must be numpy or cuda')
    counts[ids] = total
    ratios[ids] = maximum / total
    accepted = (maximum >= minimum_frames) & (ratios[ids] >= agreement) & (ties == 1)
    final[ids[accepted]] = winners[accepted]
    return final, counts, ratios


def cuda_counts(face, frame, label, minimum_pixels):
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Aggregation requires exactly one leased GPU')
    labels = int(label.max())+1; frames = int(frame.max())+1
    if (int(face.max())+1)*frames*labels > np.iinfo(np.int64).max:
        raise ValueError('Vote IDs exceed lossless int64 packing')
    values = torch.as_tensor((face*frames+frame)*labels+label, device='cuda')
    values, numbers = torch.unique(values, sorted=True, return_counts=True)

    def group_winners(codes, numbers):
        groups, inv, lengths = torch.unique_consecutive(codes//labels, return_inverse=True, return_counts=True)
        maximum = torch.zeros_like(groups).scatter_reduce_(0, inv, numbers, reduce='amax', include_self=True)
        possible = numbers == maximum[inv]
        ties = torch.zeros_like(groups).scatter_add_(0, inv, possible.long())
        total = torch.zeros_like(groups).scatter_add_(0, inv, numbers)
        first = torch.full_like(groups, len(codes)).scatter_reduce_(0, inv,
            torch.where(possible, torch.arange(len(codes), device='cuda'), len(codes)), reduce='amin', include_self=True)
        return groups, codes[first]%labels, maximum, total, ties

    groups, winner, maximum, _, ties = group_winners(values, numbers)
    keep = (ties == 1) & (maximum >= minimum_pixels)
    votes = (groups[keep]//frames)*labels + winner[keep]
    if not len(votes):
        return tuple(np.empty(0, np.int64) for _ in range(5))
    values, numbers = torch.unique(votes, sorted=True, return_counts=True)
    return tuple(a.cpu().numpy() for a in group_winners(values, numbers))


def main():
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('--input', required=True); p.add_argument('--output', required=True)
    p.add_argument('--faces', type=int, required=True); p.add_argument('--minimum-frames', type=int, default=2)
    args = p.parse_args()
    with np.load(args.input) as data:
        result = {}
        for kind, unknown in [('classes',255), ('instances',0)]:
            for name, value in zip(['labels','counts','ratios'], aggregate(data['face'],data['frame'],data[kind],
                    args.faces, unknown, args.minimum_frames, backend='cuda')):
                result[kind+'_'+name] = value
    np.savez(args.output, **result)


if __name__ == '__main__':
    main()
