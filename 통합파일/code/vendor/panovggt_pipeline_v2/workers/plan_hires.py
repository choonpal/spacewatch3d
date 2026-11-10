from runtime import R, C, read, save

groups = read(R/'observation_summary.json')['components']
labels = {f: i for i, group in enumerate(groups) for f in group}
pairs = [[a, a+d] for a in range(C['frames']) for d in C['matching']['high_resolution']['gaps']
         if a+d < C['frames'] and labels[a] != labels[a+d]]
save(R/'hires_pairs.json', pairs)
