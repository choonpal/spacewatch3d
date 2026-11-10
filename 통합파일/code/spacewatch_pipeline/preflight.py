"""Read-only input contracts, executed in the geometry runtime by doctor."""
from .io import read,sha
from .geometry import similarity
from .provided import checked_arrays
from .size_instances import assumptions,canonical_labels

def inputs(config):
    _,spec=assumptions(config)
    for c in config['captures']:
        similarity(read(c['coordinates'])['source_to_mesh'])
        if config['segmentation']['source']=='provided':
            if c['semantic_frame_id']!=c['id']+'_mesh' or sha(c['coordinates'])!=c['semantic_coordinates_sha256']:
                raise ValueError('Provided mesh coordinate contract mismatch: '+c['id'])
            catalog=read(config['segmentation']['vocabulary'])
            if [r['id'] for r in catalog]!=list(range(len(catalog))) or len(catalog)>255:raise ValueError('Vocabulary IDs must be contiguous 0..254')
            data=checked_arrays(c['semantic_arrays'])
            if c.get('planning_profile'):
                from .target_profile import checked_profile
                checked_profile(config,c,data)
            if not set(data['face_class_id']).issubset({r['id'] for r in catalog}|{255}):raise ValueError('Missing class in vocabulary')
            canonical_labels(data['face_class_id'],catalog,spec['priors'],config['size_segmentation'].get('class_aliases',{}))
