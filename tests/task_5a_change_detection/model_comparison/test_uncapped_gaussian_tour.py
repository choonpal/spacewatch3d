"""The selected K1-K21 tour reuses the existing detections, not local inference."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tools/gaussian_change'
saved={name:sys.modules.pop(name,None) for name in ['prepare','prepare_keyframes','build_tour_candidates','link_segmented_objects','build_change_regions']}
sys.path.insert(0,str(TOOLS))
try:
    spec=importlib.util.spec_from_file_location('uncapped_tour_validation',TOOLS/'validate_uncapped_tour.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
finally:
    sys.path.pop(0)
    for name,previous in saved.items():
        sys.modules.pop(name,None)
        if previous is not None:sys.modules[name]=previous


class UncappedTour(unittest.TestCase):
    def test_selected_route_and_existing_detections_survive_publication(self):
        folder=ROOT/'src/spacewatch3d/tasks/task_5a_change_detection/assets/panorama-change-20261025/gaussian-uncapped-tour-20261026'
        report=validator.validate(folder,metadata_only=True)
        self.assertEqual(report['keyframes'],21);self.assertEqual(report['stored_links'],20)
        self.assertEqual(report['connected_segments'],1);self.assertEqual(report['original_filtered_candidates'],5)
        self.assertFalse(report['new_simsac_inference'])
        read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
        data=read('gaussian-data.json');source=read('route-map-source.json')
        by_source={v['viewpoint_id']:v for v in source['viewpoints']}
        for view in data['zones']:
            original=by_source[view['viewpoint_id']]
            self.assertEqual(view['id'],original['label'])
            np.testing.assert_allclose(np.asarray(view['source_reference_from_camera'])[:3,3],original['position'],atol=1e-10)
        fixed=read('tour-candidates.json')
        for mode in ['raw','filtered']:
            original=read('simsac-candidates.json')['boxes'][mode]
            self.assertEqual(fixed['boxes'][mode],original)


if __name__=='__main__':unittest.main()
