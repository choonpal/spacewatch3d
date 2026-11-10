"""Small contract tests; synthetic fixtures, no model/mesh/point-cloud execution."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from spacewatch_pipeline import paper_changes as changes
from spacewatch_pipeline.io import digest, read, record, save, sha


def config():
    return dict(captures=[dict(id='alpha'), dict(id='beta')],
                runtime=dict(cpu_threads=1), changes=dict(keypoint_nearby=True, native_rgb=True))


def registration():
    identity = np.eye(4).tolist()
    return dict(state='passed_geometric_gate',
        common_from_world=[identity, identity], region_matrices=[[identity, identity], [identity, identity]],
        fresh_structural=dict(height_u=4., regions=[dict(supported=True)]))


class Regional:
    def assign(self, points, epoch):
        return np.ones(len(points), dtype=int)


def frame(fid, x=0., stamp=0., mask=0., bystander=0, labels=None):
    pose = np.eye(4); pose[0, 3] = x
    return dict(id=str(fid), common_position=[x, 0., 0.], world_from_camera=pose.tolist(),
        label_counts=labels or {'box': 50}, timestamp_s=stamp, masked_fraction=mask,
        bystander_pixels=bystander, sharpness=1., source_frame_index=int(stamp*30))


def importance_dependencies(root, cfg):
    means = {'ceiling light': 4, 'water dispenser': 4, 'sign': 2}
    names = sorted(means)
    deps = {}
    cfg['requested_inputs'] = dict(object_classes=names)
    for cid in ['alpha', 'beta']:
        ratings_path, cache_path = root/(cid+'-ratings.json'), root/(cid+'-cache.json')
        ratings = dict(mode='cached_response_reconstruction', current_model_calls=0, repeats=5,
            model='fixture-cached-model', manual_hash=digest(['current fixture manual']),
            class_list_hash=digest(names), class_names=names,
            classes={name:dict(n=5, ratings=[value]*5, mean=value, score=value/5) for name,value in means.items()})
        save(ratings_path, ratings)
        save(cache_path, dict(capture_id=cid, rating_source=record(ratings_path),
            manual_hash=ratings['manual_hash'], model_requested=ratings['model'],
            scores={cid+'-current-object-'+name:dict(class_name=name, class_mean=value, score=value/5)
                    for name,value in means.items()}))
        deps[cid+'.importance'] = dict(ratings=str(ratings_path), cache=str(cache_path), class_list_hash=digest(names))
    return deps


def prepared_image_fixture(root):
    """Tiny byte receipts, not actual arrays/images and never decoded."""
    inputs=root/'current-inputs';inputs.mkdir();prepared=root/'prepared';(prepared/'views').mkdir(parents=True)
    manifest=dict(schema='mesh-simsac-input-v1');views=[]
    for epoch,(phase,capture) in enumerate([('before','alpha'),('after','beta')]):
        image=inputs/phase/'rgb'/'native.png';image.parent.mkdir(parents=True)
        image.write_bytes(('source RGB file fixture '+capture).encode())
        frame=dict(id='current-frame-'+capture,source_frame_index=epoch*30,
            image=image.relative_to(inputs).as_posix(),sha256=sha(image))
        manifest[phase]=dict(id=capture,frames=[frame])
        vid='E'+str(epoch)+'-V00000'
        for suffix in ['.npz','.jpg']:(prepared/'views'/(vid+suffix)).write_bytes(('unchanged raster '+vid+suffix).encode())
        views.append(dict(id=vid,epoch=epoch,capture_id=capture,frame_id=frame['id'],
            source_frame_index=frame['source_frame_index'],source_image=frame['image'],
            geometry_sha256=sha(prepared/'views'/(vid+'.npz')),rgb_sha256=digest('unchanged projected RGB '+vid),
            K=np.eye(3).tolist(),pose=np.eye(4).tolist()))
    save(inputs/'manifest.json',manifest)
    save(prepared/'manifest.json',dict(schema='mesh-simsac-prepared-v1',views=views,source_hashes={},settings=dict(fusion='mean')))
    return inputs/'manifest.json',prepared


class PaperChangeTests(unittest.TestCase):
    def test_paper_options_dynamic_captures_and_fixed_method(self):
        changes.validate_options(config())
        bad = config(); bad['changes']['settings'] = dict(fusion='geo')
        with self.assertRaises(ValueError): changes.validate_options(bad)
        bad = config(); bad['changes']['keypoint_nearby'] = dict(registration='old')
        with self.assertRaises(ValueError): changes.validate_options(bad)

    def test_lowmask_spatial_gate_mask_and_person_not_object(self):
        rows = [frame('near', stamp=0., labels={'person':10000, 'box':50}),
                frame('rich', stamp=2., mask=.04, labels={'box':500, 'clothes':500}),
                frame('far', x=9., stamp=3., labels={'box':99999})]
        selected = changes.select_lowmask(rows, [0, 0, 0], 1., count=1)
        self.assertEqual(selected[0]['id'], 'near')
        self.assertEqual(selected[0]['label_type_count'], 1)
        self.assertNotIn('person', selected[0]['visible_labels'])

    def test_lowmask_prefers_separate_timestamps_before_filling_budget(self):
        rows = [frame('best', stamp=0.), frame('close', stamp=.1), frame('spread', stamp=2.)]
        self.assertEqual([row['id'] for row in changes.select_lowmask(rows, [0,0,0], 1., count=2)],
                         ['best', 'spread'])

    def test_extended_mask_retains_person_outside_camera_fixed_prior(self):
        mask = np.zeros((40, 60), bool); person = mask.copy(); prior = mask.copy()
        person[35, 30] = True; prior[35, 30] = True
        person[5, 30] = True
        extended = changes.extend_photographer(mask, person, prior)
        self.assertTrue(extended[35, 30]); self.assertFalse(extended[5, 30])

    def test_current_plan_arbitrary_count_and_ids_no_fixed_fifteen(self):
        records = [[frame('not-an-index', stamp=0.), frame('other-id', stamp=2.)] for _ in range(2)]
        views = [dict(viewpoint_id='selected-'+str(i), reference_from_camera=np.eye(4).tolist()) for i in range(2)]
        plans = [dict(capture_id='alpha', viewpoints=[]), dict(capture_id='beta', viewpoints=views)]
        plan = changes.select_current_plan(records, plans, registration(), 'beta', 'fixture', Regional())
        self.assertEqual(plan['anchor_count'], 2)
        self.assertEqual([row['viewpoint_id'] for row in plan['keypoints']], ['selected-0','selected-1'])
        self.assertEqual(plan['selected_frames'][0][0]['keypoint_ids'], ['beta-K01','beta-K02'])

    def test_missing_current_plan_dependencies_fail_before_export(self):
        with self.assertRaises(ValueError):
            changes._require_plan_dependencies(dict(config=config(), dependencies={}))

    def test_crop_binding_uses_current_five_response_means_and_capture_identities(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg = config(); deps = importance_dependencies(root, cfg)
            manifest = {phase:dict(segmentation_labels=['wall','person','ceiling light','water dispenser','sign'])
                        for phase in ['before','after']}
            mapping, binding = changes.current_crop_importance(dict(config=cfg, dependencies=deps), manifest)
            self.assertEqual(mapping, {'ceiling light':4., 'water dispenser':4., 'sign':2.})
            self.assertFalse(binding['legacy_static_map_used'])
            self.assertEqual(binding['semantic_status'], 'UNVERIFIED')
            self.assertEqual(binding['new_model_calls'], 0)
            ratings = Path(deps['beta.importance']['ratings']); value = read(ratings)
            value['classes']['sign']['mean'] = 3; save(ratings,value)
            with self.assertRaises(ValueError):
                changes.current_crop_importance(dict(config=cfg, dependencies=deps), manifest)

    def test_prepare_clones_input_folder_binds_mean_and_preserves_imported_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root/'immutable-inputs'; source.mkdir()
            out = root/'new-prepare'; out.mkdir(); cfg = config(); deps = importance_dependencies(root,cfg)
            fixture = source/'asset.npz'; fixture.write_bytes(b'no mesh loaded; hardlinked test input')
            document = dict(settings=dict(fusion='mean'))
            for phase in ['before','after']:
                document[phase] = dict(segmentation_labels=['wall','person','ceiling light','water dispenser','sign'])
            source_manifest = source/'manifest.json'; save(source_manifest,document); original = sha(source_manifest)
            plan = root/'current-plan.json'; save(plan,dict(current=True))
            deps['pair.rgb_inputs'] = dict(manifest=str(source_manifest), keypoint_plan=str(plan),
                fresh_registration=True, registration_computed_in_current_pipeline=False,
                stage_reuse=dict(source_run='prior-execution'))
            def prepare(job):
                manifest = Path(job['dependencies']['pair.rgb_inputs']['manifest'])
                self.assertEqual(manifest.parent, out/'current-inputs')
                self.assertNotEqual(manifest.stat().st_ino, source_manifest.stat().st_ino)
                self.assertEqual((manifest.parent/'asset.npz').stat().st_ino, fixture.stat().st_ino)
                self.assertEqual(read(manifest)['settings']['crop_importance_by_class']['ceiling light'],4.)
                self.assertEqual(sha(source_manifest),original)
                save(out/'result.json',dict(root=str(out/'detection')))
            with patch('spacewatch_pipeline.mesh_rgb.rgb_prepare',prepare):
                changes.rgb_prepare(dict(config=cfg, dependencies=deps, output=str(out)))
            value = read(out/'result.json')
            self.assertEqual(sha(source_manifest),original)
            self.assertNotIn('crop_importance_by_class',read(source_manifest)['settings'])
            self.assertEqual(value['inputs_manifest'],str(out/'current-inputs/manifest.json'))
            self.assertFalse(value['registration_computed_in_current_pipeline'])
            self.assertEqual(len(read(value['crop_importance_binding'])['source_files']),4)

    def test_infer_propagates_new_manifest_and_registration_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            prepared = dict(inputs_manifest='new-immutable-manifest.json', keypoint_plan='current-plan.json',
                root='original-prepared-root',crop_importance_binding='current-binding.json', registration_computed_in_current_pipeline=False)
            def infer(job):
                self.assertEqual(job['dependencies']['pair.rgb_prepare']['root'],'private-bound-root')
                save(out/'result.json',dict(pairs=7))
            binding=dict(root='private-bound-root',source_image_binding='source-binding.json',bound_views=2,bound_original_frames=2)
            with patch.object(changes,'bind_prepared_source_images',return_value=binding), \
                 patch('spacewatch_pipeline.mesh_rgb.rgb_infer',infer):
                changes.rgb_infer(dict(output=str(out),dependencies={'pair.rgb_prepare':prepared}))
            value = read(out/'result.json')
            for key,item in prepared.items():
                if key!='root':self.assertEqual(value[key],item)
            self.assertEqual(value['source_prepared_root'],prepared['root'])
            self.assertEqual(value['source_image_binding'],'source-binding.json')

    def test_prepared_erp_binding_private_clone_preserves_raster_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs,prepared=prepared_image_fixture(root)
            source_sha=sha(prepared/'manifest.json');input_sha=sha(inputs)
            original=read(prepared/'manifest.json')
            binding=changes.bind_prepared_source_images(prepared,inputs,root/'new-infer/prepared-binding')
            destination=Path(binding['root']);current=read(destination/'manifest.json')
            self.assertEqual(binding['bound_views'],2);self.assertEqual(binding['bound_original_frames'],2)
            self.assertEqual(sha(prepared/'manifest.json'),source_sha);self.assertEqual(sha(inputs),input_sha)
            for old,new in zip(original['views'],current['views']):
                self.assertTrue(Path(new['source_image']).is_absolute())
                self.assertEqual(sha(new['source_image']),new['source_image_sha256'])
                for key in old:
                    if key!='source_image':self.assertEqual(old[key],new[key])
                for suffix in ['.npz','.jpg']:
                    old_file=prepared/'views'/(old['id']+suffix);new_file=destination/'views'/(new['id']+suffix)
                    self.assertEqual(sha(old_file),sha(new_file));self.assertTrue(old_file.samefile(new_file))
            receipt=read(binding['source_image_binding'])
            self.assertEqual(receipt['hardlinked_raster_assets'],4)
            self.assertFalse(receipt['raster_arrays_opened']);self.assertEqual(receipt['model_calls'],0)
            with self.assertRaises(ValueError):changes.bind_prepared_source_images(prepared,inputs,destination)

    def test_prepared_erp_binding_rejects_input_path_escape_before_clone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);inputs,prepared=prepared_image_fixture(root)
            manifest=read(inputs);manifest['before']['frames'][0]['image']='../escaped.png'
            (root/'escaped.png').write_bytes(b'not a declared bundle source');save(inputs,manifest)
            before=sha(prepared/'manifest.json');destination=root/'private-bound'
            with self.assertRaises(ValueError):changes.bind_prepared_source_images(prepared,inputs,destination)
            self.assertFalse(destination.exists());self.assertEqual(sha(prepared/'manifest.json'),before)

    def test_prepared_erp_binding_rejects_checksum_capture_frame_and_path_mismatches(self):
        cases=['checksum','capture','frame','source-index','declared-image']
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary);inputs,prepared=prepared_image_fixture(root)
                if case=='checksum':
                    manifest=read(inputs);manifest['before']['frames'][0]['sha256']='0'*64;save(inputs,manifest)
                else:
                    manifest=read(prepared/'manifest.json');view=manifest['views'][0]
                    if case=='capture':view['capture_id']='unrelated-capture'
                    if case=='frame':view['frame_id']='unrelated-frame'
                    if case=='source-index':view['source_frame_index']=999
                    if case=='declared-image':view['source_image']='../outside.png'
                    save(prepared/'manifest.json',manifest)
                before=sha(prepared/'manifest.json');destination=root/'private-bound'
                with self.assertRaises(ValueError):changes.bind_prepared_source_images(prepared,inputs,destination)
                self.assertFalse(destination.exists());self.assertEqual(sha(prepared/'manifest.json'),before)

    def test_current_input_stage_contract_fits_before_selecting_and_decodes_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); out = root/'stage'; out.mkdir()
            cfg = config(); deps = {}
            for cid in ['alpha', 'beta']:
                p = root/(cid+'-plan.json')
                save(p, dict(capture_id=cid, viewpoints=[dict(viewpoint_id=cid+'-current',
                    reference_from_camera=np.eye(4).tolist())]))
                deps[cid+'.plan'] = dict(plan=str(p))
            calls = []
            def export(job):
                folder = Path(job['output'])/'inputs'; folder.mkdir(parents=True)
                document = dict(schema='mesh-simsac-input-v1')
                save(Path(job['output'])/'adapter.json', dict(captures=cfg['captures']))
                for side, cap in zip(['before','after'], ['alpha','beta']):
                    for sub in ['rgb','segmentation','person']:
                        (folder/side/sub).mkdir(parents=True, exist_ok=True)
                    for name in ['mesh.npz','points.npz']:
                        (folder/side/name).write_bytes(b'synthetic-unused-geometry-fixture')
                    frames = []
                    for i in range(4):
                        rgb = folder/side/'rgb'/f'{i}.png'; Image.fromarray(np.zeros((40,80,3), 'u1')).save(rgb)
                        cls = np.ones((40,80), 'u2'); cls[10:20,10:20] = 2
                        person = np.zeros(cls.shape, bool)
                        segmentation = folder/side/'segmentation'/f'{i}.npz'
                        np.savez_compressed(segmentation, class_id=cls, person=person, instance_id=np.ones(cls.shape, 'u4'))
                        mask = folder/side/'person'/f'{i}.png'; Image.fromarray(np.zeros(cls.shape,'u1')).save(mask)
                        frames.append(dict(id=str(i), timestamp_s=float(i), source_frame_index=i*30,
                            world_from_camera=np.eye(4).tolist(), image=str(rgb.relative_to(folder)), sha256=sha(rgb),
                            segmentation=str(segmentation.relative_to(folder)), person_mask=str(mask.relative_to(folder))))
                    document[side] = dict(id=cap, mesh=side+'/mesh.npz', pointcloud=side+'/points.npz',
                        segmentation_labels=['wall','box','person'], frames=frames)
                save(folder/'manifest.json', document)
                calls.append('export')
            def fit(manifest, output, **options):
                self.assertTrue(options['registration_only'])
                self.assertEqual(len(read(manifest)['after']['frames']), 4)
                Path(output).mkdir(); save(Path(output)/'registration.json', registration())
                calls.append('fresh-registration'); return registration()
            original = changes.select_current_plan
            def select(*args, **kwargs):
                calls.append('current-plan-selection')
                return original(*args, regional=Regional(), **kwargs)
            def native(folder, adapter, runtime):
                self.assertEqual(len(read(Path(folder)/'manifest.json')['after']['frames']), 3)
                calls.append('native-selected-rgb')
            with patch('spacewatch_pipeline.mesh_rgb.rgb_inputs', export), \
                 patch('tools.mesh_simsac.prepare.prepare', fit), \
                 patch.object(changes, 'select_current_plan', select), \
                 patch('spacewatch_pipeline.mesh_rgb.materialize_native_frames', native):
                changes.rgb_inputs(dict(config=cfg, dependencies=deps, output=str(out)))
            value = read(out/'result.json'); manifest = read(value['manifest'])
            self.assertEqual(calls, ['export','fresh-registration','current-plan-selection','native-selected-rgb'])
            self.assertEqual(manifest['keypoint_plan']['anchor_count'], 1)
            self.assertTrue(manifest['frozen_structural_registration']['computed_in_current_stage'])
            self.assertEqual(manifest['settings']['fusion'], 'mean')
            self.assertEqual(read(value['keypoint_plan'])['source_plan_sha256']['beta'], sha(deps['beta.plan']['plan']))

    def test_finalize_exposes_actual_general_grouping_and_current_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary); plan = out/'current-plan.json'; save(plan, dict(current=True))
            root = out/'detection'; root.mkdir(); (root/'results').mkdir()
            events = [dict(id='one', label='box', rgb_only=True, bounds=None, center=[0,0,0], radius_u=.5,
                           before='object-a', after=None),
                      dict(id='two', label='box', rgb_only=True, bounds=None, center=[.1,0,0], radius_u=.5,
                           before=None, after='object-b')]
            def final(job):
                self.assertEqual(job['dependencies']['pair.rgb_inputs']['manifest'],'strict-current-inputs.json')
                save(root/'viewer-data.json', dict(events=events, counts={}, policy=dict(registration_error_u=.1),
                     job_provenance=dict(fresh_inference=True)))
                save(out/'result.json', dict(viewer_dir=str(root)))
            job = dict(output=str(out), dependencies={
                'pair.rgb_inputs':dict(manifest='old-reference-inputs.json', keypoint_plan=str(plan),
                    fresh_registration=True, registration_computed_in_current_pipeline=False,
                    stage_reuse=dict(source_run='prior-execution')),
                'pair.rgb_infer':dict(inputs_manifest='strict-current-inputs.json',
                    registration_computed_in_current_pipeline=False, crop_importance_binding='strict-binding.json')})
            with patch('spacewatch_pipeline.mesh_rgb.rgb_finalize', final):
                changes.rgb_finalize(job)
            value = read(out/'result.json')
            self.assertEqual(value['verified_change_count'], 0)
            self.assertTrue(Path(value['verified_changes']).is_file())
            self.assertEqual(value['inputs_manifest'], 'strict-current-inputs.json')
            self.assertFalse(value['registration_computed_in_current_pipeline'])
            self.assertFalse(read(value['viewer_data'])['job_provenance']['fresh_registration_in_current_pipeline'])
            self.assertFalse(read(value['viewer_data'])['job_provenance']['detector_event_replay'])


class ObjectCropBindingTests(unittest.TestCase):
    def test_legacy_reference_unchanged_and_strict_all_provided_classes_eligible(self):
        from tools.mesh_simsac.objcrop import IMPORTANCE,crop_importance
        labels = ['wall','person','ceiling light','water dispenser','sign','new fixture class']
        self.assertEqual(crop_importance({},labels),IMPORTANCE)
        self.assertNotIn('ceiling light',crop_importance({},labels))
        self.assertEqual(crop_importance({},labels)['water dispenser'],3)
        self.assertEqual(crop_importance({},labels)['sign'],3)
        supplied = {'ceiling light':4.,'water dispenser':4.,'sign':2.,'new fixture class':3.6}
        self.assertEqual(crop_importance(dict(crop_importance_by_class=supplied),labels),supplied)

    def test_strict_invalid_class_or_mean_fails_no_neutral_fallback(self):
        from tools.mesh_simsac.objcrop import crop_importance
        labels = ['wall','person','ceiling light']
        for supplied in ({}, {'wall':3}, {'person':3}, {'unknown class':3}, {'ceiling light':0},
                         {'ceiling light':6}, {'ceiling light':float('nan')},
                         {'ceiling light':float('inf')}, {'ceiling light':True}, {'ceiling light':'4'}):
            with self.subTest(supplied=supplied), self.assertRaises(ValueError):
                crop_importance(dict(crop_importance_by_class=supplied),labels)

    def test_actual_candidate_path_includes_light_only_in_strict_mode(self):
        from tools.mesh_simsac.objcrop import plan_crops
        class Scene:
            def __init__(self,**kwargs):pass
            def add_triangles(self,*args):pass
            def cast_rays(self,*args):return {'t_hit':SimpleNamespace(numpy=lambda:np.array([2.]))}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); sides=[]
            for epoch in [0,1]:
                classes=np.ones((64,128),'u2');classes[28:36,60:68]=2
                segmentation=root/f'fixture-{epoch}.npz'
                np.savez_compressed(segmentation,class_id=classes,person=np.zeros(classes.shape,bool))
                sides.append(dict(epoch=epoch,vertices_world=np.array([[0.,0,0],[1,0,0],[0,1,0]]),
                    faces=np.array([[0,1,2]]),spec=dict(segmentation_labels=['wall','ceiling light','person'],
                    frames=[dict(id='fixture-'+str(epoch),keypoint_ids=['current-keypoint'],
                        world_from_camera=np.eye(4).tolist(),segmentation=segmentation.name)])))
            settings=dict(visibility_tolerance_u=.15)
            with patch('tools.mesh_simsac.objcrop.o3d.t.geometry.RaycastingScene',Scene):
                legacy=plan_crops(sides,[np.eye(4),np.eye(4)],root,settings)
                strict=plan_crops(sides,[np.eye(4),np.eye(4)],root,
                    dict(settings,crop_importance_by_class={'ceiling light':4.}))
            self.assertEqual(legacy['report']['targets'],[])
            self.assertEqual(len(strict['report']['targets']),2)
            self.assertEqual({row['label'] for row in strict['report']['targets']},{'ceiling light'})
            self.assertEqual(strict['report']['settings']['importance_source'],'current_class_mean_binding')


if __name__ == '__main__':
    unittest.main()
