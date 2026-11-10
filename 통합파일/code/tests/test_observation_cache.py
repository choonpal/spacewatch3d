"""Cache integrity/invalidation checks without importing inference workers."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1]/'vendor/panovggt_pipeline_v2/workers/observation_cache.py'
spec = importlib.util.spec_from_file_location('observation_cache', SOURCE)
cache_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_module)


class PairCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.context = {'geometry': {'depths': 'depth-hash', 'valid': 'mask-hash'}, 'registration': {'seed': 7}, 'code': 'code-hash'}
        self.key = {'file': 'pair_001.npz', 'source': 'matches', 'sha256': 'match-hash'}
        self.result = {'train': [[0.0, -0.0, 1.2345678901234567]], 'audit': {'accepted': False, 'reason': 'Too few image matches'}}
        self.cache = cache_module.PairObservationCache(self.temp.name, self.context)

    def test_round_trip_new_instance_preserves_rows_and_float_bits(self):
        self.cache.put(self.key, self.result)
        other = cache_module.PairObservationCache(self.temp.name, self.context)
        actual = other.get(self.key)
        self.assertEqual(actual, self.result)
        self.assertEqual([x.hex() for x in actual['train'][0]], [x.hex() for x in self.result['train'][0]])
        self.assertEqual(other.statistics()['hits'], 1)

    def test_geometry_settings_and_code_changes_invalidate(self):
        self.cache.put(self.key, self.result)
        for field in self.context:
            changed = dict(self.context, **{field: 'changed'})
            other = cache_module.PairObservationCache(self.temp.name, changed)
            self.assertIsNone(other.get(self.key), field)

    def test_source_priority_filename_and_content_changes_invalidate(self):
        self.cache.put(self.key, self.result)
        for field in self.key:
            self.assertIsNone(self.cache.get(dict(self.key, **{field: 'changed'})), field)

    def test_partial_write_is_not_read(self):
        path = self.cache.directory/(cache_module.signature(self.key)+'.partial')
        path.write_text('{')
        self.assertIsNone(self.cache.get(self.key))
        self.cache.put(self.key, self.result)
        self.assertEqual(self.cache.get(self.key), self.result)

    def test_corruption_recomputes_instead_of_reusing_changed_rows(self):
        self.cache.put(self.key, self.result)
        path = self.cache.directory/(cache_module.signature(self.key)+'.json')
        data = json.loads(path.read_text())
        data['result']['train'][0][2] = 99.0
        path.write_text(json.dumps(data))
        self.assertIsNone(self.cache.get(self.key))
        for bad in ['{', '[]', 'null', '{"result":null}']:
            path.write_text(bad)
            self.assertIsNone(self.cache.get(self.key))

    def test_rejected_pairs_are_reusable_and_return_fresh_objects(self):
        self.cache.put(self.key, self.result)
        first = self.cache.get(self.key)
        first['audit']['accepted'] = True
        self.assertFalse(self.cache.get(self.key)['audit']['accepted'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
