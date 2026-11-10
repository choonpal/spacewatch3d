"""Regression checks for scoring without allocating display geometry."""
from dataclasses import replace
import unittest
import numpy as np
from verify_v51_routing import Route, Config, connect_tour, without_geometry


class RoutingScoresTests(unittest.TestCase):
    def test_exact_paths_lengths_failures_and_final_geometry(self):
        route = Route(origin=np.zeros(2), grid_m=1, walkable=np.ones((1, 5)),
                      clearance=np.ones((1, 5)), cells=[(0, i) for i in range(5)],
                      positions=np.array([[i, 0, 1] for i in range(5)], dtype=float),
                      components=[1, 1, 1, 1, 2],
                      adjacency={0: [(1, 1.)], 1: [(0, 1.), (2, 1.)],
                                 2: [(1, 1.), (3, 1.)], 3: [(2, 1.)], 4: []},
                      candidates=list(range(5)), obstacles=None)
        cases = [(None, set()), (1., set()), (1., {1, 2})]
        for limit, banned in cases:
            config = replace(Config(), max_link_m=limit)
            full = connect_tour(route, [0, 3, 4], config, banned)
            fast = connect_tour(route, [0, 3, 4], config, banned, include_geometry=False)
            self.assertEqual(without_geometry(full), fast)
            self.assertTrue(all('polyline_m' in link and 'route_node_ids' in link for link in full[1]))
            self.assertEqual(sum(link['length_m'] for link in fast[1]), 3.)
            self.assertIn('disconnected_tour', [f['code'] for f in fast[3]])
            if banned:
                self.assertIn('transition_gap', [f['code'] for f in fast[3]])
            elif limit is not None:
                self.assertEqual(fast[2], [1, 2])


if __name__ == '__main__':
    unittest.main()
