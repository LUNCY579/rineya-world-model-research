import collections
import random
import unittest
from generators.critic.generate import world_parent,candidates_for
from generators.critic.claim_controls import augment
from factory.common import canonical,split_for
from oracles.independent import solve


class ClaimControls(unittest.TestCase):
    def test_signed_and_location_controls_in_every_new_world(self):
        for index in range(180,192):
            rng=random.Random(20261009+31*index)
            world,_=world_parent(index,split_for(index),rng)
            candidates=augment(world,candidates_for(world,rng),rng)
            evaluations=solve({'kind':'critic','world':world,'candidates':candidates})['evaluations']
            by={e['candidate_id']:e for e in evaluations}
            signs=collections.defaultdict(set);locations=set()
            for candidate in candidates:
                plan=canonical(candidate['plan'])
                actual=solve({**world,'plan':candidate['plan']})['after']
                if candidate['claimed_after']['node']==actual['node']:
                    delta=candidate['claimed_after']['fuel']-actual['fuel']
                    signs[plan].add((delta>0)-(delta<0))
                elif candidate['claimed_after']['fuel']==actual['fuel']:locations.add(plan)
                if not by[candidate['candidate_id']]['claim_correct']:
                    self.assertTrue(by[candidate['candidate_id']]['valid'] or candidate['plan'])
            for plan in [[],[world['nodes'][1]]]:
                self.assertEqual(signs[canonical(plan)],{-1,0,1})
                self.assertIn(canonical(plan),locations)


if __name__=='__main__':unittest.main()
