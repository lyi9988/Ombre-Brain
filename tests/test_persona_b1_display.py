import unittest
from datetime import timedelta
from persona_b1_layers import display_choice
from persona_b1 import policy
from test_persona_b1_layers import layers_policy
from test_persona_b1 import NOW
class Display(unittest.TestCase):
 def rules(self):return {'mode':'primary','lit_min_valence':.5,'fresh_minutes':30,'fresh_lit_priority':4,'aged_lit_priority':.5,'cause_max_valence':.55,'phase_max_valence':.48,'phase_priority':2}
 def state(self):return {'valence':.52,'lit':{'word':'心动','at':NOW.isoformat(),'until':(NOW+timedelta(hours=2)).isoformat()},'cause_label':{'label':'hurt','priority':3,'since':NOW.isoformat()},'connection_phase':None}
 def test_fresh_then_aged_priority_keeps_both_inner_feelings(self):
  a=self.state();self.assertEqual(display_choice(a,NOW,self.rules())['primary'],'心动');self.assertEqual(display_choice(a,NOW+timedelta(minutes=31),self.rules())['primary'],'hurt');self.assertEqual(a['lit']['word'],'心动')
 def test_unknown_not_calm_and_default_not_forced_threshold(self):
  a=self.state();a['valence']=None;r=display_choice(a,NOW,self.rules());self.assertIsNone(r['primary']);self.assertEqual(r['status'],'partial_unknown');self.assertIn('心动',r['secondary'])
  r=display_choice(a,NOW,{});self.assertEqual(r['mode'],'coexist');self.assertEqual(r['secondary'],['心动','hurt'])
 def test_explicit_configuration_required_and_no_intimacy_clamp(self):
  p=layers_policy();p['display_policy']={'mode':'primary'}
  with self.assertRaises(ValueError):policy(p)
  p['display_policy']=self.rules();self.assertEqual(policy(p)['display_policy'],self.rules())
  a=self.state();a['valence']=1;self.assertEqual(display_choice(a,NOW,self.rules())['primary'],'心动');self.assertEqual(a['valence'],1)
