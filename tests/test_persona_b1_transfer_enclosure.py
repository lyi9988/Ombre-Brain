import unittest,math
from persona_b1_coupling import coupled
from persona_b1 import advance
class TransferEnclosure(unittest.TestCase):
 def test_narrow_clipped_transfer_lobe_not_missed(self):
  law={'half_life_hours':math.log(2),'target':.00005,'growth_per_hour':0}
  laws={'grieve':{'half_life_hours':math.log(2)/2,'target':0,'growth_per_hour':0,'_transfer':{'anger':1,'anger_rate':1,'grief_rate':2,'ratio':1}},'favored':{'half_life_hours':None,'target':0,'growth_per_hour':0}}
  source={'grieve':0,'favored':.5};coeff={'grieve':1,'favored':-.5}
  whole,meta=coupled(.5,law,1,source,coeff,laws);y=.5
  for i in range(100):
   local={key:advance(v,laws[key],i*.01) for key,v in source.items()};current={key:dict(v) for key,v in laws.items()};current['grieve']['_transfer']={**laws['grieve']['_transfer'],'anger':math.exp(-i*.01)}
   y,_=coupled(y,law,.01,local,coeff,current)
  self.assertAlmostEqual(whole,y,places=9);self.assertGreater(whole,.18394);self.assertGreater(meta['intervals'],5)
