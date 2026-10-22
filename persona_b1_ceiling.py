"""Continuous rest ceiling: event peaks above the ceiling are not clipped."""
import math

def ceiling_advance(value,hours,ceiling,growth_per_hour,relax_per_hour,freeze_growth=False):
    for name,x in [('hours',hours),('ceiling',ceiling),('growth',growth_per_hour),('relax',relax_per_hour)]:
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x):raise ValueError('finite '+name+' required')
    if hours<0 or not 0<=ceiling<=1 or growth_per_hour<0 or relax_per_hour<0:raise ValueError('nonnegative ceiling law required')
    if value is None:return None
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:raise ValueError('bounded value required')
    if value>ceiling:return ceiling+(value-ceiling)*math.exp(-relax_per_hour*hours)
    return value if freeze_growth else min(ceiling,value+growth_per_hour*hours)

def equivalent_law(value,law,freeze_growth=False):
    if 'rest_ceiling' not in law:return dict(law)
    if value is not None and value>law['rest_ceiling']:
        q=law['relax_per_hour']
        return {'half_life_hours':math.log(2)/q if q else None,'target':law['rest_ceiling'],'growth_per_hour':0}
    return {'half_life_hours':None,'target':0,'growth_per_hour':0 if freeze_growth or value==law['rest_ceiling'] else law['growth_per_hour']}
