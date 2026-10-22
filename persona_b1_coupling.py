"""One-way drive -> emotion target convolution, never tick-sized delta.

Exact exponential convolution in an interior interval; adaptive convolution
at clipping boundaries. Numerical error estimate/budget are reported, not
misrepresented as a proof of exact integration at every clipping boundary.
"""
import math
class IntegrationBudgetError(RuntimeError):pass

def _raw(v,law,t):
    k=math.log(2)/law['half_life_hours'] if law['half_life_hours'] else 0
    g=law['growth_per_hour'];target=law['target']
    if not k:return v+g*t
    gain=-math.expm1(-k*t)
    return v+(target-v)*gain+g*t*(gain/(k*t) if t else 1)
def _rate(law):return math.log(2)/law['half_life_hours'] if law['half_life_hours'] else 0

def coupled(value,law,h,sources,coefficients,laws,freeze_sources=()):
    if value is None or any(sources.get(key) is None for key,c in coefficients.items() if c):
        return None,{'method':'unknown_input','error_estimate':None}
    if not h:return value,{'method':'zero_time','error_estimate':0}
    k=_rate(law)
    if not k:return value,{'method':'zero_decay','error_estimate':0}
    if law['growth_per_hour']!=0:raise ValueError('coupled emotion coordinate has no extra growth')
    if any('_transfer' in laws[key] for key,c in coefficients.items() if c):
        # A physical-timescale panel alone can still miss a narrow clipped
        # positive lobe. Enclose every source over each interval using its
        # known monotonic pieces and transfer peak, then isolate all target clipping boundaries.
        from persona_b1 import advance
        from persona_b1_layers import integrate_mean
        calls=0;boundary_error=0
        def integrand(t):
            target=law['target']+sum(c*advance(sources[key],laws[key],t,key in freeze_sources) for key,c in coefficients.items() if c)
            return k*math.exp(-k*(h-t))*min(1,max(0,target))
        def piece(a,b,tolerance,depth):
            nonlocal calls,boundary_error
            calls+=1
            if calls>20000:raise IntegrationBudgetError('transfer enclosure budget exhausted')
            lower=upper=law['target'];m=(a+b)/2
            for key,c in coefficients.items():
                if not c:continue
                r=laws[key];v=advance(sources[key],r,m,key in freeze_sources)
                if '_transfer' in r:
                    from persona_b1_conflict import grief_peak
                    tr=r['_transfer'];peak=grief_peak(sources[key],tr['anger'],tr['anger_rate'],tr['grief_rate'],tr['ratio'])
                    points=[a,b]
                    if peak is not None and a<peak<b:points.append(peak)
                    endpoints=[advance(sources[key],r,t,key in freeze_sources) for t in points]
                    lo=min(endpoints);hi=max(endpoints)
                else:
                    endpoints=[advance(sources[key],r,t,key in freeze_sources) for t in (a,b)]
                    lo=min(endpoints);hi=max(endpoints)
                lower+=min(c*lo,c*hi);upper+=max(c*lo,c*hi)
            weight=math.exp(-k*(h-b))*(-math.expm1(-k*(b-a)))
            if upper<=0:return 0.0
            if lower>=1:return weight
            if lower>=0 and upper<=1:
                # No clipping threshold can be hidden here. Smooth-source
                # quadrature retains its numerical (not rigorous) estimate.
                part=integrate_mean(integrand,a,b,tol=tolerance)
                if part is None:raise IntegrationBudgetError('unknown transfer trajectory')
                return part
            lo=max(0,min(1,lower));hi=max(0,min(1,upper));bound=weight*(hi-lo)/2
            if bound<=tolerance:
                boundary_error+=bound;return weight*(lo+hi)/2
            if not depth:raise IntegrationBudgetError('transfer clipping precision exhausted')
            return piece(a,m,tolerance/2,depth-1)+piece(m,b,tolerance/2,depth-1)
        total=piece(0,h,1e-10,80)
        return min(1,max(0,value*math.exp(-k*h)+total)),{'method':'enclosed_transfer_convolution','clipping_enclosure_bound':boundary_error,'requested_tolerance':1e-10,'intervals':calls,'smooth_error_is_numerical':True}
    if any('rest_ceiling' in laws[key] for key,c in coefficients.items() if c):
        from persona_b1_ceiling import equivalent_law,ceiling_advance
        from persona_b1 import advance
        cuts={0.0,h}
        for key,c in coefficients.items():
            r=laws[key];v=sources[key]
            if c and 'rest_ceiling' in r and key not in freeze_sources and v<r['rest_ceiling'] and r['growth_per_hour']>0:
                t=(r['rest_ceiling']-v)/r['growth_per_hour']
                if 0<t<h:cuts.add(t)
        result=value;ordered=sorted(cuts)
        for a,b in zip(ordered,ordered[1:]):
            local={key:advance(v,laws[key],a,key in freeze_sources) for key,v in sources.items()}
            simple={key:equivalent_law(local[key],r,key in freeze_sources) for key,r in laws.items()}
            result,_=coupled(result,law,b-a,local,coefficients,simple,freeze_sources)
        return result,{'method':'segmented_ceiling_convolution','error_estimate':None}
    active={key:dict(laws[key]) for key,c in coefficients.items() if c}
    for key in freeze_sources:
        if key in active and active[key]['growth_per_hour']-_rate(active[key])*(sources[key]-active[key]['target'])>0:
            active[key]={'half_life_hours':None,'target':0,'growth_per_hour':0}
    bounds=[];interior=True
    for key,r in active.items():
        x0=sources[key];xh=_raw(x0,r,h)
        if not 0<=xh<=1:interior=False
        endpoints=[coefficients[key]*x0,coefficients[key]*xh];bounds.append((min(endpoints),max(endpoints)))
    low=law['target']+sum(x[0] for x in bounds);high=law['target']+sum(x[1] for x in bounds)
    interior=interior and 0<=low<=high<=1
    expk=math.exp(-k*h);gain=-math.expm1(-k*h)
    if interior:
        y=value*expk+law['target']*gain
        for key,r in active.items():
            x0=sources[key];q=_rate(r);g=r['growth_per_hour'];c=coefficients[key]
            if not q:
                z=k*h
                lin=h*(z/2-z*z/6+z*z*z/24) if abs(z)<1e-5 else h-gain/k
                part=x0*gain+g*lin
            else:
                eq=r['target']+g/q;d=k-q
                if d==0:cross=k*h*expk
                elif abs(d*h)<1e-4:cross=k*expk*math.expm1(d*h)/d
                else:cross=k*(math.exp(-q*h)-expk)/d
                part=eq*gain+(x0-eq)*cross
            y+=c*part
        return min(1,max(0,y)),{'method':'analytic_convolution','error_estimate':0}
    # Interval enclosure isolates clipping boundaries. Endpoint bounds are
    # valid because each affine-exponential source is monotone. Unlike sparse
    # Simpson sampling, a hidden positive lobe cannot receive error=0.
    calls=0
    def integral_interior(start,stop,local_sources,local_laws):
        duration=stop-start;ek=math.exp(-k*duration);gk=-math.expm1(-k*duration)
        result=law['target']*gk
        for key,r in local_laws.items():
            x0=local_sources[key];q=_rate(r);g=r['growth_per_hour'];c=coefficients[key]
            if not q:
                z=k*duration
                lin=duration*(z/2-z*z/6+z*z*z/24) if abs(z)<1e-5 else duration-gk/k
                part=x0*gk+g*lin
            else:
                eq=r['target']+g/q;d=k-q
                if d==0:cross=k*duration*ek
                elif abs(d*duration)<1e-4:cross=k*ek*math.expm1(d*duration)/d
                else:cross=k*(math.exp(-q*duration)-ek)/d
                part=eq*gk+(x0-eq)*cross
            result+=c*part
        return math.exp(-k*(h-stop))*result
    def piece(start,stop,tolerance,depth):
        nonlocal calls
        calls+=1
        if calls>20000:raise IntegrationBudgetError('coupling enclosure budget exhausted')
        lower=upper=law['target'];local_sources={};local_laws={};simple=True
        for key,r in active.items():
            xa=_raw(sources[key],r,start);xb=_raw(sources[key],r,stop)
            ca=min(1,max(0,xa));cb=min(1,max(0,xb));c=coefficients[key]
            lower+=min(c*ca,c*cb);upper+=max(c*ca,c*cb)
            local_sources[key]=ca
            if max(xa,xb)<=0 or min(xa,xb)>=1:
                local_laws[key]={'half_life_hours':None,'target':0,'growth_per_hour':0}
            elif 0<=min(xa,xb)<=max(xa,xb)<=1:local_laws[key]=r
            else:simple=False
        weight=math.exp(-k*(h-stop))*(-math.expm1(-k*(stop-start)))
        if upper<=0:return 0.0,0.0
        if lower>=1:return weight,0.0
        if simple and lower>=0 and upper<=1:
            return integral_interior(start,stop,local_sources,local_laws),0.0
        low=max(0,min(1,lower));high=max(0,min(1,upper))
        bound=weight*(high-low)/2
        if bound<=tolerance:return weight*(low+high)/2,bound
        if not depth:raise IntegrationBudgetError('coupling enclosure precision exhausted')
        middle=(start+stop)/2
        left,le=piece(start,middle,tolerance/2,depth-1);right,re=piece(middle,stop,tolerance/2,depth-1)
        return left+right,le+re
    integral,bound=piece(0,h,1e-10,80)
    return min(1,max(0,value*expk+integral)),{'method':'bounded_convolution','error_estimate':bound,'absolute_enclosure_bound':bound,'evaluations':calls}

def reflected_coordinate(value,law,h,sources,coefficients,laws,freeze_sources=(),extra_drift=0):
    """Bounded scalar ODE for explicit unresolved-grudge drift.
    Adaptive step doubling with a physical-timescale step ceiling. The local
    estimate is numerical, not a proof of all-policy global floating error.
    """
    from persona_b1 import advance
    if value is None or any(sources.get(key) is None for key,c in coefficients.items() if c):return None,{'method':'unknown_drift_input'}
    rates=[_rate(law)]
    for key,c in coefficients.items():
        if c:
            rates.append(_rate(laws[key]));tr=laws[key].get('_transfer')
            if tr:rates.extend([tr['anger_rate'],tr['grief_rate']])
    maximum=min(1.0,.1/max(rates)) if max(rates)>0 else 1.0
    k=_rate(law);calls=0
    def derivative(t,y):
        target=law['target']+sum(c*advance(sources[key],laws[key],t,key in freeze_sources) for key,c in coefficients.items() if c)
        d=k*(min(1,max(0,target))-y)+law['growth_per_hour']+extra_drift
        return max(0,d) if y<=0 else min(0,d) if y>=1 else d
    def step(t,y,dt):
        a=derivative(t,y);b=derivative(t+dt/2,y+dt*a/2);c=derivative(t+dt/2,y+dt*b/2);d=derivative(t+dt,y+dt*c)
        return min(1,max(0,y+dt*(a+2*b+2*c+d)/6))
    t=0;y=value;dt=min(h,maximum);error=0
    while t<h:
        calls+=1
        if calls>100000:raise IntegrationBudgetError('grudge coordinate work budget exhausted')
        dt=min(dt,h-t);whole=step(t,y,dt);half=step(t+dt/2,step(t,y,dt/2),dt/2);estimate=abs(half-whole)/15
        tolerance=1e-10*dt/max(h,1)
        if estimate<=tolerance:
            y=half;t+=dt;error+=estimate;dt=min(maximum,dt*2)
        else:
            dt/=2
            if dt<1e-12:raise IntegrationBudgetError('grudge coordinate precision exhausted')
    return y,{'method':'timescale_reflected_drift','local_error_sum':error,'steps':calls}
