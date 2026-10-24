"""Internal authenticated candidate ingress. No global config or anonymous route.

Registration reuses server._authorized_memory_write (same existing internal
bearer). A missing explicit candidate adapter returns 503, never a legacy write.
"""
import json,sqlite3
from starlette.responses import JSONResponse
from persona_b1 import Conflict,ModulationSourceUnavailable
from persona_projection import safe_persona_projection
MAX_BODY=128*1024


def _source_unavailable(exc):
    return JSONResponse({'error':'b1_modulation_source_unavailable','code':exc.code,
        'source_kind':exc.source_kind,'coordinate':exc.coordinate,
        'interval_start':exc.interval_start,'interval_end':exc.interval_end},status_code=503)

def register_candidate_route(mcp,engine_getter,authorized_internal):
    @mcp.custom_route('/internal/persona/b1/candidate',methods=['POST'])
    async def ingest(request):
        if not authorized_internal(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        engine=engine_getter()
        if getattr(engine,'b1_candidate',None) is None:return JSONResponse({'error':'b1_candidate_not_configured'},status_code=503)
        chunks=[];size=0
        async for part in request.stream():
            size+=len(part)
            if size>MAX_BODY:return JSONResponse({'error':'request_too_large'},status_code=413)
            chunks.append(part)
        try:
            data=json.loads(b''.join(chunks))
            if not isinstance(data,dict) or set(data)!={'envelope','turn'}:raise ValueError('complete request required')
            turn=data['turn']
            if not isinstance(turn,dict):raise ValueError('turn required')
            result=await engine.update_from_exchange(
                turn.get('conversation_id'),turn.get('user_event',{}).get('content'),turn.get('assistant_event',{}).get('content'),
                b1_event=data['envelope'],b1_turn=turn)
            result_receipt=result['b1_receipt'];receipt=result_receipt['receipt']
            # Source hash is for the trusted transport acknowledgement only;
            # public/owner projection is the separate allowlist below.
            return JSONResponse({'status':'candidate_applied','replayed':result_receipt['replayed'],
                'scope':receipt['scope'],'event_id':receipt['event_id'],'source_hash':receipt['source_hash'],
                'revision':receipt['after_revision'],'b1_receipt':{'receipt':{k:receipt[k] for k in ('scope','event_id','source_hash','occurred_at','after_revision','status')},'replayed':result_receipt['replayed']},'projection':safe_persona_projection(result)})
        except ModulationSourceUnavailable as exc:return _source_unavailable(exc)
        except Conflict:return JSONResponse({'error':'b1_identity_or_revision_conflict'},status_code=409)
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'invalid_b1_candidate_input'},status_code=400)
        except sqlite3.OperationalError:return JSONResponse({'error':'b1_candidate_storage_unavailable'},status_code=503)
        except Exception:return JSONResponse({'error':'b1_candidate_failed'},status_code=500)
    @mcp.custom_route('/internal/persona/b1/phase',methods=['POST'])
    async def phase(request):
        if not authorized_internal(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        engine=engine_getter();adapter=getattr(engine,'b1_candidate',None)
        if adapter is None:return JSONResponse({'error':'b1_candidate_not_configured'},status_code=503)
        chunks=[];size=0
        async for part in request.stream():
            size+=len(part)
            if size>MAX_BODY:return JSONResponse({'error':'request_too_large'},status_code=413)
            chunks.append(part)
        try:
            data=json.loads(b''.join(chunks))
            if not isinstance(data,dict) or set(data)!={'scope','phase_id','occurred_at','policy_revision','phase','source_kind'}:raise ValueError('explicit lifecycle envelope required')
            if data['scope']!=adapter.scope:raise ValueError('phase scope mismatch')
            if data['source_kind'] not in ('internal_owner_request','internal_rest_plan'):raise ValueError('internal lifecycle, not physical sleep claim')
            result=adapter.set_phase(data['phase_id'],data['occurred_at'],data['policy_revision'],data['phase'],data['source_kind']);r=result['b1_receipt']
            return JSONResponse({'status':'candidate_applied','physical_sleep_claim':False,'b1_receipt':{'replayed':r['replayed'],'receipt':{k:r['receipt'][k] for k in ('scope','event_id','source_hash','occurred_at','after_revision','status')}},'projection':safe_persona_projection(result)})
        except ModulationSourceUnavailable as exc:return _source_unavailable(exc)
        except Conflict:return JSONResponse({'error':'b1_identity_or_revision_conflict'},status_code=409)
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'invalid_b1_phase_input'},status_code=400)
        except sqlite3.OperationalError:return JSONResponse({'error':'b1_candidate_storage_unavailable'},status_code=503)
        except Exception:return JSONResponse({'error':'b1_candidate_failed'},status_code=500)
    @mcp.custom_route('/internal/persona/b1/numeric',methods=['GET'])
    async def numeric(request):
        if not authorized_internal(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        engine=engine_getter();adapter=getattr(engine,'b1_candidate',None)
        if adapter is None:return JSONResponse({'error':'b1_candidate_not_configured'},status_code=503)
        try:
            if set(request.query_params)-{'scope','at'} or request.query_params.get('scope')!=adapter.scope:raise ValueError('numeric scope required')
            value=adapter.store.numeric_at(adapter.scope,request.query_params.get('at'))
            return JSONResponse({'snapshot':value,'status':'available' if value is not None else 'no_historical_snapshot'})
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'invalid_b1_numeric_input'},status_code=400)
        except sqlite3.OperationalError:return JSONResponse({'error':'b1_candidate_storage_unavailable'},status_code=503)
    @mcp.custom_route('/internal/persona/b1/clock',methods=['POST'])
    async def clock(request):
        if not authorized_internal(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        engine=engine_getter()
        if getattr(engine,'b1_candidate',None) is None:return JSONResponse({'error':'b1_candidate_not_configured'},status_code=503)
        chunks=[];size=0
        async for part in request.stream():
            size+=len(part)
            if size>MAX_BODY:return JSONResponse({'error':'request_too_large'},status_code=413)
            chunks.append(part)
        try:
            from persona_b1 import at,digest
            data=json.loads(b''.join(chunks))
            if not isinstance(data,dict) or set(data)-{'pressure_frames'}!={'scope','occurred_at','policy_revision','growth_frames'}:raise ValueError('explicit clock frame required')
            if data['scope']!=engine.b1_candidate.scope:raise ValueError('clock scope mismatch')
            when=at(data['occurred_at'])
            event_id='http:'+digest({'namespace':'brain-b1-clock','scope':data['scope'],'at':when})
            result=engine.advance_b1_clock(event_id,when,data['policy_revision'],data['growth_frames'],data.get('pressure_frames'))
            receipt=result['b1_receipt']
            return JSONResponse({'status':'candidate_applied','b1_receipt':{'replayed':receipt['replayed'],'receipt':{key:receipt['receipt'][key] for key in ('scope','event_id','source_hash','occurred_at','after_revision','status')}},'projection':safe_persona_projection(result)})
        except ModulationSourceUnavailable as exc:return _source_unavailable(exc)
        except Conflict:return JSONResponse({'error':'b1_identity_or_revision_conflict'},status_code=409)
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'invalid_b1_clock_input'},status_code=400)
        except sqlite3.OperationalError:return JSONResponse({'error':'b1_candidate_storage_unavailable'},status_code=503)
        except Exception:return JSONResponse({'error':'b1_candidate_failed'},status_code=500)
    @mcp.custom_route('/internal/persona/b1/result',methods=['POST'])
    async def operation_result(request):
        if not authorized_internal(request):return JSONResponse({'error':'unauthorized'},status_code=401)
        engine=engine_getter()
        if getattr(engine,'b1_candidate',None) is None:return JSONResponse({'error':'b1_candidate_not_configured'},status_code=503)
        chunks=[];size=0
        async for part in request.stream():
            size+=len(part)
            if size>MAX_BODY:return JSONResponse({'error':'request_too_large'},status_code=413)
            chunks.append(part)
        try:
            data=json.loads(b''.join(chunks))
            if not isinstance(data,dict) or set(data)!={'result','policy_revision'}:raise ValueError('canonical result required')
            result=engine.b1_candidate._record_verified_operation(data['result'],data['policy_revision'])
            receipt=result['b1_receipt']
            return JSONResponse({'status':'candidate_applied','b1_receipt':{'replayed':receipt['replayed'],'receipt':{key:receipt['receipt'][key] for key in ('scope','event_id','source_hash','occurred_at','after_revision','status')}},'projection':safe_persona_projection(result)})
        except ModulationSourceUnavailable as exc:return _source_unavailable(exc)
        except Conflict:return JSONResponse({'error':'b1_identity_or_revision_conflict'},status_code=409)
        except (ValueError,TypeError,KeyError,OverflowError):return JSONResponse({'error':'invalid_b1_result_input'},status_code=400)
        except sqlite3.OperationalError:return JSONResponse({'error':'b1_candidate_storage_unavailable'},status_code=503)
        except Exception:return JSONResponse({'error':'b1_candidate_failed'},status_code=500)
    return ingest
