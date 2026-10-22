"""Gateway rollout gate, not an authority grant or a new Persona writer.

Authority requires the engine to verify a persisted profile fence and
candidate-store binding. No config bool or claimed coverage token grants it.
"""
from dataclasses import dataclass


class B1WriterConfigurationError(ValueError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class B1WriterGate:
    mode: str = 'off'
    scopes: tuple = ()

    def binding(self, profile_id):
        return next((row for row in self.scopes if row[0] == profile_id), None)

    def verify_engine(self, engine, profile_id):
        if self.mode != 'candidate_authority':
            return False
        engine_profile = getattr(engine, 'profile_id', None)
        if engine is not None and engine_profile != profile_id and (self.binding(profile_id) or self.binding(engine_profile)):
            raise B1WriterConfigurationError('b1_writer_profile_mismatch')
        if self.binding(profile_id) is None:
            return False
        _, scope, epoch = self.binding(profile_id)
        if engine is None or getattr(engine, 'profile_id', None) != profile_id:
            raise B1WriterConfigurationError('b1_writer_candidate_engine_required')
        verify = getattr(engine, 'verify_b1_authority', None)
        if not callable(verify):
            raise B1WriterConfigurationError('b1_writer_coverage_incomplete')
        # The engine checks persisted fencing in the actual legacy DB and the
        # candidate store binding. Config values are expectations, not proof.
        record = verify(scope, epoch)
        if not isinstance(record, dict) or record.get('profile_id') != profile_id or record.get('scope') != scope or type(record.get('epoch')) is not int or record['epoch'] != epoch or record.get('mode') != 'candidate':
            raise B1WriterConfigurationError('b1_writer_authority_receipt_invalid')
        return True


def parse_writer_gate(config):
    if config is None:
        return B1WriterGate()
    if not isinstance(config, dict) or set(config) - {'mode', 'scopes'}:
        raise B1WriterConfigurationError('b1_writer_invalid_config')
    mode = config.get('mode', 'off')
    if not isinstance(mode, str) or mode not in {'off', 'shadow', 'candidate_authority'}:
        raise B1WriterConfigurationError('b1_writer_invalid_mode')
    scopes = config.get('scopes', [])
    if not isinstance(scopes, list):
        raise B1WriterConfigurationError('b1_writer_invalid_scopes')
    bindings = []
    profiles = set()
    destinations = set()
    for scope in scopes:
        if not isinstance(scope, dict) or set(scope) != {'profile_id', 'scope', 'epoch'}:
            raise B1WriterConfigurationError('b1_writer_invalid_binding')
        profile, destination, epoch = scope['profile_id'], scope['scope'], scope['epoch']
        if any(not isinstance(v, str) or not v or v != v.strip() or len(v) > 256 or any(ord(c) < 32 for c in v) for v in (profile, destination)):
            raise B1WriterConfigurationError('b1_writer_invalid_identity')
        if type(epoch) is not int or epoch < 1:
            raise B1WriterConfigurationError('b1_writer_invalid_epoch')
        if profile in profiles or destination in destinations:
            raise B1WriterConfigurationError('b1_writer_ambiguous_binding')
        profiles.add(profile); destinations.add(destination)
        bindings.append((profile, destination, epoch))
    if mode == 'candidate_authority' and not bindings:
        raise B1WriterConfigurationError('b1_writer_binding_required')
    return B1WriterGate(mode, tuple(bindings))
