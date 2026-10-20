"""Credential receipts prove in-memory equality, never secret existence."""
from types import SimpleNamespace
import unittest
from credential_activation import adoption_receipt


class CredentialActivationTest(unittest.TestCase):
    def test_present_but_old_consumer_cannot_ack_new_key(self):
        service = SimpleNamespace(reranker_engine=SimpleNamespace(api_key='TEST_ONLY_old'))
        result = adoption_receipt(service, {
            '_credential_generation': 'a' * 32, 'reranker': {'api_key': 'TEST_ONLY_new'}})
        self.assertEqual(result['confirmed_fields'], [])
        self.assertFalse(result['provider_verified'])
        self.assertNotIn('TEST_ONLY', str(result))

    def test_exact_consumer_matches_return_only_opaque_generation(self):
        service = SimpleNamespace(reranker_engine=SimpleNamespace(api_key='TEST_ONLY_new'))
        result = adoption_receipt(service, {
            '_credential_generation': 'b' * 32, 'reranker': {'api_key': 'TEST_ONLY_new'}})
        self.assertEqual(result['confirmed_fields'], ['reranker.api_key'])
        self.assertEqual(result['generation'], 'b' * 32)
        self.assertNotIn('TEST_ONLY', str(result))

    def test_upstream_receipt_checks_primary_order_and_backup_membership(self):
        primary, backup = 'TEST_ONLY_primary', 'TEST_ONLY_backup'
        service = SimpleNamespace(upstreams=[dict(name='fixture',
            api_keys=[dict(value=backup), dict(value=primary)])])
        body = dict(_credential_generation='c' * 32, gateway=dict(upstreams=[
            dict(name='fixture', credential_primary_env='PRIMARY',
                 api_keys=[dict(api_key=primary, label='env:PRIMARY'),
                           dict(api_key=backup, label='owner-backup')])]))
        self.assertNotIn('gateway.upstreams', adoption_receipt(service, body)['confirmed_fields'])
        service.upstreams[0]['api_keys'].reverse()
        self.assertEqual(adoption_receipt(service, body)['confirmed_fields'], ['gateway.upstreams'])

    def test_legacy_request_has_no_invented_generation(self):
        self.assertIsNone(adoption_receipt(SimpleNamespace(), {'reranker': {'api_key': 'TEST_ONLY'}}))
