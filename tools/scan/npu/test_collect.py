"""stdlib 测试；真实网络 tracer 用 NPU_LIVE=1 单独开启。"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch
import urllib.error
import hashlib
import unittest

HERE = Path(__file__).resolve().parent
SCRATCH = Path('/root/.hermes/profiles/third/cache/scratch/npu-collector-tests')


class ValidationTests(unittest.TestCase):
    def test_page_count_and_unique_ids_are_checked(self):
        import collect
        start, end = '2026-10-07T00:00:00Z', '2026-10-14T00:00:00Z'
        item = dict(number='IABC', title='精度错误', state='open', body='NaN',
                    created_at=start, updated_at=start)
        for data, headers, page, seen in [
            ([item], {'total_count': '2', 'total_page': '1'}, 1, set()),
            ([item, item], {'total_count': '2', 'total_page': '1'}, 1, set()),
            ([item], {'total_count': '1', 'total_page': '2'}, 1, set()),
        ]:
            with self.subTest(data=data, headers=headers), self.assertRaises(collect.FetchError):
                collect.validate_page(data, headers, 'gitcode', page, seen, start, end)

    def test_complete_two_page_walk_and_github_link_integrity(self):
        import collect
        start, end = '2026-10-07T00:00:00Z', '2026-10-14T00:00:00Z'
        items = [dict(number=str(n), updated_at=start) for n in range(102)]
        seen = set()
        headers = dict(total_count='102', total_page='2')
        self.assertTrue(collect.validate_page(items[:100], headers, 'gitcode', 1, seen, start, end))
        self.assertFalse(collect.validate_page(items[100:], headers, 'gitcode', 2, seen, start, end))
        self.assertEqual(len(seen), 102)
        with self.assertRaises(collect.FetchError):
            collect.validate_page(items[:100], {'link': '<https://api.github.com/repos/a/b/issues?page=2>; rel="last"'},
                                  'github', 1, set(), start, end)
        with self.assertRaises(collect.FetchError):
            collect.validate_page(items[:100], {'link': '<https://api.github.com/repos/a/b/issues?page=9>; rel="next"'},
                                  'github', 1, set(), start, end)

    def test_github_cursor_next_link_is_followed(self):
        import collect
        source = dict(platform='github', repo='example/npu', web_url='https://github.com/example/npu',
                      scopes=['inference'])
        start, end = '2026-10-07T00:00:00Z', '2026-10-14T00:00:00Z'
        first = [dict(number=str(i), updated_at='2026-10-08T00:00:00Z') for i in range(100)]
        final = [dict(number='100', updated_at='2026-10-09T00:00:00Z')]
        cursor_url = ('https://api.github.com/repositories/42/issues?per_page=100'
                      '&page=2&after=opaque-cursor%3D')
        calls = []
        def controlled(url, raw):
            calls.append(url)
            if len(calls) == 1:
                return first, {'link': '<' + cursor_url + '>; rel="next"'}
            self.assertEqual(url, cursor_url, 'must retain GitHub after/before cursor from Link')
            return final, {}
        with patch('collect.request', side_effect=controlled):
            rows = list(collect.fetch_list(source, 'issues', start, end, SCRATCH, {}))
        self.assertEqual(len(rows), 101)
        self.assertEqual(len(calls), 2)

    def test_github_cursor_links_cannot_leak_auth_to_other_origins(self):
        import collect
        for url in ('https://evil.example/issues?page=2&after=x',
                    'http://api.github.com/repositories/42/issues?page=2&after=x',
                    'https://api.github.com@evil.example/issues?page=2&after=x'):
            with self.subTest(url=url), self.assertRaises(collect.FetchError):
                collect.github_next({'link': '<' + url + '>; rel="next"'})

    def test_time_filter_and_unexpected_json_are_errors(self):
        import collect
        start, end = '2026-10-07T00:00:00Z', '2026-10-14T00:00:00Z'
        for data in [{'message': 'token required'}, [dict(number='1', updated_at='2020-01-01T00:00:00Z')]]:
            with self.subTest(data=data), self.assertRaises(collect.FetchError):
                collect.validate_page(data, {'total_count': '1', 'total_page': '1'}, 'gitcode', 1, set(), start, end)

    def test_falsy_nontext_body_and_broken_link_are_errors(self):
        import collect
        source = dict(platform='github', repo='a/b', web_url='https://github.com/a/b', scopes=['inference'])
        item = dict(number='1', title='t', state='open', created_at='2026-10-07T00:00:00Z', updated_at='2026-10-08T00:00:00Z')
        for body in (0, False, [], {}):
            with self.subTest(body=body), self.assertRaises(collect.FetchError):
                collect.normalize(source, dict(item, body=body), 'issue', SCRATCH)
        with self.assertRaises(collect.FetchError):
            collect.validate_page([item], {'link': 'broken-link-header'}, 'github', 1, set(),
                                  '2026-10-07T00:00:00Z', '2026-10-14T00:00:00Z')

    def test_string_identity_kind_and_canonical_domain(self):
        import collect
        source = dict(platform='gitcode', repo='mindspore/mindspore',
                      web_url='https://atomgit.com/mindspore/mindspore', scopes=['training'])
        item = dict(number='I001AB', title='title', body='body', state='open',
                    created_at='2026-10-07T00:00:00Z', updated_at='2026-10-08T00:00:00Z')
        issue = collect.normalize(source, item, 'issue', SCRATCH)
        pr = collect.normalize(source, item, 'pr', SCRATCH)
        self.assertEqual(issue['number'], 'I001AB')
        self.assertNotEqual(issue['key'], pr['key'])
        self.assertEqual(issue['url'], source['web_url'] + '/issues/I001AB')
        self.assertEqual(pr['url'], source['web_url'] + '/merge_requests/I001AB')
        self.assertTrue(issue['content_digest'])


class StateTests(unittest.TestCase):
    def setUp(self):
        import collect
        self.collect = collect
        SCRATCH.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=SCRATCH)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.registry = self.root / 'sources.json'
        data = json.loads((HERE / 'sources.json').read_text())
        data['sources'] = [next(s for s in data['sources'] if s['repo'] == 'Ascend/msprobe')]
        collect.dump(self.registry, data)
        self.source = data['sources'][0]
        self.now = datetime(2026, 10, 14, tzinfo=timezone.utc)
        self.batch = self.root / 'seed'
        self.batch.mkdir()
        row = collect.normalize(self.source, dict(number='001', title='NaN', body='body', state='open',
            created_at='2026-10-07T00:00:00Z', updated_at='2026-10-08T00:00:00Z'), 'issue', self.batch)
        Path(row['body_path']).parent.mkdir()
        Path(row['body_path']).write_text('body')
        for name in ('normalized', 'inference', 'training'):
            (self.batch / (name + '.jsonl')).write_text(json.dumps(row) + '\n')
        collect.dump(self.batch / 'manifest.json', dict(complete=True, generated_at=collect.stamp(self.now),
            registry_hash=hashlib.sha256(self.registry.read_bytes()).hexdigest(),
            coverage=[dict(platform='gitcode', repo='Ascend/msprobe', complete=True)], failures=[],
            counts=dict(inference=1, training=1), registry_sources=1, selected_sources=1))
        self.state = dict(batch=str(self.batch), watermarks={'gitcode:Ascend/msprobe': collect.stamp(self.now)})
        collect.dump(self.root / 'state.json', self.state)

    def test_initial_window_uses_utc_calendar_cutoff(self):
        output = self.root / 'fresh-calendar-window'
        with patch('collect.request', side_effect=TimeoutError('controlled timeout')) as fetch:
            result = self.collect.run('inference', None, output, registry=self.registry,
                                      now=self.now + timedelta(hours=12, minutes=30))
        self.assertFalse(result['complete'])
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(fetch.call_args.args[0]).query)
        self.assertEqual(query['updated_after'], ['2026-10-07T00:00:00Z'])

    def test_scope_cache_reuse_and_registry_hash_invalidation(self):
        first = self.collect.run('inference', None, self.root, registry=self.registry, now=self.now)
        second = self.collect.run('training', None, self.root, registry=self.registry, now=self.now)
        self.assertTrue(first['cached'] and second['cached'])
        self.assertEqual(first['batch'], second['batch'])
        self.registry.write_text(self.registry.read_text() + '\n')
        error = urllib.error.HTTPError('https://api.gitcode.com/x', 503, 'controlled error', {}, None)
        with patch('collect.request', side_effect=error):
            changed = self.collect.run('training', None, self.root, registry=self.registry, now=self.now)
        self.assertFalse(changed['cached'])
        self.assertFalse(changed['complete'])
        self.assertEqual(changed['candidate_count'], 1)

    def test_failure_does_not_advance_watermark_or_erase_candidates(self):
        error = TimeoutError('controlled timeout')
        for turn in range(2):
            with patch('collect.request', side_effect=error) as fetch:
                result = self.collect.run('training', None, self.root, force=(turn == 0),
                                          registry=self.registry, now=self.now + timedelta(hours=4))
            self.assertFalse(result['complete'])
            self.assertFalse(result['cached'])
            self.assertEqual(result['failures'][0]['error'], 'TimeoutError')
            self.assertEqual(result['failures'][0]['page'], 1)
            self.assertEqual(result['candidate_count'], 1)
            self.assertEqual(json.loads((self.root / 'state.json').read_text())['watermarks'], self.state['watermarks'])
            query = urllib.parse.parse_qs(urllib.parse.urlsplit(fetch.call_args.args[0]).query)
            self.assertEqual(query['updated_after'], ['2026-10-13T23:00:00Z'])

    def test_only_run_cannot_touch_full_watermarks(self):
        with patch('collect.request', side_effect=TimeoutError('controlled')):
            result = self.collect.run('inference', 'Ascend/msprobe', self.root,
                                      registry=self.registry, now=self.now)
        self.assertNotEqual(Path(result['batch']).parent, self.root)
        self.assertEqual(json.loads((self.root / 'state.json').read_text()), self.state)


class DetailAndNetworkTests(unittest.TestCase):
    def test_url_requires_full_identity_and_preserves_legacy_string_id(self):
        import collect
        registry = json.loads((HERE / 'sources.json').read_text())['sources']
        source, kind, number = collect.parse_url('https://gitcode.com/mindspore/mindspore/issues/007', registry)
        self.assertEqual(source['web_url'], 'https://atomgit.com/mindspore/mindspore')
        self.assertEqual(number, '007')
        source, kind, number = collect.parse_url('https://gitee.com/ascend/pytorch/issues/IABC01', registry)
        self.assertEqual((source['platform'], source['repo'], kind, number), ('gitee', 'ascend/pytorch', 'issue', 'IABC01'))
        for url in ('123', 'https://github.com/evil/r/issues/1?token=x', 'https://github.com.evil/r/r/issues/1'):
            with self.subTest(url=url), self.assertRaises(collect.FetchError):
                collect.parse_url(url, registry)

    def test_github_token_never_sent_to_gitcode_or_gitee(self):
        import collect
        with patch.dict(os.environ, {'GITHUB_TOKEN': '', 'TOKEN': 'ghp_controlled-test-only'}):
            self.assertIn('Authorization', collect.auth_headers('https://api.github.com/repos/a/b/issues'))
            for url in ('https://api.gitcode.com/api/v5/repos/a/b/issues', 'https://gitee.com/api/v5/repos/a/b/issues'):
                self.assertNotIn('Authorization', collect.auth_headers(url))
        with patch.dict(os.environ, {'GITHUB_TOKEN': '', 'TOKEN': 'unrelated'}):
            self.assertNotIn('Authorization', collect.auth_headers('https://api.github.com/repos/a/b/issues'))

    def test_invalid_json_and_http_error_keep_raw_without_cookies(self):
        import collect
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from http.client import IncompleteRead
        from threading import Thread
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(503 if self.path == '/error' else 200)
                self.send_header('Set-Cookie', 'DO-NOT-PERSIST')
                if self.path == '/truncated':
                    self.send_header('Content-Length', '999')
                self.end_headers()
                self.wfile.write(b'{malformed')
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        SCRATCH.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
                for tail, error in [('/invalid', json.JSONDecodeError), ('/error', urllib.error.HTTPError), ('/truncated', IncompleteRead)]:
                    raw = Path(directory) / tail.lstrip('/')
                    with self.assertRaises(error):
                        collect.request('http://127.0.0.1:' + str(server.server_port) + tail, raw)
                    self.assertEqual(raw.with_suffix('.body').read_bytes(), b'{malformed')
                    self.assertNotIn('DO-NOT-PERSIST', raw.with_suffix('.json').read_text())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


class LiveTracer(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_gitcode_single_repository(self):
        result = subprocess.run(
            ['/usr/bin/python3', str(HERE / 'collect.py'), '--scope', 'inference',
             '--only', 'Ascend/msprobe', '--output-dir', str(SCRATCH / 'tracer'), '--force'],
            capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summary = json.loads(result.stdout)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['coverage'][0]['repo'], 'Ascend/msprobe')
        self.assertTrue(Path(summary['candidates']).is_file())
        rows = [json.loads(s) for s in Path(summary['candidates']).read_text().splitlines()]
        self.assertGreater(len(rows), 0)
        self.assertEqual({r['type'] for r in rows}, {'issue', 'pr'})
        self.assertTrue(all(Path(r['body_path']).is_file() for r in rows))
    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_github_collection_and_scope_cache(self):
        summaries = []
        for scope in ('inference', 'training'):
            command = ['/usr/bin/python3', str(HERE / 'collect.py'), '--scope', scope,
                       '--only', 'sgl-project/sgl-kernel-npu', '--output-dir', str(SCRATCH / 'github-live')]
            if scope == 'inference':
                command.append('--force')
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            summaries.append(json.loads(result.stdout))
        self.assertEqual(summaries[0]['batch'], summaries[1]['batch'])
        self.assertTrue(summaries[1]['cached'])
        self.assertGreater(summaries[0]['candidate_count'], 0)
        rows = [json.loads(line) for line in Path(summaries[0]['candidates']).read_text().splitlines()]
        self.assertIn('pr', {r['type'] for r in rows})
        state = json.loads((Path(summaries[0]['batch']).parent / 'state.json').read_text())
        self.assertEqual(state['watermarks']['github:sgl-project/sgl-kernel-npu'], summaries[0]['generated_at'])

    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_detail_gitcode_body_and_comments(self):
        result = subprocess.run(['/usr/bin/python3', str(HERE / 'collect.py'), 'detail',
            'https://gitcode.com/Ascend/msprobe/issues/620', '--output-dir', str(SCRATCH / 'detail-live')],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['number'], '620')
        self.assertEqual(summary['comment_count'], 2)
        self.assertTrue(Path(summary['body_path']).is_file())
        self.assertEqual(len(Path(summary['comments']).read_text().splitlines()), 2)
    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_failed_pull_endpoint_does_not_advance_repository(self):
        import collect
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            original = collect.request
            def controlled(url, raw):
                if '/pulls?' in url:
                    raise TimeoutError('controlled pulls-only error')
                return original(url, raw)
            with patch('collect.request', side_effect=controlled):
                failed = collect.run('inference', 'Ascend/msprobe', Path(directory))
            state_path = Path(failed['batch']).parent / 'state.json'
            self.assertFalse(failed['complete'])
            self.assertGreater(failed['candidate_count'], 0)
            self.assertNotIn('gitcode:Ascend/msprobe', json.loads(state_path.read_text())['watermarks'])
            recovered = collect.run('training', 'Ascend/msprobe', Path(directory))
            self.assertTrue(recovered['complete'])
            self.assertFalse(recovered['cached'])
            self.assertGreater(recovered['candidate_count'], failed['candidate_count'])
            self.assertIn('gitcode:Ascend/msprobe', json.loads(state_path.read_text())['watermarks'])

    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_concurrent_scopes_publish_one_shared_batch(self):
        SCRATCH.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=SCRATCH) as directory:
            command = ['/usr/bin/python3', str(HERE / 'collect.py'), '--only', 'Ascend/msprobe',
                       '--output-dir', directory]
            jobs = [subprocess.Popen(command + ['--scope', scope], stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True) for scope in ('inference', 'training')]
            results = []
            for job in jobs:
                out, err = job.communicate(timeout=120)
                self.assertEqual(job.returncode, 0, err + out)
                results.append(json.loads(out))
            self.assertEqual(results[0]['batch'], results[1]['batch'])
            self.assertEqual({r['cached'] for r in results}, {True, False})
            self.assertFalse(list(Path(results[0]['batch']).parent.glob('.building-*')))

    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_github_issue_url_for_pr_is_rejected(self):
        import collect
        result = collect.detail('https://github.com/sgl-project/sgl-kernel-npu/issues/840',
                                SCRATCH / 'wrong-pr-route', HERE / 'sources.json')
        self.assertFalse(result['complete'], 'PR through /issues must not claim issue completeness')
        self.assertIsNone(result['body_path'])
        self.assertEqual(result['comment_count'], 0)
        self.assertEqual(result['failures'][0]['error'], 'FetchError')
        self.assertIn('/pull/', result['failures'][0]['message'])

    @unittest.skipUnless(os.environ.get('NPU_LIVE') == '1', '显式启用真实网络测试')
    def test_gitcode_pr_and_github_pr_details(self):
        import collect
        for url in ('https://gitcode.com/Ascend/msprobe/merge_requests/1082',
                    'https://github.com/sgl-project/sgl-kernel-npu/pull/840'):
            with self.subTest(url=url):
                result = collect.detail(url, SCRATCH / 'pr-details', HERE / 'sources.json')
                self.assertTrue(result['complete'], result)
                comments = [json.loads(line) for line in Path(result['comments']).read_text().splitlines()]
                self.assertEqual(len(comments), result['comment_count'])
                self.assertTrue(all(isinstance(c['id'], str) for c in comments))


if __name__ == '__main__':
    unittest.main()
