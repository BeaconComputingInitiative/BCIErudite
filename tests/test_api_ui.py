# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import contextlib
import io
import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from unittest.mock import patch
import main
from core.inference.engine import Engine
from core.inference.tokenizer import BPETokenizer
from core.inference.model import ModelConfig

class EngineContracts(unittest.TestCase):
    def test_incremental_utf8_stream_and_exact_budget(self):
        class Model:
            config=ModelConfig(262,width=32,layers=1,heads=4,context=32)
            def generate(self,*_):yield from [226,130,172]
        engine=Engine({'max_output_tokens':4,'device':'cpu','temperature':0,'top_p':1})
        engine.model=Model();engine.tokenizer=BPETokenizer();engine.metadata={'system_prompt':'BCI'};engine.manifest={'step':1}
        with patch.object(engine,'ensure_loaded'):
            events=list(engine.stream([{'role':'user','content':'Hi'}]))
            self.assertEqual(''.join(e.get('text','') for e in events),'€')
            with self.assertRaises(ValueError):list(engine.stream([{'role':'user','content':'x'*100}]))
    def test_cancelled_generation_is_explicit(self):
        class Model:
            config=ModelConfig(262,width=32,layers=1,heads=4,context=32)
            def generate(self,*_):
                yield 65
                engine.cancel.set();yield 66
        engine=Engine({'max_output_tokens':4,'device':'cpu','temperature':0,'top_p':1})
        engine.model=Model();engine.tokenizer=BPETokenizer();engine.metadata={'system_prompt':'BCI'};engine.manifest={'step':1}
        with patch.object(engine,'ensure_loaded'):
            events=list(engine.stream([{'role':'user','content':'Hi'}]))
        self.assertEqual(events[-1]['event'],'cancelled')

class HTTPContracts(unittest.TestCase):
    def setUp(self):
        self.server=main.Server(('127.0.0.1',0),main.Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_port}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
    def request(self,body=None,headers=None,endpoint='/chat'):
        return urlopen(Request(self.base+endpoint,data=None if body is None else json.dumps(body).encode(),headers=headers or {'Content-Type':'application/json'}),timeout=5)
    def test_health_and_origin_host(self):
        with self.request(endpoint='/health') as r:self.assertEqual(json.load(r)['api'],'ready')
        for header in [{'Origin':'https://example.com'},{'Host':'evil.example'}]:
            with self.assertRaises(HTTPError) as caught:self.request(endpoint='/health',headers=header)
            self.assertEqual(caught.exception.code,403)
    def test_invalid_request_rejected_before_engine(self):
        with patch.object(main.ENGINE,'stream') as engine:
            with self.assertRaises(HTTPError) as caught:self.request({'messages':[]})
            self.assertEqual(caught.exception.code,400);engine.assert_not_called()
    def test_busy_request_rejected(self):
        main.GATE.acquire()
        try:
            with self.assertRaises(HTTPError) as caught:self.request({'messages':[{'role':'user','content':'Hi'}]})
            self.assertEqual(caught.exception.code,429)
        finally:main.GATE.release()
    def test_error_and_telemetry_failure_release_generation_gate(self):
        import sqlite3
        def fail(_):raise ValueError('No checkpoint');yield
        finished=threading.Event()
        def telemetry(*_):finished.set();raise sqlite3.OperationalError('full')
        with patch.object(main.ENGINE,'stream',side_effect=fail),patch.object(main.metrics,'record',side_effect=telemetry),patch.object(main.LOGGER,'exception'):
            with self.assertRaises(HTTPError) as caught:self.request({'messages':[{'role':'user','content':'Hi'}]})
            self.assertEqual(caught.exception.code,503)
            self.assertTrue(finished.wait(5))
        self.assertTrue(main.GATE.acquire(blocking=False));main.GATE.release()
    def test_serialization_error_is_reported_and_gate_released(self):
        import pickle
        def fail(_):raise pickle.UnpicklingError('Untrusted checkpoint');yield
        finished=threading.Event()
        with patch.object(main.ENGINE,'stream',side_effect=fail),patch.object(main.metrics,'record',side_effect=lambda *_:finished.set()),patch.object(main.LOGGER,'exception'):
            with self.assertRaises(HTTPError) as caught:self.request({'messages':[{'role':'user','content':'Hi'}]})
            self.assertEqual(caught.exception.code,503)
            self.assertTrue(finished.wait(5))
        self.assertTrue(main.GATE.acquire(blocking=False));main.GATE.release()
    def test_slow_metadata_does_not_hold_inference_slot(self):
        logging_started=threading.Event();finish_logging=threading.Event()
        def fail(_):raise ValueError('No checkpoint');yield
        def slow_logging(*_):logging_started.set();finish_logging.wait(5)
        try:
            with patch.object(main.ENGINE,'stream',side_effect=fail),patch.object(main.metrics,'record',side_effect=slow_logging),patch.object(main.LOGGER,'exception'):
                with self.assertRaises(HTTPError):self.request({'messages':[{'role':'user','content':'Hi'}]})
                self.assertTrue(logging_started.wait(5))
                self.assertTrue(main.GATE.acquire(blocking=False));main.GATE.release()
        finally:finish_logging.set()
    def test_stream_completion_and_cancel_route(self):
        def stream(_):
            yield {'event':'start','dropped_turns':0};yield {'event':'token','text':'Hello'};yield {'event':'done','generated_tokens':1}
        with patch.object(main.ENGINE,'stream',side_effect=stream),patch.object(main.metrics,'record'):
            with self.request({'messages':[{'role':'user','content':'Hi'}]}) as r:
                events=[json.loads(line) for line in r]
        self.assertEqual(events[-1]['event'],'done')
        with self.request({},endpoint='/cancel') as r:self.assertEqual(json.load(r)['status'],'cancellation_requested')
        main.ENGINE.cancel.clear()

class UISmoke(unittest.TestCase):
    def test_chat_and_navigation_render(self):
        from streamlit.testing.v1 import AppTest
        from pathlib import Path
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/'ui/app.py')).run(timeout=10)
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value,'Erudite')
        app.sidebar.radio[0].set_value('Telemetry').run(timeout=10);self.assertFalse(app.exception)
        app.sidebar.radio[0].set_value('Document retrieval').run(timeout=10);self.assertFalse(app.exception)
        self.assertIn('not implemented',app.info[0].value)
if __name__=='__main__':unittest.main()
