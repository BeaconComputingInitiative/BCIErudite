# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from core.guardrails.validation import messages, record
from core.inference.tokenizer import BPETokenizer, SPECIAL
from core.inference.model import EruditeLM, ModelConfig
from core.inference.checkpoint import load, save
from training.data import PreparedData, batch
from training.prepare_dataset import prepare
from training.train_model import train, evaluate

torch.set_num_threads(1)

def corpus(directory, count=30):
    source = Path(directory) / "raw.jsonl"
    source.write_text("".join(json.dumps({"text": f"Document {i}. Erudite learns a small local language model. The sun rises in the morning."}) + "\n" for i in range(count)))
    return source

class TokenizerTests(unittest.TestCase):
    def test_utf8_unknown_words_and_whitespace(self):
        tokenizer = BPETokenizer.train(["Erudite learns bytes and words.\n" * 5], 300)
        for text in ["Erudite learns", "café 🌌 漢字", "\n\t  test", "", "👨‍👩‍👧"]:
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
    def test_deterministic_and_actual_size(self):
        a = BPETokenizer.train(["aa"], 1000); b = BPETokenizer.train(["aa"], 1000)
        self.assertEqual(a.fingerprint, b.fingerprint); self.assertEqual(a.vocab_size, 263)
    def test_save_load_and_mutation(self):
        tokenizer = BPETokenizer.train(["native tokenizer"], 280)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"; tokenizer.save(path)
            self.assertEqual(BPETokenizer.load(path).fingerprint, tokenizer.fingerprint)
            spec = json.loads(path.read_text()); spec["vocab_size"] += 1; path.write_text(json.dumps(spec))
            with self.assertRaises(ValueError): BPETokenizer.load(path)
    def test_chat_specials_and_mask(self):
        tokenizer = BPETokenizer()
        ids, mask = tokenizer.conversation([{"role": "user", "content": "A"}, {"role": "assistant", "content": "B"}])
        self.assertEqual(ids, [256, 260, 65, 257, 261, 66, 257]); self.assertEqual(mask, [0, 0, 0, 0, 0, 1, 1])
    def test_unknown_id_rejected(self):
        with self.assertRaises(ValueError): BPETokenizer().decode([9999])
    def test_invalid_merge_rejected(self):
        with self.assertRaises(ValueError): BPETokenizer([(9999, 65)])

class SchemaTests(unittest.TestCase):
    def test_good_conversation_and_raw_record(self):
        self.assertEqual(record({"text": " hi "}), {"text": "hi"})
        self.assertEqual(len(messages([{"role": "user", "content": "Hi"}], generation=True)), 1)
    def test_no_fabricated_log_answers(self):
        with self.assertRaises(ValueError): record({"timestamp": 1, "event": "observed"})
    def test_invalid_roles_empty_and_order(self):
        bad = [[], [{"role": "assistant", "content": "x"}], [{"role": "user", "content": ""}],
               [{"role": "user", "content": "x"}, {"role": "user", "content": "y"}],
               [{"role": "user", "content": "x"}, {"role": "system", "content": "y"}]]
        for history in bad:
            with self.assertRaises(ValueError): messages(history, generation=True)

class DataTests(unittest.TestCase):
    def test_dedup_group_split_and_training_only_tokenizer(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); raw = corpus(d); raw.write_text(raw.read_text() + raw.read_text())
            output = d / "prepared"; report = prepare([raw], output, vocab=280)
            data = PreparedData(output)
            self.assertEqual(report["unique_payloads"], 30)
            self.assertEqual(report["duplicate_rows_excluding_identical_files"], 30)
            groups = [{r["group"] for r in rows} for rows in data.rows.values()]
            self.assertTrue(all(not a & b for i, a in enumerate(groups) for b in groups[i+1:]))
            original = [record(json.loads(line)) for line in raw.read_text().splitlines()]
            payloads = {hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest(): r for r in original}
            trained = BPETokenizer.train([payloads[r["source_sha256"]]["text"] for r in data.rows["train"]], 280)
            self.assertEqual(trained.fingerprint, data.tokenizer.fingerprint)
    def test_same_prompt_stays_together(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); raw = corpus(d)
            with raw.open("a") as out:
                for answer in ["First", "Second"]:
                    out.write(json.dumps({"messages": [{"role": "user", "content": "same prompt"}, {"role": "assistant", "content": answer}]}) + "\n")
            prepare([raw], d / "prepared", vocab=280); data = PreparedData(d / "prepared")
            matches = [name for name, rows in data.rows.items() if any(row["kind"] == "chat" for row in rows)]
            self.assertEqual(len(matches), 1)
    def test_corruption_and_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); raw = corpus(d); output = d / "prepared"; prepare([raw], output, vocab=280)
            with self.assertRaises(FileExistsError): prepare([raw], output)
            with (output / "train.jsonl").open("a") as out: out.write("\n")
            with self.assertRaises(ValueError): PreparedData(output)
    def test_invalid_row_fails_loudly(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory); raw = corpus(d); raw.write_text(raw.read_text() + 'broken-json\n')
            with self.assertRaisesRegex(ValueError, 'raw.jsonl'): prepare([raw], d / "prepared")
    def test_assistant_only_window_masking(self):
        data = object.__new__(PreparedData)
        data.rows = {"train": [{"tokens": [256, 260, 65, 257, 261, 66, 257], "assistant_mask": [0,0,0,0,0,1,1], "kind": "chat"}]}
        windows = data.windows("train", 8, "assistant")
        self.assertEqual(windows[0][1], [-100,-100,-100,-100,66,257])

class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = EruditeLM(ModelConfig(vocab_size=280, width=32, layers=2, heads=4, context=32)).eval()
    def test_future_tokens_do_not_change_past_logits(self):
        ids = torch.tensor([[256,65,66,67,68]])
        alternative = ids.clone(); alternative[0,3:] = torch.tensor([90,91])
        with torch.no_grad(): a,_,_ = self.model(ids); b,_,_ = self.model(alternative)
        self.assertTrue(torch.allclose(a[:,:3], b[:,:3], atol=1e-6))
    def test_single_token_and_chunk_cache_equivalence(self):
        ids = torch.tensor([[256,65,66,67,68,69]])
        with torch.no_grad():
            full,_,_ = self.model(ids); _,_,cache = self.model(ids[:,:3])
            chunk,_,_ = self.model(ids[:,3:], caches=cache)
            self.assertTrue(torch.allclose(full[:,3:], chunk, atol=2e-6))
            token,_,_ = self.model(ids[:,3:4], caches=cache)
            self.assertTrue(torch.allclose(full[:,3:4], token, atol=2e-6))
    def test_padding_does_not_affect_real_targets(self):
        rows = [([256,65],[65,257]), ([256,66,67],[66,67,257])]
        x,y = batch(rows,'cpu'); logits,loss,_ = self.model(x,y)
        self.assertEqual(y[0,-1].item(),-100)
        expected = torch.nn.functional.cross_entropy(logits.reshape(-1,280), y.reshape(-1), ignore_index=-100)
        self.assertTrue(torch.allclose(loss, expected))
        single,_,_ = self.model(torch.tensor([[256,65]]))
        self.assertTrue(torch.allclose(logits[0,:2], single[0], atol=2e-6))
    def test_real_gradient_and_overfit_tiny_sequence(self):
        model = self.model.train(); optimizer = torch.optim.AdamW(model.parameters(),lr=0.01)
        x = torch.tensor([[256,65,66,67]]); y = torch.tensor([[65,66,67,257]])
        _,initial,_ = model(x,y)
        for _ in range(15):
            optimizer.zero_grad(); _,loss,_ = model(x,y); loss.backward(); optimizer.step()
        _,final,_ = model(x,y)
        self.assertLess(final.item(), initial.item())
        self.assertTrue(torch.isfinite(model.embedding.weight.grad).all())
    def test_context_architecture_cache_contracts(self):
        with self.assertRaises(ValueError): ModelConfig(280,width=31,heads=4)
        with self.assertRaises(ValueError): self.model(torch.ones(1,33,dtype=torch.long))
        with self.assertRaises(ValueError): self.model(torch.ones(1,1,dtype=torch.long),caches=[])
        with self.assertRaises(ValueError): list(self.model.generate(torch.tensor([[65]]),32))
    def test_generation_forbids_control_tokens(self):
        output = list(self.model.generate(torch.tensor([[256,260,65,257,261]]),4,temperature=0))
        self.assertTrue(all(t<256 or t>=262 for t in output))
    def test_checkpoint_hash_pairing_and_weight_tying(self):
        tokenizer = BPETokenizer.train(["native words"],280)
        model = EruditeLM(ModelConfig(tokenizer.vocab_size,width=32,layers=1,heads=4,context=32))
        with tempfile.TemporaryDirectory() as directory:
            save(directory,model,tokenizer,None,{"step":1})
            restored,t,_,manifest = load(Path(directory)/'latest.json')
            self.assertEqual(t.fingerprint,tokenizer.fingerprint)
            self.assertIs(restored.output.weight,restored.embedding.weight)
            self.assertTrue(torch.equal(restored.embedding.weight,model.embedding.weight))
            self.assertEqual(str(next(restored.parameters()).device),'cpu')
            with (Path(directory)/manifest['file']).open('ab') as out:out.write(b'changed')
            with self.assertRaises(ValueError):load(Path(directory)/'latest.json')
    def test_tokenizer_mismatch_detected(self):
        tokenizer = BPETokenizer(); model=EruditeLM(ModelConfig(262,width=32,layers=1,heads=4,context=32))
        with tempfile.TemporaryDirectory() as directory:
            save(directory,model,tokenizer,None,{"step":1}); pointer=Path(directory)/'latest.json'
            manifest=json.loads(pointer.read_text());manifest['tokenizer_sha256']='wrong';pointer.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):load(pointer)

class EndToEnd(unittest.TestCase):
    def test_prepare_train_evaluate_native_inference_and_resume(self):
        from core.inference.engine import Engine
        from training.evaluate_model import run
        with tempfile.TemporaryDirectory() as directory:
            d=Path(directory);raw=corpus(d);data=d/'prepared';prepare([raw],data,vocab=280)
            config={"prepared_dir":str(data),"output_dir":str(d/'run'),"architecture":{"width":32,"layers":1,"heads":4,"context":256},
                    "device":"cpu","steps":6,"batch_size":2,"learning_rate":0.003,"weight_decay":0.1,"warmup_steps":1,
                    "grad_clip":1.0,"eval_every":3,"seed":42,"objective":"all"}
            cp=d/'config.json';cp.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):pointer=train(cp)
            result=run(pointer,data);self.assertGreater(result['evaluated_tokens'],0)
            engine=Engine({"checkpoint":str(pointer),"device":"cpu","max_output_tokens":4,"temperature":0,"top_p":1})
            # Serve exactly the tokenizer and system prompt captured by real training.
            model,t,bundle,_=load(pointer)
            directory2=d/'serve';save(directory2,model,t,None,{**bundle['metadata'],"step":1})
            engine.config['checkpoint']=str(directory2/'latest.json')
            events=list(engine.stream([{"role":"user","content":"Hi"}]))
            self.assertEqual(events[0]['event'],'start');self.assertEqual(events[-1]['event'],'done')
            revision=engine.revision
            save(directory2,model,t,None,{**bundle['metadata'],"step":2})
            list(engine.stream([{"role":"user","content":"Hi"}]))
            self.assertNotEqual(revision,engine.revision)
            # Resume exact data/tokenizer/architecture into a fresh run directory.
            config['output_dir']=str(d/'resumed');config['steps']=9;cp.write_text(json.dumps(config))
            with contextlib.redirect_stdout(io.StringIO()):resumed=train(cp,d/'run/latest.json')
            self.assertTrue(resumed.is_file())
            with self.assertRaises(FileExistsError):train(cp)

if __name__=='__main__':unittest.main()
