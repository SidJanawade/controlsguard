import unittest

from controlsguard.config import Settings
from controlsguard.context import ContextBuilder, SourceRegistry
from controlsguard.ingestion import Document, chunk_document, clean_text, ingest
from controlsguard.memory import Session, SessionStore
from controlsguard.retrieval import HybridRetriever
from controlsguard.textutil import count_tokens, stem, tokenize
from controlsguard.workflow import END, StateGraph

from .helpers import shared_agent


class TextUtilTests(unittest.TestCase):
    def test_stemming_aligns_word_forms(self):
        self.assertEqual(stem("approves"), stem("approved"))
        self.assertEqual(stem("policies"), stem("policy"))
        self.assertEqual(stem("accesses"), stem("access"))

    def test_hyphenated_ids_kept_whole_and_split(self):
        toks = tokenize("Check ledger-core and C-AC-01")
        self.assertIn("ledger-core", toks)
        self.assertIn("ledger", toks)
        self.assertIn("c-ac-01", toks)


class IngestionTests(unittest.TestCase):
    def test_corpus_loads_and_parses_fields(self):
        docs, chunks = ingest(Settings().data_dir)
        self.assertEqual(len(docs), 7)
        c = next(c for c in chunks if c.entity_id == "C-AC-01")
        self.assertEqual(c.fields["risks"], ["R-UNAUTH"])
        self.assertIn("ledger-core", c.fields["systems"])
        self.assertTrue(c.text.startswith("Access Control Policy > C-AC-01"))

    def test_clean_text_strips_control_and_zero_width_chars(self):
        self.assertEqual(clean_text("a​b\x00c   d"), "abc d")

    def test_long_sections_split_with_overlap(self):
        sentences = " ".join(f"Sentence number {i} talks about retention of records." for i in range(40))
        doc = Document("d", "Doc", f"# Doc\n\n## Long section\n{sentences}", "x")
        chunks = chunk_document(doc, max_tokens=60, overlap_tokens=15)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(count_tokens(c.text) <= 60 + 40 for c in chunks))
        # overlap: the last sentence of a window reappears at the start of the next
        first_tail = chunks[0].text.strip().split(". ")[-1][:25]
        self.assertIn(first_tail.split(" talks")[0][:20], chunks[1].text)


class RetrievalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = shared_agent().retriever

    def test_bm25_vector_and_hybrid_find_the_control(self):
        for mode in ("bm25", "vector", "hybrid"):
            ids = [h.chunk.entity_id for h in self.r.search("quarterly access recertification attest", k=3, mode=mode)]
            self.assertIn("C-AC-01", ids, mode)

    def test_hybrid_reports_per_retriever_ranks(self):
        hit = self.r.search("regulator notification 72 hours", k=1)[0]
        self.assertEqual(hit.chunk.entity_id, "C-IR-03")
        self.assertEqual(hit.ranks["bm25"], 1)

    def test_doc_filter(self):
        hits = self.r.search("approval", k=5, doc_id="change-management")
        self.assertTrue(hits and all(h.chunk.doc_id == "change-management" for h in hits))

    def test_query_coverage_separates_on_and_off_topic(self):
        self.assertGreaterEqual(self.r.query_coverage("who approves emergency deployments"), 0.9)
        self.assertLess(self.r.query_coverage("what is the weather in Pune"), 0.4)

    def test_entity_lookup(self):
        self.assertEqual(self.r.by_entity("c-ac-02").entity_id, "C-AC-02")
        self.assertIsNone(self.r.by_entity("C-NOPE-99"))


class GraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = shared_agent().graph

    def test_structure(self):
        st = self.g.stats()
        self.assertEqual(st["by_type"]["control"], 19)
        self.assertEqual(st["by_type"]["risk"], 8)
        self.assertEqual(st["by_type"]["system"], 6)

    def test_resolve_ids_and_names(self):
        self.assertEqual(self.g.resolve("what protects ledger-core?"), ["ledger-core"])
        self.assertIn("R-UNAUTH", self.g.resolve("risk r-unauth and unauthorized access"))
        self.assertEqual(self.g.resolve("nothing relevant here"), [])

    def test_coverage_finds_the_known_gap_and_no_false_gaps(self):
        self.assertEqual(self.g.coverage("ledger-core")["gaps"], ["R-VENDOR"])
        for system in ("payments-gateway", "genai-assistant", "trade-db", "prod-servers", "analytics-warehouse"):
            self.assertEqual(self.g.coverage(system)["gaps"], [], system)

    def test_neighbors_and_paths_carry_provenance(self):
        triples = self.g.neighbors("C-AC-01", depth=1)
        self.assertTrue(all(t.chunk_id for t in triples))
        self.assertIn(("C-AC-01", "mitigates", "R-UNAUTH"), [(t.subject, t.relation, t.object) for t in triples])
        path = self.g.path("C-TP-03", "R-AI-HARM")
        self.assertTrue(path)
        self.assertEqual(self.g.path("C-TP-03", "no-such-node"), [])

    def test_controls_for_risk_and_system(self):
        self.assertEqual(self.g.controls_mitigating("R-INSIDER"), ["C-AC-02", "C-CM-03"])
        self.assertIn("C-TP-01", self.g.controls_protecting("payments-gateway"))


class WorkflowTests(unittest.TestCase):
    def test_conditional_routing_and_trail(self):
        g = StateGraph()
        g.add_node("a", lambda s: {"x": 1})
        g.add_node("b", lambda s: {"path": "b"})
        g.add_node("c", lambda s: {"path": "c"})
        g.set_entry_point("a")
        g.add_conditional_edges("a", lambda s: "go_c" if s["x"] == 1 else "go_b", {"go_b": "b", "go_c": "c"})
        g.add_edge("b", END)
        g.add_edge("c", END)
        out = g.compile().invoke({})
        self.assertEqual(out["path"], "c")
        self.assertEqual([n["node"] for n in out["node_trail"]], ["a", "c"])

    def test_cycle_guard(self):
        g = StateGraph()
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_edge("a", "a")
        with self.assertRaises(RuntimeError):
            g.compile().invoke({})

    def test_unknown_node_rejected_at_compile(self):
        g = StateGraph()
        g.add_node("a", lambda s: {})
        g.set_entry_point("a")
        g.add_edge("a", "missing")
        with self.assertRaises(AssertionError):
            g.compile()


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.agent = shared_agent()
        self.chunks = self.agent.chunks[:12]

    def build(self, budget, memory=None):
        return ContextBuilder(budget).build(
            system_prompt="SYS " * 50, tool_specs=[{"name": "t", "description": "d", "parameters": {}}],
            query="what is required?", chunks=self.chunks, graph_facts=[f"fact {i} about controls" for i in range(30)],
            memory=memory or [], registry=SourceRegistry())

    def test_budget_is_respected_and_sources_are_dropped_when_tight(self):
        big, small = self.build(4000), self.build(500)
        self.assertGreater(big.report["sources_included"], small.report["sources_included"])
        self.assertLessEqual(small.report["total"], 500 + 60)  # fixed costs are never dropped
        self.assertGreater(small.report["graph_facts_dropped"], 0)

    def test_registry_tags_are_stable(self):
        reg = SourceRegistry()
        self.assertEqual(reg.cite(self.chunks[0]), "S1")
        self.assertEqual(reg.cite(self.chunks[1]), "S2")
        self.assertEqual(reg.cite(self.chunks[0]), "S1")

    def test_memory_window_summarises_old_turns(self):
        s = Session("x")
        for i in range(10):
            s.add_turn(f"question number {i} about policies " * 3, f"answer {i} " * 30)
        msgs = s.window(120)
        self.assertLess(len(msgs), 20)
        self.assertIn("Earlier in this conversation", msgs[0]["content"])
        self.assertTrue(msgs[-2]["content"].startswith("question number 9"))

    def test_session_store_evicts_expired_and_overflow(self):
        store = SessionStore(ttl_s=0, max_sessions=2)
        a = store.get("a")
        a.last_seen -= 10
        store.get("b")
        self.assertNotIn("a", store._sessions)
        store2 = SessionStore(ttl_s=999, max_sessions=2)
        for sid in ("a", "b", "c"):
            store2.get(sid)
        self.assertLessEqual(len(store2), 2)


if __name__ == "__main__":
    unittest.main()
