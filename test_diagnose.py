import json
import unittest
from pathlib import Path

import diagnose

ROOT = Path(__file__).resolve().parent


class CosineTests(unittest.TestCase):
    def test_identical_vectors_score_one(self):
        self.assertAlmostEqual(diagnose.cosine([1, 2, 3], [1, 2, 3]), 1)

    def test_orthogonal_vectors_score_zero(self):
        self.assertAlmostEqual(diagnose.cosine([1, 0], [0, 1]), 0)

    def test_parallel_vectors_score_one(self):
        self.assertAlmostEqual(diagnose.cosine([1, 2], [2, 4]), 1)


class YearScopeTests(unittest.TestCase):
    def test_inclusive_bounds(self):
        self.assertTrue(diagnose.year_in_scope("2013"))
        self.assertTrue(diagnose.year_in_scope("2018"))

    def test_outside_years_and_non_numbers(self):
        self.assertFalse(diagnose.year_in_scope("2012"))
        self.assertFalse(diagnose.year_in_scope("2019"))
        self.assertFalse(diagnose.year_in_scope("2020"))
        self.assertFalse(diagnose.year_in_scope("nope"))


class SubmitDiagnosisTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "retrieved": [
                {"id": "wp-01", "status": "possible"},
                {"id": "cc-01", "status": "possible"},
            ]
        }

    def test_rejects_submit_before_search(self):
        result, done = diagnose.submit_diagnosis(
            {"chunk_ids": ["wp-01"], "causes": []},
            {"retrieved": []},
        )
        self.assertFalse(done)
        self.assertFalse(result["accepted"])

    def test_rejects_empty_citations(self):
        result, done = diagnose.submit_diagnosis(
            {"chunk_ids": [], "causes": []},
            self.state,
        )
        self.assertFalse(done)
        self.assertFalse(result["accepted"])

    def test_rejects_id_that_was_not_retrieved(self):
        result, done = diagnose.submit_diagnosis(
            {
                "chunk_ids": ["tank-01"],
                "causes": [
                    {
                        "name": "cracked tank",
                        "likelihood": "high",
                        "chunk_id": "tank-01",
                        "status": "confirmed",
                    }
                ],
            },
            self.state,
        )
        self.assertFalse(done)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["retrieved_ids"], ["cc-01", "wp-01"])

    def test_rejects_status_that_was_changed(self):
        result, done = diagnose.submit_diagnosis(
            {
                "chunk_ids": ["wp-01"],
                "causes": [
                    {
                        "name": "water-pump weep hole",
                        "likelihood": "medium",
                        "chunk_id": "wp-01",
                        "status": "confirmed",
                    }
                ],
            },
            self.state,
        )
        self.assertFalse(done)
        self.assertFalse(result["accepted"])
        self.assertEqual(result["mismatched"][0]["expected"], "possible")

    def test_accepts_status_copied_from_the_note(self):
        arguments = {
            "chunk_ids": ["wp-01"],
            "causes": [
                {
                    "name": "water-pump weep hole",
                    "likelihood": "medium",
                    "chunk_id": "wp-01",
                    "status": "possible",
                }
            ],
            "repair_suggestion": ["Watch the weep hole."],
            "ride": "limited",
        }
        result, done = diagnose.submit_diagnosis(arguments, self.state)
        self.assertTrue(done)
        self.assertTrue(result["accepted"])
        self.assertEqual(result["diagnosis"]["causes"][0]["status"], "possible")


class WrongYearRunTests(unittest.TestCase):
    def test_stops_before_retrieval(self):
        diagnose.run(ROOT / "cases" / "wrong-year.txt")
        trace = json.loads((ROOT / "traces" / "wrong-year.json").read_text())
        self.assertEqual(trace["steps"], [])
        self.assertEqual(trace["final"]["stopped"], "year outside 2013-2018")
        self.assertEqual(trace["final"]["year"], "2020")


if __name__ == "__main__":
    unittest.main()
