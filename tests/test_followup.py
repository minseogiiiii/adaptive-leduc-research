import json
import tempfile
import unittest
from pathlib import Path
from statistics import fmean

from pokerlab.followup import OPPONENTS, render_followup


class FollowupReportTest(unittest.TestCase):
    def setUp(self):
        self.paths = tuple(Path(f"results/followup-{name}.json")
                           for name in (*OPPONENTS, "switch"))
        self.kwargs = dict(strategy=Path("strategy.json"), rules=Path("rules.md"),
                           protocol=Path("FOLLOWUP_PROTOCOL.md"))

    def test_saved_report_reconstructs_primary_mean_from_raw_hands(self):
        report = render_followup(self.paths, **self.kwargs)
        rows = [json.loads(path.read_text())["hands"] for path in self.paths[:3]]
        by_rep = []
        for rep in range(24):
            differences = []
            for condition in rows:
                seat_means = [fmean(row["rewards"]["learning"]
                                    - row["rewards"]["frozen_prior"]
                                    for row in condition
                                    if row["replicate"] == rep and row["seat"] == seat)
                              for seat in (0, 1)]
                differences.append(fmean(seat_means))
            by_rep.append(fmean(differences))
        self.assertIn(f"**{fmean(by_rep):+.4f} [", report)
        self.assertEqual(report, Path("FOLLOWUP.md").read_text())

    def _mutated(self, name, mutate):
        with tempfile.TemporaryDirectory() as directory:
            data = json.loads(self.paths[name].read_text())
            mutate(data)
            changed = Path(directory) / "altered.json"
            changed.write_text(json.dumps(data))
            paths = list(self.paths)
            paths[name] = changed
            render_followup(tuple(paths), **self.kwargs)

    def test_rejects_changed_seed_schedule_even_when_record_is_valid(self):
        with self.assertRaisesRegex(ValueError, "different hand seed schedules"):
            self._mutated(1, lambda result: result["hands"][0].__setitem__(
                "seed", -123456789))

    def test_rejects_changed_reward_without_consistent_summaries(self):
        with self.assertRaisesRegex(ValueError, "Inconsistent"):
            self._mutated(0, lambda result: result["hands"][0]["rewards"].__setitem__(
                "learning", 123.0))

    def test_rejects_changed_resampling_plan(self):
        with self.assertRaisesRegex(ValueError, "Unexpected follow-up workload"):
            self._mutated(2, lambda result: result["design"].__setitem__(
                "bootstrap_resamples", 1999))


if __name__ == "__main__":
    unittest.main()
