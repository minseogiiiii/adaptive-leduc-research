"""Observability follow-up tests."""
import unittest

from pokerlab.qt_v6_observability import _auc,run

class ObservabilityTests(unittest.TestCase):
    def test_auc(self):
        self.assertEqual(_auc([1,2],[0,0]),1)
        self.assertEqual(_auc([0],[1]),0)
        self.assertEqual(_auc([1],[1]),.5)

    def test_probe_reproducibility_and_bounds(self):
        a=run(seed=82,replicates=2,hands_per_type=3)
        b=run(seed=82,replicates=2,hands_per_type=3)
        self.assertEqual(a["probe_metrics"],b["probe_metrics"])
        self.assertEqual(a["contrasts"],b["contrasts"])
        for obj in a["probe_metrics"].values():
            for name in ("auc","accuracy","sensitivity","false_positive"):
                self.assertTrue(0<=obj[name]<=1)

if __name__=="__main__":
    unittest.main()
