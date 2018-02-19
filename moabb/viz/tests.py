import unittest
import numpy as np
import moabb.viz.meta_analysis as ma

class Test_Stats(unittest.TestCase):

    def test_rmanova(self):
        matrix=np.asarray([[45,50,55],
                           [42,42,45],
                           [36,41,43],
                           [39,35,40],
                           [51,55,59],
                           [44,49,56]])
        f, p = ma._rmanova(matrix)
        self.assertAlmostEqual(f, 12.53, places=2)
        self.assertAlmostEqual(p, 0.002, places=3)



if __name__ == "__main__":
    unittest.main()
