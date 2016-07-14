"""
@author: dncampo
"""

import unittest
import numpy as np
import sys

sys.path.insert(0, '../')
from youtube_experiment import *

an_obj = Youtube_experiments()

class TestYoutubeExperimentFunctions(unittest.TestCase):
    def setUp(self):
        self.seq = np.array([1, 1, 0, 2])

    def test_sort_plot_values_by_overlap(self):
        avg_over_1 = [1.1, 0]
        index_1    = [[0.81, 0.82], [0.98, 0.97]]
        avg_over_1_ok = [0, 1.1]
        index_1_ok    = [[0.98, 0.97], [0.81, 0.82]]

        avg_over_2 = [0, 1.1, 3, 1.1]
        index_2 = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.5, 0.4, 0.6], [0.9, 0.9, 1]]
        avg_over_2_ok = [0, 1.1, 1.1, 3]
        index_2_ok = [[0.1, 0.2, 0.3], [0.9, 0.8, 0.7], [0.9, 0.9, 1], [0.5, 0.4, 0.6]]

        avg_over_1_test, index_1_test = an_obj.sort_plot_values_by_overlap(avg_over_1, index_1)
        avg_over_2_test, index_2_test = an_obj.sort_plot_values_by_overlap(avg_over_2, index_2)

        self.assertListEqual(avg_over_1_ok, avg_over_1_test, "avg_overlap_1 test and ok non equals")
        self.assertListEqual(index_1_ok, index_1_test, "index_1 test and ok non equals")

        self.assertListEqual(avg_over_2_ok, avg_over_2_test, "avg_overlap_2 test and ok non equals")
        self.assertListEqual(index_2_ok, index_2_test, "index_2 test and ok non equals")

    def test_remove_null_rows(self):
        arr_1 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        arr_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        arr_3 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 5]])
        arr_4 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [0, 0, 0, 0]])

        arr_1_ok = np.array([[1, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0]])
        arr_2_ok = np.array([[4, 0, 0, 0]])
        arr_3_ok = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 5]])
        arr_4_ok = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4]])

        arr_1_removed = an_obj.remove_null_rows(arr_1)
        arr_2_removed = an_obj.remove_null_rows(arr_2)
        arr_3_removed = an_obj.remove_null_rows(arr_3)
        arr_4_removed = an_obj.remove_null_rows(arr_4)

        self.assertListEqual(arr_1_ok.tolist(), arr_1_removed.tolist(), "arr_1 non equal")
        self.assertListEqual(arr_2_ok.tolist(), arr_2_removed.tolist(), "arr_2 non equal")
        self.assertListEqual(arr_3_ok.tolist(), arr_3_removed.tolist(), "arr_3 non equal")
        self.assertListEqual(arr_4_ok.tolist(), arr_4_removed.tolist(), "arr_4 non equal")


    def test_remove_null_columns(self):
        arr_1 = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [3, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        arr_2 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0, 0]])
        arr_3 = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 5]])
        arr_4 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [0, 0, 0, 0]])
        arr_5 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 5]])
        arr_6 = np.array([[1, 1, 1, 1]])

        arr_1_2_ok_1 = np.array([[1], [0], [3], [4], [0]])
        arr_1_2_ok_2 = np.array([[0], [0], [0], [4], [0]])

        arr_1_3_ok_1 = np.array([[], [], [], [], []])
        arr_1_3_ok_3 = np.array([[], [], [], [], []])

        arr_4_5_ok_4 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [0, 0, 0, 0]])
        arr_4_5_ok_5 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 5]])

        arr_4_6_ok_4 = np.array([[1, 0, 0, 0], [0, 2, 0, 0], [0, 0, 3, 0], [0, 0, 0, 4], [0, 0, 0, 0]])
        arr_4_6_ok_6 = np.array([[1, 1, 1, 1]])

        arr_1_2_removed_1, arr_1_2_removed_2 = an_obj.remove_null_columns(arr_1, arr_2)
        arr_1_3_removed_1, arr_1_3_removed_3 = an_obj.remove_null_columns(arr_1, arr_3)
        arr_4_5_removed_4, arr_4_5_removed_5 = an_obj.remove_null_columns(arr_4, arr_5)
        arr_4_6_removed_4, arr_4_6_removed_6 = an_obj.remove_null_columns(arr_4, arr_6)

        self.assertListEqual(arr_1_2_ok_1.tolist(), arr_1_2_removed_1.tolist())
        self.assertListEqual(arr_1_2_ok_2.tolist(), arr_1_2_removed_2.tolist())

        self.assertListEqual(arr_1_3_ok_1.tolist(), arr_1_3_removed_1.tolist())
        self.assertListEqual(arr_1_3_ok_3.tolist(), arr_1_3_removed_3.tolist())

        self.assertListEqual(arr_4_5_ok_4.tolist(), arr_4_5_removed_4.tolist())
        self.assertListEqual(arr_4_5_ok_5.tolist(), arr_4_5_removed_5.tolist())

        self.assertListEqual(arr_4_6_ok_4.tolist(), arr_4_6_removed_4.tolist())
        self.assertListEqual(arr_4_6_ok_6.tolist(), arr_4_6_removed_6.tolist())


if __name__ == '__main__':
    unittest.main()
