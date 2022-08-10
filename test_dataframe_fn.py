import unittest
import pandas as pd
import _01_dataframe_fn as edf
import numpy as np


class TestSameIO(unittest.TestCase):

    df_input = pd.DataFrame(
        {'name': ['foo', 'bar', 'foo', 'foo', 'bar', 'foo', 'bar', 'bar'],
         'measure_1': [5, 35, 10, 15, 20, 25, 30, 12],
         'measure_2': [100, 500, 150, 25, 250, 300, 400, 200]})

    df_output = edf.rows_for_minvalue_for_each_unique_name(df_input)

    def test_same_io(self):

        df_expected = pd.DataFrame(
            {'name': ['bar', 'foo'],
             'measure_1': [12, 5],
             'measure_2': [200, 100]})
        same = np.array_equal(self.__class__.df_output.values, df_expected.values)
        self.assertTrue(same)

    def test_num_of_cols(self):
        self.assertEqual(self.__class__.df_output.shape[1], self.__class__.df_input.shape[1])

    def test_num_of_rows(self):
        self.assertLessEqual(self.__class__.df_output.shape[0], self.__class__.df_input.shape[0])

    def test_order_of_rows(self):
        df_o = self.__class__.df_output.copy()
        df_o = df_o.reindex(index=df_o.index[::-1])
        # print(df_o)
        # print(self.__class__.df_output)
        same = np.array_equal(np.sort(df_o.values, axis=0), np.sort(self.__class__.df_output.values, axis=0))
        self.assertTrue(same)


if __name__ == '__main__':
    unittest.main()
