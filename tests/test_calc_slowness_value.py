import math
from unittest import TestCase

import numpy as np

from config import SEED
from functions.calc_slowness_value import calc_slowness_for_array


class Test(TestCase):

    def test_calc_slowness_for_array_based_on_one_delta(self):

        # GIVEN
        generator = np.random.default_rng(SEED)
        a = generator.standard_normal(size=(1000, 1))

        # WHEN
        delta = calc_slowness_for_array(a)

        # THEN
        assert math.isclose(delta, 2.0, abs_tol=0.1)

    def test_calc_slowness_for_array_based_on_two_delta(self):

        # GIVEN
        generator = np.random.default_rng(SEED)
        a = generator.standard_normal(size=(1000, 2))

        # WHEN
        delta = calc_slowness_for_array(a, dims=2)

        # THEN
        print(delta)
        assert math.isclose(delta, 4.0, abs_tol=0.1)
