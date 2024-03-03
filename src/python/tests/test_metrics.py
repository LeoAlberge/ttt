import unittest

import torch

from src.python.training.metrics import MeanDiceScore, SegmentationMultiDiceScores
import numpy as np


class MetricsTestCase(unittest.TestCase):

    def test_mean_dice(self):
        m1 = torch.tensor(np.ones((3, 5, 10, 10, 10)), )
        m2 = torch.tensor(np.ones((3, 5, 10, 10, 10)))
        mdc = MeanDiceScore(apply_argmax=False, apply_softmax=False)
        mdc.update(m1, m2)
        m = mdc.compute()
        assert m.numpy() == 1

    def test_mean_dice_case_2(self):
        m1 = np.ones((4, 3, 10, 10, 10))
        m1[:2, :, :, :, :] = 0
        m1 = torch.tensor(m1)
        m2 = torch.tensor(np.ones((4, 3, 10, 10, 10)))
        mdc = MeanDiceScore(apply_argmax=False, apply_softmax=False)
        mdc.update(m1, m2)
        m = mdc.compute()
        assert m.numpy() == 0.5

    def test_dice_multi(self):
        m1 = np.ones((4, 3, 10, 10, 10))
        m1[:2, :, :, :, :] = 0
        m1 = torch.tensor(m1)
        m2 = torch.tensor(np.ones((4, 3, 10, 10, 10)))
        mdc = SegmentationMultiDiceScores(apply_argmax=False, apply_softmax=False)
        mdc.update(m1, m2)
        m = mdc.compute().numpy()
        np.testing.assert_almost_equal(m, np.ones(3) * 0.5)

    def test_dice_multi_case2(self):
        m1 = np.ones((4, 3, 10, 10, 10))
        m1[:, 1, :, :, :] = 0
        m1 = torch.tensor(m1)
        m2 = torch.tensor(np.ones((4, 3, 10, 10, 10)))
        mdc = SegmentationMultiDiceScores(apply_argmax=False, apply_softmax=False)
        mdc.update(m1, m2)
        m = mdc.compute().numpy()
        np.testing.assert_almost_equal(m, [1, 0, 1])

    def test_dice_multi_case2_wiith_argmax(self):
        m1 = np.ones((4, 2, 10, 10, 10))
        m1[:, 1, :, :, :] = -1
        m1 = torch.tensor(m1)
        m2 = torch.tensor(np.ones((4, 2, 10, 10, 10)))
        mdc = SegmentationMultiDiceScores(apply_argmax=True, apply_softmax=False)
        mdc.update(m1, m2)
        m = mdc.compute().numpy()
        np.testing.assert_almost_equal(m, [1, 0])
    def test_dice_multi_cast_to_list_with_last_batch(self):
        m1 = np.ones((4, 2, 10, 10, 10))
        m1[:, 1, :, :, :] = -1
        m1 = torch.tensor(m1)
        m2 = torch.tensor(np.ones((4, 2, 10, 10, 10)))
        mdc = SegmentationMultiDiceScores(apply_argmax=True, apply_softmax=False)
        mdc.update(m1, m2)
        m1 = torch.tensor(np.ones((2, 2, 10, 10, 10)))
        m2 = torch.tensor(np.ones((2, 2, 10, 10, 10)))
        mdc.update(m1, m2)

        m = mdc.compute().numpy().tolist()
        np.testing.assert_almost_equal(m, [1, 0])
