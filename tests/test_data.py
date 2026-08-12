import unittest

import torch

from thetamem.data import mad, mqar


class MQARDataTest(unittest.TestCase):
    def test_shapes_and_labels(self):
        inputs, labels = mqar.generate(8, 64, 4, vocab_size=128, seed=0)
        self.assertEqual(inputs.shape, (8, 64))
        self.assertEqual(labels.shape, (8, 64))
        for row in range(8):
            answered = labels[row] != -100
            self.assertEqual(int(answered.sum()), 4)
            # every answer is a value token (upper vocabulary half)
            self.assertTrue((labels[row][answered] >= 64).all())

    def test_query_positions_hold_keys(self):
        inputs, labels = mqar.generate(
            4, 64, 4, vocab_size=128, seed=3, random_fillers=False
        )
        for row in range(4):
            for position in torch.nonzero(labels[row] != -100).flatten():
                query = inputs[row, position]
                self.assertLess(int(query), 64)
                pairs = inputs[row, :8].view(4, 2)
                match = pairs[pairs[:, 0] == query]
                self.assertEqual(int(match[0, 1]), int(labels[row, position]))

    def test_determinism(self):
        a = mqar.generate(4, 64, 4, vocab_size=128, seed=7)
        b = mqar.generate(4, 64, 4, vocab_size=128, seed=7)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1], b[1])
        c = mqar.generate(4, 64, 4, vocab_size=128, seed=8)
        self.assertFalse(torch.equal(a[0], c[0]))

    def test_mixture_segments(self):
        segments = mqar.mixture(
            ((64, 4, 4), (128, 2, 8)), vocab_size=256, seed=0
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0][0].shape, (4, 64))
        self.assertEqual(segments[1][0].shape, (2, 128))


class MADDataTest(unittest.TestCase):
    def test_fuzzy_shapes(self):
        inputs, targets = mad.fuzzy_recall(6, seed=0, train=True)
        self.assertEqual(inputs.shape, (6, 128))
        self.assertEqual(targets.shape, (6, 128))
        # training targets are the shifted inputs
        torch.testing.assert_close(targets[:, :-1], inputs[:, 1:])

    def test_fuzzy_eval_targets_are_values(self):
        inputs, targets = mad.fuzzy_recall(6, seed=1, train=False)
        answered = targets != -100
        self.assertTrue(bool(answered.any()))
        self.assertTrue((targets[answered] >= 7).all())
        self.assertTrue((targets[answered] < 15).all())

    def test_selective_shapes_and_answers(self):
        inputs, targets = mad.selective_copy(6, seed=2)
        self.assertEqual(inputs.shape, (6, 256))
        answered = targets != -100
        self.assertEqual(int(answered.sum()), 6 * 16)
        # answers reproduce the content tokens in order
        for row in range(6):
            content = inputs[row][(inputs[row] != 14) & (inputs[row] != 15)]
            torch.testing.assert_close(
                targets[row][targets[row] != -100], content[:16]
            )

    def test_generate_dispatch(self):
        mad.generate("fuzzy", 2, seed=0)
        mad.generate("selective", 2, seed=0)
        with self.assertRaises(ValueError):
            mad.generate("copying", 2)

    def test_determinism(self):
        a = mad.fuzzy_recall(3, seed=5)
        b = mad.fuzzy_recall(3, seed=5)
        torch.testing.assert_close(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
