import unittest


class PublicApiTest(unittest.TestCase):
    def test_exports(self):
        import thetamem

        for name in thetamem.__all__:
            self.assertTrue(hasattr(thetamem, name), msg=name)
        self.assertTrue(
            {"key_part", "normalize", "replay_fit", "replay_read"}
            <= set(thetamem.__all__)
        )

    def test_version(self):
        import thetamem

        self.assertEqual(thetamem.__version__, "0.1.0")

    def test_quickstart(self):
        import torch

        import thetamem

        layer = thetamem.ThetaMemLayer(
            64, heads=2, key_dim=16, value_dim=32, state="outer", chunk=8
        )
        out = layer(torch.randn(1, 12, 64))
        self.assertEqual(out.shape, (1, 12, 64))


if __name__ == "__main__":
    unittest.main()
