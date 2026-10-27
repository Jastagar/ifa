import unittest
from unittest.mock import patch

from ifa.utils.devices import resolve_device


class ResolveDeviceTests(unittest.TestCase):
    def check(self, choice, cuda, mps, auto_mps=True):
        with patch.dict("os.environ", {"X_DEVICE": choice}), \
                patch("torch.cuda.is_available", return_value=cuda), \
                patch("torch.backends.mps.is_available", return_value=mps):
            return resolve_device("X_DEVICE", auto_mps=auto_mps)

    def test_auto_prefers_cuda_then_apple_gpu_then_cpu(self):
        self.assertEqual(self.check("auto", cuda=True, mps=False), "cuda")
        self.assertEqual(self.check("auto", cuda=False, mps=True), "mps")
        self.assertEqual(self.check("auto", cuda=False, mps=True, auto_mps=False), "cpu")
        self.assertEqual(self.check("auto", cuda=False, mps=False), "cpu")

    def test_unavailable_choice_falls_back_to_cpu(self):
        self.assertEqual(self.check("cuda", cuda=False, mps=True), "cpu")
        self.assertEqual(self.check("mps", cuda=False, mps=False), "cpu")
        self.assertEqual(self.check("mps", cuda=False, mps=True, auto_mps=False), "mps")
