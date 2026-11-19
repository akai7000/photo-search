import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np


class ClipEngineTests(unittest.TestCase):
    def test_embeddings_use_cpu_even_when_acceleration_is_available(self):
        torch = MagicMock()
        torch.get_num_threads.return_value = 2
        torch.cuda.is_available.return_value = True
        open_clip = MagicMock()
        model = MagicMock()
        preprocess = MagicMock()
        image_tensor = MagicMock()
        tokens = MagicMock()
        embedding = MagicMock()
        embedding.cpu.return_value.numpy.return_value = np.array([[0.6, 0.8]], dtype=np.float32)
        model.encode_image.return_value = embedding
        model.encode_text.return_value = embedding
        preprocess.return_value.unsqueeze.return_value = image_tensor
        open_clip.create_model_and_transforms.return_value = (model, None, preprocess)
        open_clip.get_tokenizer.return_value.return_value = tokens
        source = Path(__file__).resolve().parents[1] / "core" / "clip_engine.py"
        spec = importlib.util.spec_from_file_location("test_clip_engine_module", source)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"torch": torch, "open_clip": open_clip}), \
             patch("core.startup.report"), patch.dict("os.environ", {"OMP_NUM_THREADS": "1"}):
            spec.loader.exec_module(module)
            with patch.object(module.Image, "open") as image_open:
                module.get_image_embedding("photo.jpg")
                image_open.assert_called_once_with("photo.jpg")
            module.get_text_embedding("example")
        model.to.assert_called_once_with("cpu")
        image_tensor.to.assert_called_once_with("cpu")
        tokens.to.assert_called_once_with("cpu")
        torch.cuda.is_available.assert_not_called()


if __name__ == "__main__":
    unittest.main()
