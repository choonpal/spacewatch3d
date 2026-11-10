import contextlib
import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path

from spacewatch_pipeline.io import ROOT
sys.path.insert(0, str(ROOT/'vendor/florence_only'))
from florence_image import PerImageGeneration, _ImageProcessor


class Pixels:
    def __init__(self, value): self.value = value
    def to(self, device): return self


class Batch(dict):
    def to(self, device): return self


class ImageProcessor:
    def __init__(self): self.calls = 0
    def __call__(self, image, **options):
        self.calls += 1
        return {'pixel_values': Pixels(image)}


class Processor:
    def __init__(self): self.image_processor = ImageProcessor()
    def __call__(self, text, images, return_tensors):
        return Batch(input_ids=text, **self.image_processor(images, return_tensors=return_tensors))


class Model:
    training = False
    def __init__(self): self.encodes = 0
    def get_input_embeddings(self): return lambda text: text
    def _encode_image(self, pixels):
        self.encodes += 1
        return pixels.value
    def _merge_input_ids_with_image_features(self, features, text): return (features, text), None
    def generate(self, input_ids, inputs_embeds, **options):
        assert input_ids is None
        return inputs_embeds, options


class ImageCacheTests(unittest.TestCase):
    def setUp(self):
        self.torch = patch.dict(sys.modules, torch=types.SimpleNamespace(inference_mode=contextlib.nullcontext))
        self.torch.start(); self.addCleanup(self.torch.stop)

    def test_prompt_changes_keep_visual_branch_and_generation_options(self):
        model, processor, image = Model(), Processor(), object()
        cache = PerImageGeneration(model, processor, image)
        for prompt in ['<OD>', '<REGION_TO_SEGMENTATION><loc_1>', '<REFERRING_EXPRESSION_SEGMENTATION>chair']:
            value = cache.generate(prompt, num_beams=1, max_new_tokens=1024, do_sample=False)
            self.assertEqual(value, ((image, prompt), dict(num_beams=1, max_new_tokens=1024, do_sample=False)))
        self.assertEqual(cache.statistics(), dict(image_processing_calls=1,image_encoding_calls=1,generation_calls=3))
        self.assertEqual(model.encodes, 1)
        self.assertIsInstance(processor.image_processor, ImageProcessor)

    def test_next_image_has_its_own_visual_features(self):
        model, processor = Model(), Processor()
        a, b = object(), object()
        x = PerImageGeneration(model, processor, a).generate('same')
        y = PerImageGeneration(model, processor, b).generate('same')
        self.assertIs(x[0][0], a); self.assertIs(y[0][0], b)
        self.assertEqual(model.encodes, 2)
        self.assertEqual(processor.image_processor.calls, 2)

    def test_training_model_rejected_on_creation_and_later(self):
        model, processor = Model(), Processor()
        cache = PerImageGeneration(model, processor, object())
        model.training = True
        with self.assertRaises(ValueError): cache.generate('x')
        with self.assertRaises(ValueError): PerImageGeneration(model, processor, object())

    def test_different_image_or_preprocessing_options_rejected(self):
        image = object(); cache = _ImageProcessor(ImageProcessor(), image)
        cache(image, return_tensors='pt')
        with self.assertRaises(ValueError): cache(object(), return_tensors='pt')
        with self.assertRaises(ValueError): cache(image, return_tensors='np')


if __name__ == '__main__': unittest.main(verbosity=2)
