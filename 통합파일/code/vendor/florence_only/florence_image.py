"""Reuse one image's processing and visual features across Florence prompts.

Scoped to one immutable PIL image and one eval model. Text processing, language
encoding and token generation still run for every prompt. This helper follows
the pinned Florence-2 generate implementation's inputs_embeds entry point.
"""
import copy


class _ImageProcessor:
    def __init__(self, processor, image):
        self.processor = processor
        self.image = image
        self.pixels = None
        self.options = None
        self.calls = 0

    def __call__(self, images, **options):
        if images is not self.image:
            raise ValueError('Per-image cache received a different image')
        if self.pixels is None:
            self.pixels = self.processor(images, **options)['pixel_values'].to('cuda')
            self.options = options.copy()
            self.calls += 1
        elif options != self.options:
            raise ValueError('Image processing options changed within one image')
        return {'pixel_values': self.pixels}

    def __getattr__(self, name):
        return getattr(self.processor, name)


class PerImageGeneration:
    def __init__(self, model, processor, image):
        if model.training:
            raise ValueError('Image feature reuse requires model.eval()')
        self.model = model
        self.image = image
        self.processor = copy.copy(processor)
        self.image_processor = _ImageProcessor(processor.image_processor, image)
        self.processor.image_processor = self.image_processor
        self.features = None
        self.feature_calls = 0
        self.generation_calls = 0

    def generate(self, prompt, **options):
        import torch
        if self.model.training:
            raise ValueError('Image feature reuse requires model.eval()')
        inputs = self.processor(text=prompt, images=self.image, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            # Preserve the original text embedding -> image encoding -> merge
            # order on the first call; only the visual branch is image-only.
            text = self.model.get_input_embeddings()(inputs['input_ids'])
            if self.features is None:
                self.features = self.model._encode_image(inputs['pixel_values'])
                self.feature_calls += 1
            merged, _ = self.model._merge_input_ids_with_image_features(self.features, text)
            output = self.model.generate(input_ids=None, inputs_embeds=merged, **options)
        self.generation_calls += 1
        return output

    def statistics(self):
        return dict(image_processing_calls=self.image_processor.calls,
                    image_encoding_calls=self.feature_calls,
                    generation_calls=self.generation_calls)
