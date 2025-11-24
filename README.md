# Festivity Mapping with DINOv3 🎄🦖


Goal - given an image taken from the street at night, possibly containing a house, possibly decorated with festive lights, provide a "festivity measure" in order to sort the most festive images from the least.

Then take this model, drive around the neighbourhood continuously taking images, and automatically detect and rank the most festive houses. Perhaps produce a map of the most festive areas in the neigbhborhood 📍🎅.

<div style="display: flex; gap: 10px;">
    <img src="./doc/000000000004.jpg" alt="festive house" style="width: 49%;">
    <img src="./doc/000000000049.jpg" alt="not festive house" style="width: 49%;">
</div>

Which house is more festive?

Ask [Dino](https://ai.meta.com/dinov3/)! 🦖🦖🦖

## Previous approach (2024): CLIP

Previously I used [CLIP](https://openai.com/index/clip/) as a zero-shot classifier to derive per-image class probabilties for "festively lit house at night" vs "house at night". Then I used the festive class probability as the output metric.

This worked OK to detect festive houses, but was not great at quantifying just how festive a display is compared to another.

<figure>
  <img src="./doc/Screenshot%20from%202025-11-23%2018-12-48.png" alt="Grid of festive houses with questionable CLIP rankings">
  <figcaption>A selection of festively decorated houses, ranked in terms of festivity using clip (top-left = most festive). Clip has done a good job of detecting the festive houses (note most images in the dataset do not contain decorated houses). However the ranking is questionable - some lower effort displays are ranked above more impressive ones.</figcaption>
</figure>

Now it is 2025, dinov3 is out, I have a bit of time. Here I look at using dino patch embeddings to produce a dense fesitvity map, which I can agregate over the image to more acurately quantify festivity.

## New approach: DINOv3 segmentation

Basically, following the notebook from their repo [foreground_segmentation.ipynb](https://github.com/facebookresearch/dinov3/blob/main/notebooks/foreground_segmentation.ipynb) but modified to run on my own dataset. This takes images and associated foreground masks, and trains a linear classifier to distinguish foreground from dino patch embeddings. For my application the "foreground" class is "festive pixels", and an overall festivity score is produced by taking mean festivity across the whole image.

A small number of labelled images are required. I started with a couple, then added more ad-hoc as I encountered interesting failure cases in the output maps. At time of writing I have 15 labelled images for training, and I'm quite excited by the results.

I used [GIMP](https://www.gimp.org/) to draw the masks (labelling instructions [here](./labelling.md)). Most of the code is co-pilot generated, based on the linked notebook from the dino repo.

Train and test code is in [train_from_labelled_dataset.py](./train_from_labelled_dataset.py).

## Results

Here's what the dense output festivity map looks like for a few cases:

<figure>
  <img src="./doc/result_IMG_20241204_184614.png" alt="A festive scene with festivity map">
  <figcaption>A festive scene, along with festivity map.</figcaption>
</figure>

<figure>
  <img src="./doc/result_IMG_20241204_184517.png" alt="Street light with festivity map">
  <figcaption>Street lights are not festive.</figcaption>
</figure>

<figure>
  <img src="./doc/result_IMG_20241204_184647.png" alt="Security light with festivity map">
  <figcaption>Security lights are not festive.</figcaption>
</figure>

<figure>
  <img src="./doc/result_IMG_20241204_184531.png" alt="Inflatable dinosaur with festivity map">
  <figcaption>Inflatable dinosaurs are festive with probabilty <code>p=1.0</code></figcaption>
</figure>

<figure>
  <img src="./doc/result_IMG_20241204_190237.png" alt="Festive palm trees with festivity map">
  <figcaption>The most festive image in the dataset, by mean per-pixel festivity.</figcaption>
</figure>

<figure>
  <img src="./doc/Screenshot%20from%202025-11-23%2018-12-44.png" alt="Grid of festive images sorted by festivity score">
  <figcaption>A selection of images sorted by festivity (top left = most festive). In contrast to CLIP, the rankings are much more consistent/appropriate.</figcaption>
</figure>

Overall results are very promising, but with some potential areas to explore:
* Close-up displays take up more pixels than those further away, so achieve higher festivity scores. Consider re-weighting with depth-anything estimates.
* Should a bunch of decorated palm trees really win the "most festive" prize? How can we capture aspects of creativity, novelty, abilty to inspire awe, etc? Probably not with a per-pixel metric - try a VLM?


## Failed approach: dino.txt

I previously attempted a text-based approach more similar to what was done with CLIP - to compare patch embeddings with text embeddings to determine festive-ness. The idea was to produce the festivity measure zero-shot without labelling any data.

`dino.txt` is provided for this use case:
* [readme-ov-file#pretrained-heads---zero-shot-tasks-with-dinotxt](https://github.com/facebookresearch/dinov3/tree/main?tab=readme-ov-file#pretrained-heads---zero-shot-tasks-with-dinotxt)
* [dinotxt_inference.ipynb](https://github.com/facebookresearch/dinov3/blob/main/notebooks/dinotxt_inference.ipynb)

AFAIU, `dino.txt` adds a text encoder as well as some additional layers on the dino backbone, in order to project image and text features to the same embedding space, where they can be compared in the same way as CLIP embeddings.

Unfortunately this didn't work on my first attempt, so, limited on time, I decided to try the label based approach above. 

<figure>
  <img src="./doc/IMG_20241204_185627_similarity_combined.png" alt="Input image and similarity map for dino.txt approach">
  <figcaption>Input image and similarity map comparing patch embeddings to text embeddings for prompt "festive house decoration". Output doesn't appear at all correlated with festivity.</figcaption>
</figure>

I expect this is pure user-error, I just don't have time to debug 🤷.

## Next steps

* Reweight based on depth-anything depths
* Project onto map to produce festivity maps for the neighbourhood 
* Bigger dataset

