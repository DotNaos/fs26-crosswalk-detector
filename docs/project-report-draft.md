# Project Report Draft: Crosswalk Detection in Aerial Images

Status: 2026-05-18

## How to Read This Report

This report is written for two audiences at the same time:

- a reader who wants to evaluate whether the project satisfies the Deep Learning course requirements;
- a reader who does not yet know much about deep learning or PyTorch and wants to understand how the system works.

The most important idea is simple: the project learns from examples. It sees many aerial image tiles where crosswalks are either present or absent. During training, it adjusts internal numerical parameters until its predictions match the training labels as well as possible. After training, it can receive a new aerial image tile and predict whether it contains a crosswalk.

Our project goes one step further than simple classification. It does not only answer "yes" or "no". It also predicts a mask that shows where the crosswalk is likely located.

## 1. Project Goal

The goal of this project is to detect pedestrian crosswalks in `25 m x 25 m` aerial image tiles from Swisstopo/SWISSIMAGE. The official assignment asks for image classification: each tile should be classified as either `crosswalk` or `no_crosswalk`.

Our approach extends the required classification task with segmentation. Instead of only predicting a single class label, the model first predicts a pixel mask that marks the likely crosswalk area. The final image-level classification is then derived from this mask:

- if the predicted crosswalk mask is large enough, the tile is classified as `crosswalk`;
- if the predicted mask is empty or too small, the tile is classified as `no_crosswalk`.

This makes the result easier to inspect and explain. A pure classifier can only say that a crosswalk is present. Our model can additionally show where it believes the crosswalk is located.

## Project Overview in Plain English

The full system can be understood as a pipeline:

```text
Swisstopo aerial images
        |
        v
25 m x 25 m image tiles
        |
        v
SAM3 pseudo-labeling
        |
        v
training export with images, labels, and masks
        |
        v
CrossMaskNet training
        |
        v
prediction mask
        |
        v
final crosswalk / no_crosswalk decision
```

Each step has a specific role:

- Swisstopo provides the aerial imagery.
- The tiling step turns large map areas into small images.
- SAM3 creates initial labels and masks at scale.
- CrossMaskNet learns from those examples.
- The final prediction is produced by CrossMaskNet, not by SAM3.

This distinction matters for the semester project. SAM3 helps us build the dataset. The model we submit and explain is our own trained model.

## 2. Dataset

### 2.1 Image Source

The image source is Swisstopo/SWISSIMAGE aerial imagery. SWISSIMAGE is an official Swiss orthophoto mosaic built from digital color aerial photographs and provides high-resolution imagery over Switzerland.[^swissimage] The imagery is split into non-overlapping `25 m x 25 m` tiles, matching the size described in the course assignment. Each tile becomes one candidate image for the classification task.

The project uses a large generated dataset rather than a manually collected small dataset. This makes the training process more realistic because the model sees many different cities, road layouts, lighting conditions, roof structures, shadows, parking areas, and vegetation patterns.

### 2.2 What an Image Tile, Label, and Mask Mean

One training example contains three important parts:

| Part | Meaning |
|---|---|
| Image tile | The `25 m x 25 m` aerial image crop. |
| Label | The image-level answer: `crosswalk` or `no_crosswalk`. |
| Mask | A black-and-white image showing which pixels belong to the crosswalk. |

For classification alone, only the label would be needed. For our segmentation-first approach, the mask is also useful. It teaches the model not only that a crosswalk exists, but also where it is.

### 2.3 Label Generation

The labels were generated with SAM3 as a pseudo-labeling tool. Meta describes SAM3 as a model for detecting, segmenting, and tracking objects from concept prompts in images and videos.[^sam3] Earlier Segment Anything work introduced the broader idea of promptable segmentation and large-scale mask generation.[^sam] SAM3 is not the final model and is not part of the architecture submitted as our model. It is used only to create initial labels and mask suggestions at scale.

This is a deliberate dataset strategy:

- manual labeling of hundreds of thousands of aerial tiles would be too slow;
- SAM3 can identify likely crosswalk regions and provide useful segmentation masks;
- the generated labels can be filtered by confidence and inspected visually;
- difficult false positives can later be reused as hard negatives.

The final trained model is our own model, `CrossMaskNet`, trained from scratch on the exported dataset.

### 2.4 Large Dataset Scan

The current large dataset is called `sam3-500k-masks-v1`.

Key figures:

| Item | Count |
|---|---:|
| Scanned image tiles | `500,000` |
| Raw SAM3 crosswalk candidates with masks | `8,811` |
| Candidates with confidence `>= 0.4` | `6,284` |
| Candidates with confidence `>= 0.5` | `5,173` |
| Candidates with confidence `>= 0.6` | `3,044` |

The masks were validated so that mask artifacts are only attached to `crosswalk` labels. `no_crosswalk` labels do not carry mask files.

### 2.5 Training Export

For model training, a smaller CrossMaskNet export was created from the large pseudo-labeled dataset. The current v4 export contains `5,800` samples.

| Split | Crosswalk | No Crosswalk | Total |
|---|---:|---:|---:|
| Train | `1,600` | `3,406` | `5,006` |
| Validation | `189` | `193` | `382` |
| Test | `211` | `201` | `412` |
| Total | `2,000` | `3,800` | `5,800` |

The test split is kept stable across v1, v2, v3, and v4 so the model versions can be compared fairly.

### 2.6 Data Quality

The dataset is useful but not perfect. Since labels are generated automatically, some incorrect labels remain. Typical false positives are:

- roof structures that look like road markings;
- parking lot markings;
- shadows and high-contrast edges;
- vegetation patterns;
- road edges;
- lane markings that are not pedestrian crossings.

This weakness is not ignored. The project uses these errors as part of the optimization process. False positives are mined and added back into the training data as hard negatives, so later model versions learn to reject them.

## 3. Model Architecture

### 3.1 Minimal Deep Learning Background

A neural network is a function with many adjustable numbers, called parameters or weights. At the beginning of training, these numbers are not useful yet. The model makes poor predictions. Training means repeatedly showing examples to the model, measuring how wrong it is, and adjusting the weights to reduce future errors.

For this project:

- the input is an aerial image tile;
- the expected output is a crosswalk mask;
- the model prediction is another mask;
- the training error measures how different the predicted mask is from the expected mask.

PyTorch is the library used to implement and train the model. It provides tensors, which are multidimensional arrays that can run efficiently on GPUs, and automatic differentiation, which computes how model weights should change during training.[^pytorch-examples] PyTorch's `autograd` system is the part that calculates gradients for backpropagation.[^pytorch-autograd]

In simpler words: PyTorch handles the math needed to update the model after each mistake.

### 3.2 Overview

The model developed for this project is called `CrossMaskNet`. It is implemented in PyTorch and trained from scratch. It does not use pretrained weights.

Important distinction:

- SAM3 is used for pseudo-label generation.
- `CrossMaskNet` is the actual model developed and trained for this project.

`CrossMaskNet` is a compact U-Net-style segmentation model. U-Net is a well-known encoder-decoder architecture originally proposed for image segmentation, where precise localization matters.[^unet] CrossMaskNet is inspired by that idea, but it is implemented specifically for this project and trained from scratch on our crosswalk dataset. It is designed for small aerial image tiles where the target object is thin, structured, and visually ambiguous.

### 3.3 Input

Earlier versions use three input channels:

- red;
- green;
- blue.

Version v4 uses four input channels:

- red;
- green;
- blue;
- road-context mask.

The road-context mask is a generated helper channel that tells the model where road-like areas are likely to be. This is useful because crosswalks should usually appear on or near roads, not on roofs, grass, or trees.

### 3.4 Network Structure

`CrossMaskNet` follows an encoder-decoder structure:

1. The encoder progressively reduces the spatial resolution and learns higher-level image features.
2. A bridge layer processes the most compressed representation.
3. The decoder upsamples the representation back to image resolution.
4. Skip connections pass fine-grained details from the encoder to the decoder.
5. A final `1 x 1` convolution produces a one-channel crosswalk probability mask.

The output is a segmentation mask, not only a class label. Each output pixel represents the model's confidence that this pixel belongs to a crosswalk.

### 3.5 How One Image Moves Through CrossMaskNet

For one image tile, the model does the following:

1. It receives the image as numbers. For v4, this means four image-like channels: red, green, blue, and road context.
2. The encoder looks for increasingly abstract patterns. Early layers can react to simple visual cues such as edges, colors, or stripes. Deeper layers combine those cues into more meaningful structures.
3. The bridge processes the compressed representation.
4. The decoder reconstructs a full-size mask.
5. Skip connections bring back fine details that would otherwise be lost during compression.
6. The final layer outputs one value per pixel.
7. A sigmoid function converts those values into probabilities between 0 and 1.
8. A threshold converts probabilities into a visible binary mask.

This is why the model can show the likely crosswalk area instead of only producing a single yes/no output.

### 3.6 Classification From Segmentation

The assignment requires classification. The model produces segmentation, and classification is derived from it.

The rule is:

- calculate how much of the image is covered by the predicted crosswalk mask;
- compare that mask coverage against a threshold;
- classify the tile as `crosswalk` if the coverage is above the threshold.

This has two advantages:

- the model remains compatible with the classification task;
- the prediction is visually explainable because the mask shows the evidence.

### 3.7 Why This Counts as Our Own Architecture

The model is not a copy of SAM3 and does not call SAM3 during inference. It is a small PyTorch model written in this repository. The architecture choices are ours:

- number of encoder stages;
- number of decoder stages;
- base channel width;
- skip-connection layout;
- segmentation output head;
- optional fourth road-context input channel;
- classification rule derived from mask coverage.

It is fair to describe it as U-Net-style, because it uses the same broad encoder-decoder idea. It should not be described as a completely new research architecture. The correct professional framing is:

> CrossMaskNet is a custom compact U-Net-style segmentation model, implemented and trained from scratch for crosswalk detection in aerial image tiles.

## 4. Training Process

### 4.1 What Training Means in Practice

Training is a loop:

```text
take a batch of images
        |
        v
predict masks
        |
        v
compare predictions with target masks
        |
        v
compute loss
        |
        v
adjust model weights
        |
        v
repeat for many batches and epochs
```

An epoch means that the model has gone through the training dataset once. A batch is a smaller group of images processed together.

The validation set is used during training to decide which checkpoint is best. The test set is only used after training to estimate how well the model performs on held-out data.

### 4.2 Loss Function

The model is trained as a segmentation model. It uses a combination of:

- binary cross-entropy with logits;
- Dice-style overlap loss.

The binary cross-entropy term helps with pixel-level correctness. The Dice component helps the model learn useful masks even when crosswalk pixels cover only a small part of the image. Dice-style losses are commonly used for segmentation tasks with imbalanced foreground and background regions.[^dice]

### 4.3 Optimization

The model is optimized with AdamW and a cosine learning-rate schedule. AdamW is a variant of Adam that decouples weight decay from the gradient update, which is a common regularization strategy in modern deep learning.[^adamw] The cosine schedule gradually changes the learning rate during training.[^cosine] The training process stores the best checkpoint based on validation positive Dice score, not simply the final epoch.

This matters because later epochs do not always produce the best validation mask quality. Keeping the best validation checkpoint reduces overfitting.

### 4.4 Data Augmentation

The training loader applies light color jitter. This is useful for aerial images because brightness, contrast, and saturation can vary between locations and capture conditions.

The augmentation is intentionally moderate. Crosswalks are thin structured objects, so overly aggressive transformations could damage the visual features the model needs to learn.

## 5. Model Iterations

The model was improved through several iterations.

### 5.1 v1: First CrossMaskNet Baseline

The first version was trained on a balanced dataset of `4,000` samples:

- `2,000` crosswalk samples;
- `2,000` no-crosswalk samples.

It learned useful crosswalk localization, but produced too many false positives on roofs, vegetation, parking lots, shadows, and tile edges.

### 5.2 v2: Hard Negatives From v1

The second version added repeated hard-negative training rows mined from the strongest false positives of v1.

This improved precision and image-level accuracy. It showed that the main problem was not only model capacity, but also missing difficult negative examples.

### 5.3 v3: More Hard Negatives

The third version added more hard negatives mined from v2. This became the strongest image-level classifier so far.

v3 is currently the best default if the goal is to decide whether a tile contains a crosswalk.

### 5.4 v4: Road Context as Model Input

The fourth version adds road context as a fourth input channel.

This improves mask quality and makes the model more aware of plausible road areas. However, it does not outperform v3 on every image-level metric. The result is:

- v4 is best for segmentation quality;
- v3 is still slightly better as the default image-level classifier.

## 6. Results

All model versions below were evaluated on the same held-out test split.

| Model | Positive Dice | Positive IoU | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| CrossMaskNet v1 | `0.763920` | `0.618018` | `0.917476` | `0.870293` | `0.985782` |
| CrossMaskNet v2 | `0.768770` | `0.624392` | `0.944175` | `0.915929` | `0.981043` |
| CrossMaskNet v3 raw | `0.791034` | `0.654306` | `0.966019` | `0.971292` | `0.962085` |
| CrossMaskNet v4 raw | `0.794408` | `0.658937` | `0.961165` | `0.957746` | `0.966825` |

### 6.1 Interpretation

v1 proves that the architecture can learn the task, but it overpredicts crosswalk-like patterns.

v2 improves precision through hard negatives.

v3 gives the best image-level classification result. It reaches:

- accuracy: `0.966019`;
- precision: `0.971292`;
- recall: `0.962085`.

v4 gives the best segmentation result. It reaches:

- positive Dice: `0.794408`;
- positive IoU: `0.658937`.

The practical conclusion is that the project now has two strong candidates:

- v3 for final classification;
- v4 for final mask visualization and explainability.

### 6.2 Threshold Calibration

For v4, the image-level decision threshold was checked separately from mask training.

At mask coverage threshold `0.005`, v4 reaches:

- accuracy: `0.961165`;
- precision: `0.957746`;
- recall: `0.966825`.

At threshold `0.010`, false positives are reduced, and precision rises to `0.971014`, but recall drops to `0.952607`.

This means the threshold should be selected based on the desired behavior:

- lower threshold: fewer missed crosswalks;
- higher threshold: fewer false positives.

### 6.3 Metrics Explained

The report uses several metrics:

| Metric | Plain-English meaning |
|---|---|
| Accuracy | How often the final `crosswalk` / `no_crosswalk` decision is correct. |
| Precision | When the model says `crosswalk`, how often it is correct. High precision means fewer false alarms. |
| Recall | Of all real crosswalks, how many the model finds. High recall means fewer missed crosswalks. |
| Positive Dice | How well the predicted crosswalk masks overlap the target masks for positive samples. |
| Positive IoU | Another overlap metric for mask quality. It compares the intersection and union of predicted and target masks. |

Precision and recall often trade off against each other. A stricter threshold can reduce false positives and improve precision, but it can also miss subtle crosswalks and reduce recall.

## 7. Discussion

### 7.1 Why Segmentation Helps

The official task is classification, but segmentation is a useful extension. A binary class label alone does not show why the model made a decision. A mask makes the prediction inspectable.

This is especially important for aerial crosswalk detection because false positives can be visually plausible. For example, parking lines, roof structures, and shadows can resemble crosswalk stripes. With a mask, these errors are easier to understand and fix.

### 7.2 Why Hard Negatives Matter

The largest improvements came from adding hard negatives. This shows that the model needed more examples of what is not a crosswalk.

In this task, easy negatives are not enough. A tile showing only grass or water is simple. The difficult negatives are visually similar to crosswalks. Training on those examples makes the model more robust.

### 7.3 Role of SAM3

SAM3 is useful because it gives us a practical way to bootstrap a large labeled dataset. It should be described as a pseudo-label generator, not as the final solution.

The project remains a deep-learning project with our own trained model because:

- the final model architecture is implemented by us;
- the model is trained from scratch;
- SAM3 is not used during final inference;
- the evaluation compares our own model versions.

## 8. Limitations

The current system still has limitations:

- some pseudo-labels are wrong because they were generated automatically;
- very small or partially visible crosswalks can be missed;
- some road markings and parking structures can still trigger false positives;
- the final threshold has not yet been fixed for the submission;
- a human-reviewed subset should still be added for stronger final validation.

These limitations do not invalidate the project. They describe the current state honestly and point to realistic next improvements.

## 9. Personal Learnings

The most important learning is that dataset quality matters at least as much as model architecture. The early model already learned useful features, but it failed on difficult negatives. Improving the data distribution had a major effect.

The second learning is that segmentation can make a classification model more understandable. Instead of only receiving `crosswalk` or `no_crosswalk`, we can inspect the predicted mask and see which part of the image influenced the decision.

The third learning is that automatic labeling is powerful but should be handled carefully. SAM3 made it possible to create a large dataset quickly, but visual review and hard-negative mining are still important for reliability.

## 10. Current Submission Readiness

The project already satisfies the main technical requirements:

- it uses Swisstopo/SWISSIMAGE aerial images;
- it uses `25 m x 25 m` image tiles;
- it has a generated training, validation, and test dataset;
- it contains the code for the trained model;
- it evaluates the model on a held-out test split;
- it shows iterative optimization across several model versions.

Before final submission, the following should still be prepared:

- package or document the exact dataset export used for training;
- decide whether v3 or v4 is the final submitted model;
- include example images with predicted masks;
- complete a small human review of important failure cases;
- finalize this report into a concise two-page version if required.

## 11. Final Positioning

The strongest way to present the project is:

> The assignment asks for crosswalk classification on aerial image tiles. We solve this with a custom segmentation model trained from scratch. The segmentation mask is used to derive the required classification, and it also makes the decision explainable.

This positions the project as meeting the assignment while going beyond it in a useful and technically justified way.

## 12. References

[^swissimage]: Federal Office of Topography swisstopo. "SWISSIMAGE." Official product page for the digital color orthophoto mosaic of Switzerland. https://www.swisstopo.admin.ch/en/orthoimage-swissimage-10

[^sam3]: Meta AI. "SAM 3: Segment Anything with Concepts." Research publication page, 2025. https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/

[^sam]: Kirillov, A. et al. "Segment Anything." ICCV 2023 / arXiv:2304.02643. https://arxiv.org/abs/2304.02643

[^unet]: Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI 2015 / arXiv:1505.04597. https://arxiv.org/abs/1505.04597

[^pytorch-examples]: PyTorch Tutorials. "Learning PyTorch with Examples." https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html

[^pytorch-autograd]: PyTorch Tutorials. "Automatic Differentiation with torch.autograd." https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html

[^dice]: Sudre, C. H. et al. "Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations." arXiv:1707.03237. https://arxiv.org/abs/1707.03237

[^adamw]: Loshchilov, I. and Hutter, F. "Decoupled Weight Decay Regularization." ICLR 2019 / arXiv:1711.05101. https://arxiv.org/abs/1711.05101

[^cosine]: PyTorch Documentation. "CosineAnnealingLR." https://docs.pytorch.org/docs/2.12/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
