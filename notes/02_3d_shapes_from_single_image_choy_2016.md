# 02 — 3D-R2N2: Reconstructing using a recurrent shape prior

**Choy, Gwak, Savarese — "3D-R2N2: A Unified Approach for Single and Multi-view 3D
Object Reconstruction" (ECCV 2016)** — arXiv:1604.00449

## Background

The immediate ancestor of Pix2Vox. It treats the *sequence of views* as a recurrent
problem: a ConvLSTM that conditions an encoder-decoder on every image it sees.

```
for each of N views:
    image -> CNN encoder -> feature
    ConvLSTM cell updates its state
after the last view:
    3D conv decoders -> 32³ volume (or 128³ with higher resolution)
```

Key advantages over a plain single-view CNN:
- The **LSTM state is a generative shape prior** — the network owns a memory of
  what the 3D shape should look like, and each new view refines it.
- A single network handles **variable numbers of views** (1..~20) at test time.
- It handles depth uncertainty across views naturally (multi-view "fusion").

Why we care:
- Our `NeRF oracle` reconstructs with **20 training views**, the ceiling of what
  multi-view can do. 3D-R2N2 is what sits between that and the 1-view models:
  a network that accepts any number of views.

## Take-aways for our pipeline

- Our Pix2Vox supports `--n-views N`. Training with 2–4 views per object combines
  the ideas of R2N2 (recurrence/attention over views) with Pix2Vox's merger.
- With long-tailed classes, multi-view supervision is often what single-shot methods
  lack most — every object now contributes *more* pixels of supervision.
- **Action:** run Pix2Vox with `--n-views 3` and `--n-views 5` for one category set
  and report both micro and macro; we already trained a 250-epoch reference.

## My notes

(add after reading)
