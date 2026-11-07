# Reconstruction lab app

`index.html` is a browser control panel for the three reconstruction
implementations in `ashry`. `server.py` launches the actual PyTorch trainers
and the page displays their real output artifacts. It provides:

- the native representation and mesh conversion route for each method;
- held-out prediction images from `heldout/`;
- the matching benchmark ground-truth image for the selected camera view;
- point-cloud extraction from each completed checkpoint;
- Chamfer, precision, recall, and F1 comparison against the normalized
  benchmark point cloud. Metrics are reported at both 1% and 2% distance
  thresholds (`precision_at_0.01`, `recall_at_0.01`, `f1_at_0.01` and the
  corresponding `*_at_0.02` values).

Open it directly in a browser:

```bash
cd ashry
python app/server.py
```

Then visit <http://localhost:8000>. Choose an object, model, iteration count,
and image size, then click **Start GPU training**. Logs and the final
`checkpoint.pt`, `heldout/`, and model-specific artifacts are written under
`ashry/{3dgs,nerf,neus}/results/app/`.

Click **Generate point cloud + compare** after a completed job. 3DGS uses Gaussian centers and writes them as a solid cyan comparison cloud.
The benchmark comparison targets the reference point count when enough
Gaussians are available. NeRF samples its density field; NeuS samples its SDF
zero level set. The generated PLY and JSON metrics are saved beside the
checkpoint. This is a point-cloud view, not a triangle mesh; exporting a
watertight textured mesh requires a separate surface extraction and
UV-texturing step.

The app also provides **Show textured surface** for completed 3DGS jobs. It
extracts a triangle surface from the learned Gaussian density and assigns
vertex colors from the learned Gaussian appearance, then renders those
triangles in the browser. This is a browser inspection mesh rather than a
production UV-textured GLB; it is intended to show the reconstructed surface
without the stray-center appearance of a raw Gaussian point cloud.
Before comparison, the reconstructed cloud is translated and uniformly
rescaled to the reference's robust 1%-99% spatial extent. This removes the
global scale/translation ambiguity of camera-only reconstruction while
preventing a few stray Gaussian centers from moving the whole comparison.

The server is intentionally local-only (`127.0.0.1`) and allows only the
existing three trainer entry points. It does not turn one arbitrary image into
a model: the selected object still uses its calibrated multi-view dataset and
`transforms.json`, exactly like the command-line trainers.
