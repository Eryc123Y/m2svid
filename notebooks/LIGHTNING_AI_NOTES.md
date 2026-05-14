# Running M2SVid notebooks on Lightning AI Studio

This repo has two local notebooks:

- `m2svid_colab_tiny_demo.ipynb` — refinement-only small demo using repo-precomputed warped video/mask.
- `m2svid_colab_full_inference.ipynb` — full pipeline: DepthCrafter -> warping -> M2SVid.

For Lightning AI Studio, prefer the same notebooks but change Colab-specific paths:

- remove `from google.colab import drive; drive.mount(...)`;
- use a persistent Studio path such as `/teamspace/studios/this_studio/m2svid_demo` or `/teamspace/studios/this_studio/m2svid_full_inference` instead of `/content/drive/MyDrive/...`;
- clone or upload this repo inside the Studio, then open notebooks from the Studio web notebook or connect local VSCode via SSH.

Recommended GPU: L4 or A100 for first full run; T4 may be too tight.

Typical upload choices:

1. Push this local repo/branch to GitHub, then clone it in Lightning Studio.
2. Use Lightning Studio's web UI/Drive upload for `.ipynb`, input MP4s, or checkpoints.
3. Use Lightning CLI upload if configured.
4. Use SSH/Remote-SSH and `scp`/`rsync` if local IDE access is enabled.

Do not upload large checkpoints via git. Download them from inside Studio or upload to Studio Drive/persistent storage.
