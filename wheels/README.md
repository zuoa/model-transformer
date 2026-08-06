# rknn-toolkit2 wheel

`rknn-toolkit2` is **not on PyPI**. Download the v2.3.2 **cp312** wheel for
x86_64 Linux and place it in this directory before building the Docker image
**locally**.

> The CI workflow (`.github/workflows/docker.yml`) fetches this wheel
> automatically via a sparse checkout of `airockchip/rknn-toolkit2`, so manual
> placement is only needed for local builds.

```
wheels/
└── rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

## Where to get it

Source repo: <https://github.com/airockchip/rknn-toolkit2> (the
`rockchip-linux/rknn-toolkit2` fork is archived/moved).

The wheel lives under:

```
rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
```

Clone and copy:

```bash
git clone https://github.com/airockchip/rknn-toolkit2.git /tmp/rknn
cp /tmp/rknn/rknn-toolkit2/packages/x86_64/rknn_toolkit2-2.3.2-cp312-cp312-*.whl ./wheels/
```

Verify against the repo's `packages/x86_64/md5sum.txt`.

## Notes

- The image still builds **without** the wheel — only the RKNN pipelines are
  disabled; the PT→ONNX path keeps working.
- This cp312 wheel runs only on **x86_64 Linux**. On macOS dev, build/run the
  image under Docker; RKNN conversion cannot run natively.
