# mamba-mini
An efficient implementation of selective scan in one file, works with both cpu and gpu, with corresponding mathematical derivation. It is probably the code which is the most close to selective_scan_cuda in mamba.

### mathematical derivation
![image](../assets/derivation.png)

### code
```python
import torch
def selective_scan_easy(us, dts, As, Bs, Cs, Ds, delta_bias=None, delta_softplus=False, return_last_state=False, chunksize=64):
    """
    # B: batch_size, G: groups, D: dim, N: state dim, L: seqlen
    us: B, G * D, L 
    dts: B, G * D, L
    As: G * D, N
    Bs: B, G, N, L
    Cs: B, G, N, L
    Ds: G * D
    delta_bias: G * D
    # chunksize can be any as you like. But as the chunksize raises, hs may get None, as exp(sum(delta) A) is really small
    """
    def selective_scan_chunk(us, dts, As, Bs, Cs, hprefix):
        """
        partial(h) / partial(t) = Ah + Bu; y = Ch + Du;
        => partial(h*exp(-At)) / partial(t) = Bu*exp(-At);
        => h_t = h_0 + sum_{0}_{t}_{Bu*exp(A(t-v)) dv};
        => h_b = exp(A(dt_a + ... + dt_{b-1})) * (h_a + sum_{a}_{b-1}_{Bu*exp(-A(dt_a + ... + dt_i)) dt_i});
           y_i = C_i*h_i + D*u_i
        """
        """
        us, dts: (L, B, G, D) # L is chunk_size
        As: (G, D, N)
        Bs, Cs: (L, B, G, N)
        Ds: (G, D)
        hprefix: (B, G, D, N)
        """
        ts = dts.cumsum(dim=0)
        Ats = torch.einsum("gdn,lbgd->lbgdn", As, ts).exp()
        scale = Ats[-1].detach()
        rAts = Ats / scale
        duts = dts * us
        dtBus = torch.einsum("lbgd,lbgn->lbgdn", duts, Bs)
        hs_tmp = rAts * (dtBus / rAts).cumsum(dim=0) 
        hs = hs_tmp + Ats * hprefix.unsqueeze(0)
        ys = torch.einsum("lbgn,lbgdn->lbgd", Cs, hs) 
        return ys, hs
    
    inp_dtype = us.dtype
    has_D = Ds is not None

    dts = dts.float()
    if delta_bias is not None:
        dts = dts + delta_bias.view(1, -1, 1).float()
    if delta_softplus:
        dts = torch.nn.functional.softplus(dts)
    
    if len(Bs.shape) == 3:
        Bs = Bs.unsqueeze(1)
    if len(Cs.shape) == 3:
        Cs = Cs.unsqueeze(1)
    B, G, N, L = Bs.shape
    us = us.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    dts = dts.view(B, G, -1, L).permute(3, 0, 1, 2).float()
    As = As.view(G, -1, N).float()
    Bs = Bs.permute(3, 0, 1, 2).float()
    Cs = Cs.permute(3, 0, 1, 2).float()
    Ds = Ds.view(G, -1).float() if has_D else None
    D = As.shape[1]
    
    oys = []
    # ohs = []
    hprefix = us.new_zeros((B, G, D, N), dtype=torch.float)
    for i in range(0, L - 1, chunksize):
        ys, hs = selective_scan_chunk(
            us[i:i + chunksize], dts[i:i + chunksize], 
            As, Bs[i:i + chunksize], Cs[i:i + chunksize], hprefix, 
        )
        oys.append(ys)
        # ohs.append(hs)
        hprefix = hs[-1]

    oys = torch.cat(oys, dim=0)
    # ohs = torch.cat(ohs, dim=0)
    if has_D:
        oys = oys + Ds * us
    oys = oys.permute(1, 2, 3, 0).view(B, -1, L)
    oys = oys.to(inp_dtype)
    # hprefix = hprefix.to(inp_dtype)

    return oys if not return_last_state else (oys, hprefix.view(B, G * D, N))

```

## Installing the CUDA `selective_scan` kernels

This directory builds the fused CUDA kernels `selective_scan_cuda_core`,
`selective_scan_cuda_ndstate`, and `selective_scan_cuda_oflex`. The steps below
target the project venv at `/mnt/store/bwarnas1/SOTA-CD/.libs/mambafcs`, whose
PyTorch is built against **CUDA 13** (`torch 2.12.1+cu130`).

> A plain `pip install .` fails here for two reasons:
> 1. `setup.py` does `import torch` at top level but there is no `pyproject.toml`,
>    so pip's PEP 517 build isolation creates a fresh env without torch →
>    `ModuleNotFoundError: No module named 'torch'`.
> 2. The only system CUDA toolkit is 12.4, while torch is `cu130`. `nvcc` and
>    torch must share the same CUDA **major** version or the build aborts.

Activate the venv first, then run the steps below from this directory
(`MambaFCS/kernels/selective_scan`).

### 1. Add a CUDA 13 toolkit inside the venv (no root needed)

torch already installs the CUDA 13 runtime + headers as pip wheels under
`.../site-packages/nvidia/cu13`. Add the matching compiler, the CCCL
(CUB/Thrust) headers, and ninja:

```bash
pip install "cuda-toolkit[nvcc,cccl]==13.0.2" ninja
# Pin nvvm + crt to the SAME version as nvcc (13.0.88). Otherwise pip grabs the
# newer 13.3 front-end, which emits PTX the 13.0 ptxas can't assemble
# ("ptxas fatal: Unsupported .version 9.3; current version is '9.0'").
pip install "nvidia-nvvm==13.0.88" "nvidia-cuda-crt==13.0.88"
```

### 2. Add the unversioned `libcudart.so` symlink

The runtime wheel ships only `libcudart.so.13`, but the linker needs an
unversioned `libcudart.so` to satisfy `-lcudart`:

```bash
CU=$(python -c "import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),'cu13'))")
ln -sf libcudart.so.13 "$CU/lib/libcudart.so"
```

### 3. Build and install

```bash
CU=$(python -c "import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),'cu13'))")
export CUDA_HOME="$CU"
export PATH="$CU/bin:$PATH"
export LD_LIBRARY_PATH="$CU/lib:$LD_LIBRARY_PATH"
export MAX_JOBS=4            # cap parallel nvcc jobs to limit RAM use

pip install . --no-build-isolation
```

`--no-build-isolation` is what makes the build reuse the venv's torch and the
nvcc installed in step 1.

### CUDA 13 source changes (already applied in this repo)

Building against CUDA 13 / CCCL 3.0 required two small edits that are already
committed here — listed so they can be redone on a fresh upstream checkout or
re-targeted to a different GPU:

- **`setup.py`** — removed the hardcoded `compute_70` (Volta) gencode, which
  CUDA 13 no longer supports, and now targets `sm_80` + `sm_86` (this box is
  RTX A5000 = sm_86). Edit the `-gencode` flags to match your GPU.
- **`csrc/selective_scan/reverse_scan.cuh`** — CUB 3.0 removed the internal
  helpers `cub::LaneId()` and `cub::CTA_SYNC()`; a small `CUB_VERSION >= 300000`
  shim re-adds them with their original semantics.

### Runtime

If importing a kernel raises `libcudart.so.13: cannot open shared object file`,
point the loader at the wheel's CUDA libs:

```bash
CU=$(python -c "import os,nvidia;print(os.path.join(os.path.dirname(nvidia.__file__),'cu13'))")
export LD_LIBRARY_PATH="$CU/lib:$LD_LIBRARY_PATH"
```

### to test
```bash
pytest test_selective_scan.py
```
