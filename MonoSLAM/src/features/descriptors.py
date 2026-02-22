# src/features/descriptors.py

import numpy as np


# Make BRIEF sampling pairs for a given patch size
def brief_make_pairs(patch_size, *, n_bits=256, seed=0, sigma=None, dtype=np.int16):

    # --- Checks ---
    # Require integer patch_size
    if not isinstance(patch_size, (int, np.integer)):
        raise ValueError(f"patch_size must be int; got {type(patch_size)}")
    # Cast
    P = int(patch_size)
    # Require odd and >= 3
    if P < 3 or (P % 2) != 1:
        raise ValueError(f"patch_size must be odd and >= 3; got {P}")

    # Require integer n_bits
    if not isinstance(n_bits, (int, np.integer)):
        raise ValueError(f"n_bits must be int; got {type(n_bits)}")
    # Cast
    B = int(n_bits)
    # Require positive
    if B <= 0:
        raise ValueError(f"n_bits must be > 0; got {B}")

    # Require integer seed
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be int; got {type(seed)}")
    # Cast
    seed = int(seed)

    # Patch radius in centred coords
    c = P // 2

    # Default sigma (ORB/BRIEF-style: concentrate comparisons near the centre)
    if sigma is None:
        sigma = float(P) / 5.0
    else:
        sigma = float(sigma)
        if (not np.isfinite(sigma)) or sigma <= 0.0:
            raise ValueError(f"sigma must be finite and > 0; got {sigma}")

    # RNG
    rng = np.random.default_rng(seed)

    # Sample (x1,y1,x2,y2) from N(0, sigma^2)
    pairs = rng.normal(loc=0.0, scale=sigma, size=(B, 4))

    # Round to integer pixel offsets in centred coords
    pairs = np.rint(pairs).astype(dtype, copy=False)

    # Clip to patch bounds so indexing stays safe pre-rotation
    pairs = np.clip(pairs, -c, c).astype(dtype, copy=False)

    # Return shape (B,4): [x1,y1,x2,y2] in centred coords
    return pairs


# Compute ORB-style orientation per patch using intensity centroid moments
def brief_orientations_from_patches(patches, *, eps=1e-12, dtype=np.float64):

    # --- Checks ---
    # Require numpy array
    if not isinstance(patches, np.ndarray):
        raise ValueError("patches must be a numpy array")
    # Require (N,P,P)
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape (N,P,P); got {patches.shape}")
    # Require square patches
    N, P1, P2 = patches.shape
    if P1 != P2:
        raise ValueError(f"patches must be square; got {P1}x{P2}")
    # Require odd size
    if (P1 % 2) != 1:
        raise ValueError(f"patch size must be odd; got {P1}")

    # Validate eps
    eps = float(eps)
    if (not np.isfinite(eps)) or eps <= 0.0:
        raise ValueError(f"eps must be finite and > 0; got {eps}")

    # Cast patches to float for moment computation
    X = np.asarray(patches, dtype=dtype)

    # Patch radius
    c = P1 // 2

    # Build centred coordinate grids (P,P)
    xs = np.arange(-c, c + 1, dtype=dtype)
    ys = np.arange(-c, c + 1, dtype=dtype)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    # Compute first moments m10 = sum(x * I), m01 = sum(y * I)
    m10 = (X * XX[None, :, :]).sum(axis=(1, 2))
    m01 = (X * YY[None, :, :]).sum(axis=(1, 2))

    # Angle = atan2(m01, m10)
    ang = np.arctan2(m01, m10)

    # If centroid signal is tiny, set angle to 0 (stable default)
    mag2 = (m10 * m10) + (m01 * m01)
    weak = mag2 < (eps * eps)
    ang[weak] = 0.0

    # Return angles in radians, shape (N,)
    return ang.astype(dtype, copy=False)


# Compute BRIEF descriptors from patches
def brief_from_patches(
    patches,
    pairs,
    *,
    angles=None,
    packbits=True,
    bitorder="little",
    dtype=np.float64,
):

    # --- Checks ---
    # Require numpy patches
    if not isinstance(patches, np.ndarray):
        raise ValueError("patches must be a numpy array")
    # Require (N,P,P)
    if patches.ndim != 3:
        raise ValueError(f"patches must have shape (N,P,P); got {patches.shape}")
    # Require numpy pairs
    if not isinstance(pairs, np.ndarray):
        raise ValueError("pairs must be a numpy array")
    # Require (B,4)
    if pairs.ndim != 2 or pairs.shape[1] != 4:
        raise ValueError(f"pairs must have shape (B,4); got {pairs.shape}")

    # Read dims
    N, P1, P2 = patches.shape
    B = int(pairs.shape[0])

    # Require square patches
    if P1 != P2:
        raise ValueError(f"patches must be square; got {P1}x{P2}")
    # Require odd patch size
    if (P1 % 2) != 1:
        raise ValueError(f"patch size must be odd; got {P1}")

    # Validate bitorder
    bitorder = str(bitorder).lower()
    if bitorder not in {"little", "big"}:
        raise ValueError(f"bitorder must be 'little' or 'big'; got {bitorder}")

    # Cast patches to float for comparisons
    P = np.asarray(patches, dtype=dtype)

    # Patch radius / centre offset
    c = P1 // 2

    # Extract centred coords for the BRIEF pairs
    x1 = pairs[:, 0].astype(dtype, copy=False)
    y1 = pairs[:, 1].astype(dtype, copy=False)
    x2 = pairs[:, 2].astype(dtype, copy=False)
    y2 = pairs[:, 3].astype(dtype, copy=False)

    # If no angles provided, we do axis-aligned BRIEF
    if angles is None:

        # Convert centred coords to integer pixel indices
        u1 = (x1 + float(c)).astype(np.int64, copy=False)
        v1 = (y1 + float(c)).astype(np.int64, copy=False)
        u2 = (x2 + float(c)).astype(np.int64, copy=False)
        v2 = (y2 + float(c)).astype(np.int64, copy=False)

        # Build patch index helper
        ii = np.arange(N, dtype=np.int64)[:, None]

        # Gather intensities
        I1 = P[ii, v1[None, :], u1[None, :]]
        I2 = P[ii, v2[None, :], u2[None, :]]

        # Bit = 1 if I1 < I2
        bits = (I1 < I2)

    else:

        # Require angles shape (N,)
        ang = np.asarray(angles, dtype=dtype)
        if ang.ndim != 1 or ang.shape[0] != N:
            raise ValueError(f"angles must have shape (N,); got {ang.shape}")

        # Compute cos/sin per patch for rotation
        cs = np.cos(ang)[:, None]
        sn = np.sin(ang)[:, None]

        # Rotate pair coords for each patch: [x';y'] = R(theta)[x;y]
        x1r = cs * x1[None, :] - sn * y1[None, :]
        y1r = sn * x1[None, :] + cs * y1[None, :]
        x2r = cs * x2[None, :] - sn * y2[None, :]
        y2r = sn * x2[None, :] + cs * y2[None, :]

        # Map to pixel indices (nearest neighbour)
        u1 = np.rint(x1r + float(c)).astype(np.int64)
        v1 = np.rint(y1r + float(c)).astype(np.int64)
        u2 = np.rint(x2r + float(c)).astype(np.int64)
        v2 = np.rint(y2r + float(c)).astype(np.int64)

        # Valid masks (inside patch bounds)
        ok1 = (u1 >= 0) & (u1 < P1) & (v1 >= 0) & (v1 < P1)
        ok2 = (u2 >= 0) & (u2 < P1) & (v2 >= 0) & (v2 < P1)
        ok = ok1 & ok2

        # Clip for safe indexing
        u1c = np.clip(u1, 0, P1 - 1)
        v1c = np.clip(v1, 0, P1 - 1)
        u2c = np.clip(u2, 0, P1 - 1)
        v2c = np.clip(v2, 0, P1 - 1)

        # Patch index helper
        ii = np.arange(N, dtype=np.int64)[:, None]

        # Gather intensities (clipped indices)
        I1 = P[ii, v1c, u1c]
        I2 = P[ii, v2c, u2c]

        # Bit = 1 if I1 < I2
        bits = (I1 < I2)

        # If either point went out of bounds, force bit to 0
        bits = bits & ok

    # Optionally pack bits into uint8 descriptors
    if bool(packbits):

        # Pack into bytes along axis=1
        desc = np.packbits(bits.astype(np.uint8, copy=False), axis=1, bitorder=bitorder)

        # Return packed descriptors, shape (N, ceil(B/8)), dtype uint8
        return desc

    # Return unpacked bits, shape (N,B), dtype uint8
    return bits.astype(np.uint8, copy=False)
