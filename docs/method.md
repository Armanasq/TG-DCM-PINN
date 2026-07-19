# TG-DCM-PINN method

TG-DCM estimates orientation from IMU streams using a 2-layer GRU.

Input at each time step:

```text
[acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z, dt]
```

Output at each time step:

```text
[w, x, y, z]
```

The output quaternion uses scalar-first Hamilton convention and maps body frame
to world frame.

The training objective combines:

- supervised yaw-invariant tilt error
- target-gated DCM vertical-direction dynamics consistency
- gravity-direction anchoring under a motion gate

The DCM term is applied only when the target transition is consistent with
gyroscope propagation. This avoids enforcing dynamics on frames where the target
orientation is missing or inconsistent.

The released checkpoint stores raw and EMA tensors with primitive metadata.
Evaluation loads it with `weights_only=True` and uses EMA by default.
