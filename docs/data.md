# Data

The evaluator reads two HDF5 schemas.

Standard RIANN schema:

| Channel | Unit | Shape |
|---|---|---|
| `acc_x`, `acc_y`, `acc_z` | m/s^2 | `(T,)` |
| `gyr_x`, `gyr_y`, `gyr_z` | rad/s | `(T,)` |
| `opt_a`, `opt_b`, `opt_c`, `opt_d` | unit quaternion, wxyz | `(T,)` |
| `dt` | s | `(T,)` |

DIODEM schema:

| Channel | Unit | Shape |
|---|---|---|
| `acc` | m/s^2 | `(T, 3)` |
| `gyr` | rad/s | `(T, 3)` |
| `quat` | unit quaternion, wxyz | `(T, 4)` |
| `timestamps` | s | `(T,)` |

Expected repository-local layout:

```text
TG-DCM-PINN/
  data/
    Myon/
    TUM-VI/
    OxIOD/
    EuRoC-MAV/
    RepoIMU/
    Caruso-Sassari/
    Caruso-Sassari_orig/
  diodem/
    diodem/
      exp01/
      ...
```

`scripts/prepare_data.py` prepares the standard RIANN datasets. DIODEM support
uses the prepared HDF5 segment files shown above.
