# Data

This repository does not redistribute OMBRIA data.

The training and evaluation scripts expect the public OMBRIA dataset under:

```text
external/OMBRIA/
```

Expected layout:

```text
external/OMBRIA/
  OmbriaS1/
    train/
      BEFORE/
      AFTER/
      MASK/
    test/
      BEFORE/
      AFTER/
      MASK/
  OmbriaS2/
    train/
      BEFORE/
      AFTER/
      MASK/
    test/
      BEFORE/
      AFTER/
      MASK/
```

The robustness matrix script attempts to clone the public OMBRIA repository automatically when `external/OMBRIA` is absent. If that route fails, place the dataset manually in the same location.

No raw data, trained checkpoints, or cloud-run artifacts should be committed.
