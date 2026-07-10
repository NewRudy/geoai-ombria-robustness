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
  2021/
    ALBANIA/
      Sentinel1/{BEFORE,AFTER,MASK}/
      Sentinel2/{BEFORE,AFTER,MASK}/
    FRANCE/
    GUYANA/
    TIMOR/
```

The confirmatory runner fetches the public OMBRIA repository at locked commit `38a490355f76da8ce27ed051138f03f3492a6e46` when `external/OMBRIA` is absent. If that route fails, place the same revision manually in the expected location.

The train folders are used for training and validation. The four 2021 folders are a separate 150-chip confirmation set and must not be used for route, checkpoint, threshold, or parameter selection.

No raw data, trained checkpoints, or cloud-run artifacts should be committed.
