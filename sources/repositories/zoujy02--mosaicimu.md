# Implementation audit: zoujy02/MosaicIMU

**Revision:** e964e10b091361126d744fd3afa74e70223f42a8
**Repository:** https://github.com/zoujy02/MosaicIMU/tree/e964e10b091361126d744fd3afa74e70223f42a8
**License/provenance:** unknown (no root LICENSE in inspected tree)
**Method:** remotely inspected source files at pinned commit. Local clone/build/tests **not** executed.

## Inspected architecture and methodology

DecoupledDeepIMUEncoder with depthwise residual 1D conv, prototype cosine router and Gumbel softmax, four temporal experts, shared decoder with BiLSTM velocity/variance heads, softplus variance and cloned-state IMU MSCKF. Adaptation adds new expert and small router adapter.

## Actual source files inspected

- [src/models/moe_net.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/moe_net.py)
- [src/models/blocks.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/blocks.py)
- [src/models/finetune_net.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/models/finetune_net.py)
- [src/utils/loss.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/utils/loss.py)
- [src/filter/msckf.py](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/src/filter/msckf.py)
- [configs/train_config.yaml](https://github.com/zoujy02/MosaicIMU/blob/e964e10b091361126d744fd3afa74e70223f42a8/configs/train_config.yaml)

## Experimental and evidence caveats

No local clone or execution; source code includes future context in both prediction heads and training has carrier prior. Reported ATE/RTE not independently reproduced.

Do not conflate code availability with reproduced research results. See [architecture review](../review/01_architectures.md) and [full synthesis](../review/08_literature_review.md).
