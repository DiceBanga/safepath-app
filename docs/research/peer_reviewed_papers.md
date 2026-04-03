# SafePath: Peer-Reviewed Paper References

**Project:** SafePath - AI-Powered Hazard Detection System  
**Student:** Dwayne Crichlow  
**Course:** AI688-001 Image and Vision Computing  
**Date:** March 22, 2026

---

## Overview

This document contains **four** peer-reviewed papers selected for their direct relevance to SafePath's technical approach. The primary three papers (2024-2026) are required for the course submission, with one additional foundational paper (2023) included for completeness.

---

## Paper 1: Safety-Critical Hazard Detection (2025)

### Full Citation

**Park, J., Lee, H., Kang, I., & Shim, H. (2025).** No Thing, Nothing: Highlighting Safety-Critical Classes for Robust LiDAR Semantic Segmentation in Adverse Weather. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*.

### Bibliographic Information

| Field | Details |
|-------|---------|
| **Title** | No Thing, Nothing: Highlighting Safety-Critical Classes for Robust LiDAR Semantic Segmentation in Adverse Weather |
| **Authors** | Junsung Park, HwiJeong Lee, Inha Kang, Hyunjung Shim |
| **Affiliation** | Korea Advanced Institute of Science and Technology (KAIST) |
| **Venue** | CVPR 2025 (IEEE/CVF Conference on Computer Vision and Pattern Recognition) |
| **Year** | 2025 |
| **DOI** | 10.48550/arXiv.2503.15910 |
| **arXiv** | https://arxiv.org/abs/2503.15910 |
| **Code** | https://github.com/engineerJPark/NTN |

### Abstract (Summary)

The paper addresses the challenge of semantic segmentation for autonomous driving in adverse weather conditions. The key insight is that "things" categories (dynamic objects like pedestrians, vehicles) are more safety-critical than "stuff" categories (static elements like road, sky). The proposed NTN (No Thing, Nothing) method prevents dangerous mispredictions by binding point features to superclasses.

### Relevance to SafePath

| Aspect | Application |
|--------|-------------|
| **Hazard Detection** | Directly addresses safety-critical object detection |
| **Robustness** | Methods for handling adverse conditions (rain, fog) |
| **Classification** | Superclass binding to prevent dangerous mispredictions |

### Key Findings Applied to SafePath

1. **NTN Method**: Binds point features to superclasses to prevent misprediction of dynamic hazards
2. **Beam-based local regions**: Improves robustness against weather/corruption
3. **Performance**: +2.6 mIoU on SemanticKITTI→SemanticSTF, +4.8-7.9 mIoU on "things" classes

---

## Paper 2: Low-Light Enhancement for Nighttime Segmentation

### Full Citation

**Sun, F., Li, C., Yang, K., Pan, Y., Yu, H., Zhang, X., & Li, Y. (2025).** FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network. *Advances in Neural Information Processing Systems (NeurIPS 2025)*.

### Bibliographic Information

| Field | Details |
|-------|---------|
| **Title** | FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network |
| **Authors** | Fangtong Sun, Congyu Li, Ke Yang, Yuchen Pan, Hanwen Yu, Xichuan Zhang, Yiying Li |
| **Affiliation** | National University of Defense Technology, Hunan University, Harbin Institute of Technology |
| **Venue** | NeurIPS 2025 (Conference on Neural Information Processing Systems) |
| **Year** | 2025 |
| **DOI** | 10.48550/arXiv.2510.23444 |
| **arXiv** | https://arxiv.org/abs/2510.23444 |
| **Code** | https://github.com/Sing-Forevet/FRBNet |

### Abstract (Summary)

The paper addresses the fundamental challenge of low-light vision degradation. It proposes a Frequency-domain Radial Basis Network (FRBNet) based on an extended Lambertian model. The key innovation is a plug-and-play module that extracts illumination-invariant features through frequency-domain channel ratio operations.

### Relevance to SafePath

| Aspect | Application |
|--------|-------------|
| **Low-Light Vision** | Directly addresses SafePath's core use case (nighttime navigation) |
| **Plug-and-Play** | Can be integrated as preprocessing module |
| **Downstream Tasks** | Proven improvement on segmentation and detection |

### Key Findings Applied to SafePath

1. **Extended Lambertian model**: Better characterizes low-light conditions for preprocessing
2. **Frequency-domain channel ratio**: Extracts illumination-invariant features
3. **Plug-and-play design**: Can be integrated into DeepLabV3+ without loss function changes
4. **Performance**: **+2.9 mIoU for nighttime segmentation**, +2.2 mAP for dark object detection

---

## Paper 3: Mobile/Edge Deployment of Segmentation Models

### Full Citation

**Cai, H., Li, J., Hu, M., Gan, C., & Han, S. (2023).** EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2023)*, 8562-8572.

### Bibliographic Information

| Field | Details |
|-------|---------|
| **Title** | EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction |
| **Authors** | Han Cai, Junyan Li, Muyan Hu, Chuang Gan, Song Han |
| **Affiliation** | MIT, Zhejiang University, Tsinghua University, MIT-IBM Watson AI Lab |
| **Venue** | ICCV 2023 (International Conference on Computer Vision) |
| **Year** | 2023 |
| **DOI** | 10.48550/arXiv.2205.14756 |
| **arXiv** | https://arxiv.org/abs/2205.14756 |
| **Code** | https://github.com/mit-han-lab/efficientvit |

### Abstract (Summary)

The paper addresses the challenge of deploying vision transformers for dense prediction tasks (semantic segmentation) on mobile and edge devices. It proposes a multi-scale linear attention mechanism that achieves global receptive fields with hardware-efficient operations, making it suitable for TensorRT deployment on mobile GPUs and NPUs.

### Relevance to SafePath

| Aspect | Application |
|--------|-------------|
| **Mobile Deployment** | Directly addresses Samsung Galaxy Z Fold 6 deployment |
| **Efficiency** | Hardware-optimized for edge GPU/NPU |
| **Segmentation** | Proven on Cityscapes benchmark |

### Key Findings Applied to SafePath

1. **Lightweight multi-scale attention**: Achieves global receptive field with hardware-efficient operations
2. **TensorRT-friendly**: Optimized for edge GPU/NPU deployment
3. **Performance**:
   - **8.8x speedup** over SegFormer on mobile
   - **3.8x speedup** over SegNeXt on Jetson AGX Orin
   - **74.5% mIoU** on Cityscapes with only 0.7M parameters
4. **Architecture insights**: Linear attention reduces complexity from O(n²) to O(n)

---

## Paper 4: Mobile/Edge Deployment with State Space Models (2025)

### Full Citation

**He, H., Zhang, J., Cai, Y., Chen, H., Hu, X., Gan, Z., Wang, Y., Wang, C., Wu, Y., & Xie, L. (2025).** MobileMamba: Lightweight Multi-Receptive Visual Mamba Network. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025)*.

### Bibliographic Information

| Field | Details |
|-------|---------|
| **Title** | MobileMamba: Lightweight Multi-Receptive Visual Mamba Network |
| **Authors** | Haoyang He, Jiangning Zhang, Yuxuan Cai, Hongxu Chen, Xiaobin Hu, Zhenye Gan, Yabiao Wang, Chengjie Wang, Yunsheng Wu, Lei Xie |
| **Affiliation** | Zhejiang University, Youtu Lab Tencent, Huazhong University of Science and Technology |
| **Venue** | CVPR 2025 (IEEE/CVF Conference on Computer Vision and Pattern Recognition) |
| **Year** | 2025 |
| **DOI** | 10.48550/arXiv.2411.15941 |
| **arXiv** | https://arxiv.org/abs/2411.15941 |
| **Code** | https://github.com/lewandofskee/MobileMamba |

### Abstract (Summary)

This paper introduces MobileMamba, a lightweight backbone using State Space Models (SSMs) specifically designed for mobile and edge devices. The Mamba architecture provides linear computational complexity O(n), unlike Transformers' O(n²), making it ideal for resource-constrained environments. The paper includes explicit benchmarks with DeepLabV3 on the ADE20K dataset.

### Relevance to SafePath

| Aspect | Application |
|--------|-------------|
| **DeepLabV3 Integration** | Directly benchmarks with DeepLabV3 — SafePath's exact architecture |
| **Mobile Optimization** | Designed for resource-constrained devices like Galaxy Z Fold 6 |
| **NPU-Friendly** | Linear complexity maps well to NPU acceleration |

### Key Findings Applied to SafePath

1. **State Space Models**: Mamba-based backbone with linear complexity for efficient inference
2. **Multi-Receptive Field**: MRFFI module combines global context with local features
3. **Performance with DeepLabV3**:
   - **36.6% mIoU** on ADE20K with only 4.7G FLOPs
   - Outperforms MobileOne (36.2% mIoU, 14.7G FLOPs)
   - Outperforms MobileViT-v2 (31.9% mIoU, 26.1G FLOPs)
4. **Efficiency**: 3x fewer FLOPs than comparable models

---

## Summary Table

| # | Paper | Venue | Year | Focus Area |
|---|-------|-------|------|------------|
| 1 | No Thing, Nothing | CVPR | 2025 | Safety-critical hazard detection |
| 2 | FRBNet | NeurIPS | 2025 | Low-light nighttime segmentation |
| 3 | MobileMamba | CVPR | 2025 | Mobile/edge deployment (SSM) |
| 4 | EfficientViT | ICCV | 2023 | Mobile/edge deployment (Transformer) |

**Note:** Papers 1-3 are from 2024-2026 as required by the course. Paper 4 (EfficientViT, 2023) is included as foundational reference for mobile deployment techniques.

---

## How These Papers Inform SafePath

### Architecture Decisions (ADR-0001)

1. **From Paper 1**: Consider superclass binding for hazard categories to prevent dangerous misclassifications
2. **From Paper 2**: Integrate FRBNet as low-light preprocessing module before DeepLabV3+
3. **From Paper 3**: Apply SSM/Mamba principles for NPU-optimized inference
4. **From Paper 4**: Apply multi-scale linear attention principles for mobile optimization

### Implementation Plan

| Paper | SafePath Application |
|-------|---------------------|
| NTN (Paper 1) | Hazard superclass classification (obstacle, ground, safe) |
| FRBNet (Paper 2) | Preprocessing pipeline for low-light enhancement |
| MobileMamba (Paper 3) | SSM-based efficiency techniques for NPU |
| EfficientViT (Paper 4) | Model optimization techniques for mobile NPU |

---

## References (BibTeX)

```bibtex
@inproceedings{park2025ntn,
  title={No Thing, Nothing: Highlighting Safety-Critical Classes for Robust LiDAR Semantic Segmentation in Adverse Weather},
  author={Park, Junsung and Lee, HwiJeong and Kang, Inha and Shim, Hyunjung},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{sun2025frbnet,
  title={FRBNet: Revisiting Low-Light Vision through Frequency-Domain Radial Basis Network},
  author={Sun, Fangtong and Li, Congyu and Yang, Ke and Pan, Yuchen and Yu, Hanwen and Zhang, Xichuan and Li, Yiying},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}

@inproceedings{he2025mobilemamba,
  title={MobileMamba: Lightweight Multi-Receptive Visual Mamba Network},
  author={He, Haoyang and Zhang, Jiangning and Cai, Yuxuan and Chen, Hongxu and Hu, Xiaobin and Gan, Zhenye and Wang, Yabiao and Wang, Chengjie and Wu, Yunsheng and Xie, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}

@inproceedings{cai2023efficientvit,
  title={EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={8562--8572},
  year={2023}
}
```

---

**Document Version:** 1.1  
**Created:** March 22, 2026  
**Updated:** March 22, 2026  
**Author:** Dwayne Crichlow
