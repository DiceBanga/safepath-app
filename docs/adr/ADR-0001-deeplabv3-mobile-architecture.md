# ADR-0001: Adopt DeepLabV3+ with MobileNet Backbone for Mobile Hazard Detection

## Status

**Proposed** - Awaiting implementation validation

## Context

SafePath requires a semantic segmentation model capable of:
1. Detecting navigational hazards (potholes, poles, uneven terrain) in real-time
2. Operating effectively in low-light conditions
3. Running efficiently on mobile hardware (Samsung Galaxy Z Fold 6)
4. Achieving >85% mIoU with >5 FPS inference speed

### Forces at Play

| Force | Description |
|-------|-------------|
| **Accuracy** | Safety application requires reliable hazard detection |
| **Speed** | Real-time operation needs >5 FPS for responsiveness |
| **Model Size** | Mobile deployment requires compact models (<5M params) |
| **Low-Light Performance** | Primary use case is dark environments |
| **Battery Efficiency** | Extended use requires power-efficient inference |
| **Development Time** | Academic project with fixed timeline (10 weeks) |

### Hardware Constraints

**Samsung Galaxy Z Fold 6 Specifications:**
- Processor: Snapdragon 8 Gen 3 (SM8650)
- CPU: 1x 3.39GHz Cortex-X4 + 3x 3.1GHz Cortex-A720 + 2x 2.9GHz Cortex-A720 + 2x 2.2GHz Cortex-A520
- GPU: Adreno 750
- **NPU: Hexagon DSP (7th Gen) with Qualcomm AI Engine**
- RAM: 12GB LPDDR5X
- Camera: 200MP main with Nightography
- Display: 7.6" foldable AMOLED

**NPU (Hexagon DSP) Capabilities:**
- INT8 quantization support (primary acceleration)
- INT16 and FP16 support
- ~45+ TOPS AI performance (AI Engine total)
- ONNX Runtime / TensorFlow Lite / Qualcomm NN SDK support
- Hexagon Direct Link for low-latency camera-to-NPU pipeline

**Hardware Acceleration Strategy:**

| Component | Hardware | Rationale |
|-----------|----------|-----------|
| **Model Inference** | **NPU (Primary)** | INT8 quantized model, 2-4x speedup over CPU |
| Preprocessing | CPU (Kryo) | OpenCV operations, minimal compute |
| Post-processing | CPU/GPU | Overlay rendering, minimal overhead |
| Fallback | CPU | If NPU unavailable or for debugging |

**Expected Performance by Hardware:**

| Hardware | Inference Time | FPS | Power Usage |
|----------|---------------|-----|-------------|
| NPU (INT8) | ~40-60ms | 16-25 | Low (~2W) |
| GPU (FP16) | ~80-100ms | 10-12 | Medium (~4W) |
| CPU (FP32) | ~200-300ms | 3-5 | High (~6W) |

## Decision

We will adopt **DeepLabV3+ with MobileNetV3-Large backbone** as the primary segmentation architecture for SafePath, with the following configuration:

### Architecture Specification

```
┌─────────────────────────────────────────────────────────────┐
│                    DeepLabV3+ Architecture                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐                                        │
│  │  Input Frame    │  480x960x3 (mobile resolution)        │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ MobileNetV3-Lrg │  Backbone (efficient feature extract) │
│  │   (Encoder)     │  Pretrained on ImageNet               │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │      ASPP       │  Atrous Spatial Pyramid Pooling       │
│  │  (rates: 6,12,18)│  Multi-scale context aggregation     │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │    Decoder      │  Low-level feature fusion             │
│  │  (DeepLabV3+)   │  Upsampling to full resolution        │
│  └────────┬────────┘                                        │
│           ▼                                                 │
│  ┌─────────────────┐                                        │
│  │ Segmentation Map│  480x960xC (C = num classes)          │
│  └─────────────────┘                                        │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **MobileNetV3-Large backbone** | 5.4M params, 219M MAdd - optimal for mobile NPU |
| **ASPP rates: 6, 12, 18** | Captures multi-scale hazards (small potholes to large obstacles) |
| **Input resolution: 480x960** | Balance between detail and inference speed |
| **Output stride: 16** | Reduces computation while preserving spatial detail |
| **INT8 quantization** | Leverages NPU for 2-4x speedup |

### Model Specifications

| Metric | Value | Notes |
|--------|-------|-------|
| Parameters | ~5.4M | MobileNetV3-Large backbone |
| FLOPs | ~2.2B | At 480x960 resolution |
| Model Size | ~22MB (FP32) | ~6MB (INT8 quantized) |
| Inference Time | ~80ms | Target on Snapdragon 8 Gen 3 |
| FPS | ~12 | Theoretical (targeting >5) |

## Consequences

### Positive

1. **Proven Architecture**: DeepLabV3+ is well-documented with extensive pre-trained weights available on Cityscapes and Pascal VOC

2. **Mobile Optimization**: MobileNetV3-Large is specifically designed for mobile efficiency with Neural Architecture Search (NAS) optimization

3. **Multi-Scale Detection**: ASPP module captures hazards at varying scales - critical for detecting both small potholes and large obstacles

4. **Transfer Learning Ready**: Pre-trained weights on Cityscapes (urban street scenes) provide strong initialization for hazard detection

5. **NPU Compatible**: Architecture supports INT8 quantization for Snapdragon NPU acceleration

6. **Active Research**: Recent papers (MFA-DeepLabV3+, DSC-DeepLabV3+) demonstrate continued improvements

### Negative

1. **Accuracy Trade-off**: MobileNet backbone achieves ~75% mIoU vs ~80%+ with ResNet/Xception backbones - acceptable for MVP but may miss edge cases

2. **Low-Light Challenge**: Standard DeepLabV3+ not optimized for low-light; requires preprocessing module (Zero-DCE or Retinexformer)

3. **Limited Hazard Classes**: Cityscapes pre-training doesn't include potholes/cracks - requires fine-tuning on hazard-specific datasets

4. **Memory Constraints**: 480x960 resolution may miss small hazards at distance; higher resolution increases latency

5. **Platform Lock-in**: Optimization for Snapdragon may require rework for other platforms

### Neutral

1. **Framework Choice**: PyTorch for development, ONNX/TFLite for deployment - standard industry practice

2. **Resolution Trade-off**: 480x960 balances speed/accuracy; can be adjusted based on testing

3. **Quantization Impact**: INT8 may reduce accuracy by 1-3% mIoU; requires validation

## Alternatives Considered

### Alternative 1: DeepLabV3+ with Xception-65 Backbone

**Description:**
- Original DeepLabV3+ architecture with Xception-65 encoder
- Achieves ~79% mIoU on Cityscapes (state-of-the-art at publication)

**Pros:**
- Higher accuracy than MobileNet variants
- More robust feature extraction
- Better handling of complex scenes

**Cons:**
- ~40M parameters (8x larger than MobileNetV3)
- ~60 GFLOPs (27x more computation)
- ~300ms inference on mobile (unacceptable for real-time)
- Battery drain concerns

**Why Not Chosen:**
Computationally expensive for mobile deployment. Would not achieve target >5 FPS without significant optimization, defeating the purpose of the original architecture choice.

### Alternative 2: U-Net with EfficientNet-B0 Backbone

**Description:**
- U-Net architecture with EfficientNet-B0 encoder
- Popular for medical/industrial segmentation

**Pros:**
- Simpler architecture, easier to debug
- Good for small datasets
- Skip connections preserve spatial detail

**Cons:**
- Typically requires more training epochs
- No multi-scale context (lacks ASPP)
- Less proven on street scene segmentation
- Lower accuracy on Cityscapes (~72% mIoU)

**Why Not Chosen:**
ASPP module in DeepLabV3+ is critical for detecting hazards at multiple scales. U-Net's simpler architecture would miss small potholes while capturing large obstacles, or vice versa.

### Alternative 3: SegFormer (Transformer-based)

**Description:**
- Transformer-based semantic segmentation
- MiT (Mix Transformer) backbone
- Achieves ~84% mIoU on Cityscapes

**Pros:**
- State-of-the-art accuracy
- Better global context understanding
- No CNN inductive bias limitations

**Cons:**
- Higher memory requirements
- Longer inference time
- Less mobile-optimized
- Newer architecture with fewer deployment examples

**Why Not Chosen:**
While more accurate, Transformers are less mature for mobile deployment. EdgeTAM (2025) demonstrates viability, but adds complexity to an already tight timeline. SafePath prioritizes proven mobile patterns.

### Alternative 4: BiSeNetV2 (Real-Time Segmentation)

**Description:**
- Bilateral segmentation network designed for real-time
- Separate detail and context branches

**Pros:**
- Specifically designed for >30 FPS
- 3.4M parameters (very lightweight)
- Proven on Cityscapes (~73% mIoU)

**Cons:**
- Lower accuracy than DeepLabV3+
- Less flexible for transfer learning
- Limited pre-trained weight availability

**Why Not Chosen:**
DeepLabV3+ with MobileNetV3 achieves comparable speed with higher accuracy and better transfer learning support. BiSeNetV2 is better suited for embedded devices with stricter constraints.

### Alternative 5: MobileNetV3-Small Backbone (Baseline Comparison)

**Description:**
- DeepLabV3+ with MobileNetV3-Small backbone as baseline
- Smaller, faster variant for comparison

**Baseline Comparison Study:**

| Model | Parameters | mIoU (Cityscapes) | FPS (Mobile) | Size (INT8) |
|-------|------------|-------------------|--------------|-------------|
| **MobileNetV3-Small + DeepLabV3+** | 2.9M | 70.5% | 18-22 FPS | ~4 MB |
| **MobileNetV3-Large + DeepLabV3+** (Proposed) | 5.4M | 75.1% | 12-16 FPS | ~6 MB |
| **MobileNetV2 + DeepLabV3+** | 4.5M | 73.8% | 14-18 FPS | ~5 MB |
| **ResNet-50 + DeepLabV3+** | 39.6M | 77.2% | 3-5 FPS | ~40 MB |
| **Xception-71 + DeepLabV3+** | 43.5M | 79.0% | 2-4 FPS | ~175 MB |

**Trade-off Analysis:**

| Metric | Small (Baseline) | Large (Proposed) | Δ |
|--------|------------------|------------------|---|
| Accuracy (mIoU) | 70.5% | 75.1% | **+4.6%** |
| Speed (FPS) | 20 FPS | 14 FPS | -6 FPS |
| Model Size | 4 MB | 6 MB | +2 MB |
| Parameters | 2.9M | 5.4M | +2.5M |

**Why MobileNetV3-Large over Small:**
1. **Accuracy Priority**: Safety application benefits from 4.6% higher mIoU
2. **Hazard Detection**: Small variant may miss small hazards (potholes, cracks)
3. **Acceptable Speed**: 14 FPS still exceeds 5 FPS minimum requirement
4. **Memory Headroom**: 2MB size increase acceptable on 12GB device

**Baseline Evaluation Plan:**
Both MobileNetV3-Small and MobileNetV3-Large will be trained and evaluated to validate the efficiency gains. The baseline comparison will be included in the Midpoint Update (Apr 2, 2026).

## Implementation Plan

### Phase 1: Foundation (Week 1-2)

1. Clone reference implementation (VainF/DeepLabV3Plus-Pytorch)
2. Load MobileNetV3-Large backbone with ImageNet weights
3. Validate inference pipeline on sample images

### Phase 2: Transfer Learning (Week 3-4)

1. Download Cityscapes and BDD100K datasets
2. Train on Cityscapes (fine annotations) for urban scene baseline
3. Validate on BDD100K validation set

### Phase 3: Hazard Fine-tuning (Week 5-6)

1. Add hazard-specific datasets (Cracks and Potholes, Pothole-600)
2. Create unified label mapping (road, sidewalk, pothole, obstacle, etc.)
3. Fine-tune with aggressive low-light augmentation

### Phase 4: Mobile Optimization (Week 7-8)

1. Export to ONNX format
2. Apply INT8 quantization
3. Validate inference speed on target hardware
4. Integrate with mobile camera pipeline

## Validation Criteria

This decision will be considered successful if:

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| mIoU on validation set | > 80% | Cityscapes-style evaluation |
| Hazard F1-score | > 75% | Per-class F1 on hazard classes |
| Inference FPS | > 5 FPS | Frame rate on Galaxy Z Fold 6 |
| Model size (quantized) | < 10 MB | File size |
| Battery drain | < 15%/hour | Power consumption test |

## References

1. Chen, L.C., et al. (2018). "Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation" - Original DeepLabV3+ paper
2. Howard, A., et al. (2019). "Searching for MobileNetV3" - MobileNetV3 architecture
3. Liu, H., et al. (2025). "MFA-DeepLabv3+: A Lightweight Semantic Segmentation Network" - Mobile optimization research
4. Yu, J., et al. (2025). "Improved DeepLabV3+ with MobileNetV3" - Recent mobile optimization techniques

---

**Author:** Dwayne Crichlow  
**Date:** March 22, 2026  
**Reviewers:** AI688 Course  
**Supersedes:** None
