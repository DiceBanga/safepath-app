# SafePath Mobile Benchmark Plan

**Project:** SafePath - AI-Powered Hazard Detection System  
**Target Device:** Samsung Galaxy Z Fold 6  
**Document Purpose:** Document device-side benchmark requirements and execution plan for mobile deployment proof

---

## 1. Target Device Specifications

```
**Device:** Samsung Galaxy Z Fold 6 (SM-F946B, SM-F956B)
**SoC:** Snapdragon 8 Gen 3 for Galaxy (SM8650)
─|----------------|-----------------|----------------|--------------------|
| Component      | Specification    | Notes           |
|----------------|-----------------|----------------|--------------------|
| CPU            | 1x 3.39GHz X4 + 3x 3.1GHz A720 + 2x 2.9GHz A720 + 2x 2.2GHz A520 | Octa-core Kryo CPU |
| GPU            | Adreno 750        | Mobile GPU |
| NPU            | Hexagon DSP (7th Gen) | Qualcomm AI Engine Direct |
| RAM            | 12GB LPDDR5X    | Unified memory |
| Camera         | 200MP + 50MP + 12MP + 10MP | Nightography support |
| Display        | 7.6" Dynamic AMOLED  | Foldable inner display |
```

**NPU Capabilities:**
- INT8 quantization support (primary acceleration method)
- INT16 and FP16 support (alternative)
- ~45+ TOPS AI performance (AI Engine total)
- Hexagon Direct Link for low-latency camera-to-NPU pipeline

---

## 2. ONNX Export Verification
```
**Baseline Model:** exports/baseline.onnx
**Proposed Model:** exports/proposed.onnx
**Export Date:** March 24, 2026
**Opset Version:** 14 (converted to 18 during export)
**Input Resolution:** 128x256 (mini), 480x960 (target)

**Verification Steps:**
- [x] Verify ONNX file loads correctly
- [ ] Check model signature compatibility
- [ ] Validate input/output shapes
- [ ] Test inference on CPU

---

## 3. Benchmark Execution Plan

### 3.1 Development Environment Setup
```bash
# Install Android Studio
# Clone SafePath mobile repository
# Import ONNX models to assets/models/
# Set up Android emulator (Pixel 7 Pro preferred)
```

### 3.2 Benchmark Metrics
```
| Metric               | Target         | Measurement Method           |
|----------------------|---------------|------------------------------|
| Inference Latency      | < 200ms         | Frame processing time          |
| FPS                   | > 5 FPS        | Frames per second              |
| Memory Usage          | < 500MB         | Runtime memory allocation   |
| Battery Drain          | < 15%/hour     | Power consumption test         |
| Model Accuracy (mIoU) | > 70%          | Validation set evaluation    |
| Startup Time          | < 3s           | Cold start to first frame |
```

### 3.3 Benchmark Test Cases
```
1. **CPU-Only Inference**
   - Load ONNX model with ONNX Runtime (CPU)
   - Measure latency, FPS, accuracy
   - Document baseline performance

2. **GPU Inference (Adreno 750)**
   - Load ONNX model with ONNX Runtime (GPU delegate)
   - Measure latency, FPS, accuracy
   - Compare to CPU baseline

3. **NPU Inference (Hexagon DSP)**
   - Load ONNX model with Qualcomm NN SDK
   - Apply INT8 quantization
   - Measure latency, FPS, accuracy
   - Primary optimization target

4. **End-to-End Pipeline Test**
   - Camera feed → preprocessing → inference → overlay
   - Measure total pipeline latency
   - Verify real-time performance
```

### 3.4 Expected Results
```
| Hardware  | Latency | FPS  | Notes                    |
|----------|---------|------|--------------------------|
| CPU      | ~200ms  | 5    | Baseline (fallback)         |
| GPU      | ~100ms  | 10   | Medium performance         |
| NPU      | ~40-60ms | 16-25 | Target optimization       |
```

---

## 4. Quantization Strategy
```
**Primary: INT8 Quantization**
- Reduces model size by ~4x
- Enables NPU acceleration
- Minimal accuracy loss (1-3% mIoU expected)

**Quantization Tool:** PyTorch Quantization API
```python
import torch.quantization as quant

# Load FP32 model
model_fp32 = torch.load('models/proposed.onnx')

# Apply dynamic quantization
model_int8 = quant.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Export quantized model
torch.save(model_int8, 'models/proposed_int8.onnx')
```

---

## 5. Proof-of-Concept Demo Requirements
```
1. **Real-time Demo Video**
   - Record screen video of app running on device
   - Show live segmentation overlay
   - Demonstrate hazard detection in real-time

2. **Performance Screenshots**
   - Capture FPS counter overlay
   - Show latency metrics
   - Display memory usage

3. **Battery Test**
   - Run app for 30 minutes continuous inference
   - Measure battery drain percentage
   - Document thermal behavior

4. **Low-Light Test**
   - Test in controlled dark environment
   - Capture sample images with detected hazards
   - Verify Nightography effectiveness
```

---

## 6. Timeline for Device Testing
```
| Phase               | Dates        | Duration | Deliverable                    |
|---------------------|-------------|----------|--------------------------------|
| Emulator Setup        | Mar 25-26    | 1 day     | Working Android project            |
| CPU/GPU Benchmarks   | Mar 27-28    | 2 days    | Performance baseline data           |
| NPU Optimization    | Mar 29-30    | 2 days    | INT8 quantized model, FPS results |
| Real-device Testing  | Mar 31-Apr 1 | 2 days    | Actual device benchmarks             |
| Demo Recording        | Apr 1-2     | 1 day     | Demo video and screenshots           |
```

---

## 7. Fallback Plan (If device unavailable)
```
If physical device is unavailable for midterm:
1. Use Android emulator (Pixel 7 Pro API 34+)
2. Report estimated performance based on:
   - Desktop GPU inference (CUDA)
   - Published Snapdragon 8 Gen 3 benchmarks
3. Document limitations of emulator testing
4. Commit to real device testing for final submission
```

---

## 8. Deliverables for Midterm Submission
```
- [x] This document (Mobile Benchmark Plan)
- [ ] ONNX model files (baseline.onnx, proposed.onnx)
- [ ] Desktop benchmark results (FPS, latency from eval_models.py)
- [ ] Screenshot of ONNX model loaded successfully
- [ ] Plan for real device testing (timeline above)
```

---

**Author:** Dwayne Crichlow  
**Course:** AI688-001 Image and Vision Computing  
**Date:** March 24, 2026
