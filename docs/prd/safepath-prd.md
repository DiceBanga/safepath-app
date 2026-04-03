# SafePath: AI-Powered Hazard Detection System

## Product Requirements Document (PRD)

**Author:** Dwayne Crichlow  
**Status:** Draft  
**Last Updated:** March 22, 2026  
**Course:** AI688 - Image and Vision Computing (Section 1)

---

## 1. OVERVIEW

### 1.1 Problem Statement

Navigating unfamiliar environments in low-light conditions poses significant safety risks to pedestrians. Hazards like potholes, uneven terrain, obstacles, and trip hazards are often obscured by darkness, leading to accidents. While smartphone flashlights are useful, they require manual operation and can be distracting. There is a critical need for a hands-free, automated solution that utilizes modern mobile hardware to continuously monitor the user's path and provide timely warnings.

### 1.2 Proposed Solution

**SafePath** is an AI-powered computer vision application that uses real-time semantic segmentation to detect navigational hazards in low-light environments. The system provides visual and auditory alerts to users, helping them navigate safely through dark or complex terrain.

**Key Innovation:** Combines Samsung Galaxy Z Fold 6's advanced low-light camera sensors with lightweight DeepLabV3+ semantic segmentation optimized for mobile NPU inference.

### 1.3 Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Detection Accuracy (mIoU) | > 85% | Validation on test set |
| Hazard Classification (F1) | > 80% | Per-class F1 scores |
| Inference Speed | > 5 FPS | Frame rate on device |
| Report Generation | < 5 sec | Processing time |
| Model Size | < 5M params | Parameter count |

---

## 2. CONTEXT

### 2.1 Background

Recent advancements in mobile hardware have introduced cameras with superior low-light sensitivity. Samsung devices excel in capturing clear images in near-darkness. Concurrently, deep learning architectures like DeepLabV3+ have achieved state-of-the-art results in semantic segmentation, offering robust scene understanding.

### 2.2 User Research

**Target Users:**
- Pedestrians navigating in low-light conditions
- Visually impaired individuals requiring navigation assistance
- Night workers (security, delivery) in unfamiliar environments
- Outdoor enthusiasts (hikers, runners) during dawn/dusk

**User Pain Points:**
- Cannot see hazards in darkness despite flashlight use
- Flashlight requires manual operation (not hands-free)
- Existing solutions don't provide real-time hazard alerts
- No automated logging/reporting of detected hazards

### 2.3 Competitive Analysis

| Solution | Limitations |
|----------|-------------|
| Smartphone Flashlight | Manual operation, distracts user |
| Night Vision Goggles | Expensive, bulky, not mainstream |
| White Canes | Limited range, requires training |
| GPS Navigation | Doesn't detect physical hazards |
| SafePath (Proposed) | Real-time, hands-free, automated alerts |

---

## 3. REQUIREMENTS

### 3.1 User Stories

**US-1: Real-Time Hazard Detection**
```
As a pedestrian walking at night
I want to receive real-time alerts about hazards in my path
So that I can avoid accidents and injuries

Acceptance Criteria:
- Given I'm walking with SafePath active
- When a hazard (pothole, pole, obstacle) enters the camera view
- Then I receive visual and/or auditory alert within 200ms
- And the hazard is highlighted on screen with appropriate color coding
```

**US-2: Low-Light Operation**
```
As a user in dark environments
I want the system to work effectively in near-darkness
So that I can navigate safely without additional lighting

Acceptance Criteria:
- Given ambient light is below 10 lux
- When camera captures the scene
- Then hazard detection maintains >70% of daytime accuracy
- And preprocessing enhances visibility automatically
```

**US-3: Incident Reporting**
```
As a user who encountered hazards
I want to generate a PDF report of detected hazards
So that I can document incidents or share with authorities

Acceptance Criteria:
- Given hazards have been detected during a session
- When I request a report
- Then a PDF is generated within 5 seconds
- And report includes timestamps, hazard types, and snapshots
```

**US-4: Mobile Performance**
```
As a mobile user
I want the app to run smoothly on my Samsung device
So that I can use it without draining battery excessively

Acceptance Criteria:
- Given app is running on Galaxy Z Fold 6
- When processing live camera feed
- Then inference achieves >5 FPS
- And battery consumption <15% per hour
```

### 3.2 Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | System shall capture real-time video from device camera | P0 |
| FR-2 | System shall preprocess frames for low-light enhancement | P0 |
| FR-3 | System shall perform semantic segmentation using DeepLabV3+ | P0 |
| FR-4 | System shall detect hazards: potholes, holes, poles, steps, uneven terrain | P0 |
| FR-5 | System shall overlay color-coded masks on live video feed | P0 |
| FR-6 | System shall generate auditory alerts for detected hazards | P1 |
| FR-7 | System shall generate PDF reports with detection logs | P1 |
| FR-8 | System shall support dual-view dashboard on foldable display | P2 |
| FR-9 | System shall log all detections with timestamps and snapshots | P1 |
| FR-10 | System shall provide configurable alert sensitivity | P2 |

### 3.3 Non-Functional Requirements

| Category | Requirement | Target |
|----------|-------------|--------|
| **Performance** | Frame processing rate | >5 FPS |
| **Performance** | Latency (detection to alert) | <200ms |
| **Performance** | Model inference time | <150ms/frame |
| **Accuracy** | Mean IoU on validation set | >85% |
| **Accuracy** | Hazard F1-score | >80% |
| **Reliability** | Uptime during session | >99% |
| **Usability** | Time to first detection | <5 sec from launch |
| **Portability** | Target device | Samsung Galaxy Z Fold 6 |
| **Maintainability** | Code documentation | >80% coverage |

---

## 4. DESIGN

### 4.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SafePath System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Camera    │→ │Preprocessing│→ │   Inference Engine  │  │
│  │   Input     │  │   Engine    │  │   (DeepLabV3+)      │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│         │                │                    │              │
│         ▼                ▼                    ▼              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │  Live Feed  │  │   Low-Light │  │  Segmentation Mask  │  │
│  │  Display    │  │  Enhance    │  │  (Hazard Overlay)   │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                                   │          │
│         ┌─────────────────────────────────────────┘          │
│         ▼                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   Hazard    │→ │   Alert     │→ │    Report           │  │
│  │  Classifier │  │   System    │  │    Generator        │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow

```
Camera Frame → Low-Light Enhancement → DeepLabV3+ Inference → 
Post-Processing → Hazard Classification → Overlay Rendering → 
Display + Alert + Logging
```

### 4.3 Hazard Color Coding

| Hazard Type | Color | RGB Value |
|-------------|-------|-----------|
| Pothole/Hole | Red | (255, 0, 0) |
| Pole/Obstacle | Orange | (255, 165, 0) |
| Uneven Terrain | Yellow | (255, 255, 0) |
| Water/Puddle | Blue | (0, 100, 255) |
| Safe Path | Green | (0, 255, 0) |

### 4.4 Technical Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Language | Python 3.9+ | Library ecosystem compatibility |
| Deep Learning | PyTorch 2.0+ | DeepLabV3+ pre-trained models |
| Image Processing | OpenCV 4.x | Real-time video handling |
| PDF Generation | ReportLab | Structured report creation |
| Model Format | ONNX/TFLite | Mobile deployment optimization |
| Camera API | CameraX (Android) | Low-latency camera access |

---

## 5. SCOPE

### 5.1 In Scope (MVP - Phase 1-3)

- [x] Real-time camera feed capture
- [x] Low-light image preprocessing
- [x] DeepLabV3+ semantic segmentation inference
- [x] Hazard detection (potholes, poles, uneven terrain)
- [x] Visual overlay with color-coded masks
- [x] Basic auditory alerts
- [x] PDF report generation
- [x] Session logging

### 5.2 Out of Scope (Future Phases)

- [ ] Depth estimation for hazard distance
- [ ] Voice command interface
- [ ] GPS integration for hazard mapping
- [ ] Cloud-based model updates
- [ ] Multi-user hazard sharing
- [ ] AR glasses integration
- [ ] iOS deployment

### 5.3 Future Considerations

- Integration with accessibility services
- Community hazard reporting
- Machine learning model updates via OTA
- Smartwatch companion app

---

## 6. TIMELINE

### Phase 1: Planning & Research (Feb 26-27, 2026)

| Task | Owner | Due Date | Status |
|------|-------|----------|--------|
| Project Proposal | Dwayne | Feb 26 | ✅ Complete |
| Environment Setup | Dwayne | Feb 27 | 🔄 In Progress |
| Dataset Research | Dwayne | Feb 27 | ✅ Complete |
| Literature Review | Dwayne | Feb 27 | ✅ Complete |

### Phase 2: Model Development (Feb 28 - Mar 15, 2026)

| Task | Owner | Due Date | Status |
|------|-------|----------|--------|
| Dataset Download & Prep | Dwayne | Mar 1 | ⬜ Pending |
| Data Loader Implementation | Dwayne | Mar 5 | ⬜ Pending |
| Augmentation Pipeline | Dwayne | Mar 8 | ⬜ Pending |
| Transfer Learning Training | Dwayne | Mar 12 | ⬜ Pending |
| Model Validation | Dwayne | Mar 15 | ⬜ Pending |

### Phase 3: Proof of Concept (Mar 16 - Apr 2, 2026)

| Task | Owner | Due Date | Status |
|------|-------|----------|--------|
| Camera Integration | Dwayne | Mar 20 | ⬜ Pending |
| Preprocessing Module | Dwayne | Mar 22 | ⬜ Pending |
| Inference Pipeline | Dwayne | Mar 26 | ⬜ Pending |
| Visualization Module | Dwayne | Mar 30 | ⬜ Pending |
| Report Generator | Dwayne | Apr 2 | ⬜ Pending |

### Phase 4: Finalization (Apr 3 - May 7, 2026)

| Task | Owner | Due Date | Status |
|------|-------|----------|--------|
| Performance Optimization | Dwayne | Apr 15 | ⬜ Pending |
| Documentation | Dwayne | Apr 25 | ⬜ Pending |
| Demo Video | Dwayne | May 1 | ⬜ Pending |
| Final Presentation | Dwayne | May 7 | ⬜ Pending |

---

## 7. RISK ASSESSMENT

| Risk | Likelihood | Impact | Mitigation Strategy |
|------|------------|--------|---------------------|
| Model Overfitting | Medium | High | Dropout regularization, early stopping, data augmentation |
| Low Inference Speed | Medium | Medium | Model quantization (INT8), NPU utilization, MobileNet backbone |
| Insufficient Low-Light Data | High | Medium | Aggressive augmentation, synthetic low-light generation, Zero-DCE enhancement |
| Hardware Incompatibility | Low | Medium | Fallback to Google Colab, cross-platform testing |
| Battery Drain | Medium | Low | Efficient inference, adaptive frame rate |

---

## 8. REFERENCE PROJECTS

Based on research, the following open-source projects serve as implementation references:

| Project | URL | Relevance |
|---------|-----|-----------|
| nmhaddad/semantic-segmentation | GitHub | Off-road DeepLabV3+ implementation |
| meiqisheng/DeepLabv3plus | GitHub | MobileNet-optimized with pretrained weights |
| FloorSegmentationApp | GitHub | Android deployment template |
| SafeVision-AI | GitHub | Hazard detection domain pattern |
| DexiNed | GitHub | Edge detection with quantized mobile model |

---

## 9. APPENDIX

### A. Dataset Sources

| Dataset | Size | Use Case | License |
|---------|------|----------|---------|
| BDD100K | 1.8TB | Primary training, nighttime subset | BDD License |
| Cityscapes | 11GB | Urban scene benchmark | CC BY-NC-SA 4.0 |
| NightCity | 4GB | Nighttime segmentation | Research |
| Cracks and Potholes | 197MB | Hazard-specific annotations | CC BY 4.0 |
| ExDARK | 3GB | Low-light object detection | BSD-3-Clause |

### B. Academic References

Key papers informing the technical approach:

1. **MFA-DeepLabV3+** (Liu et al., 2025) - MobileNetV2 backbone optimization
2. **CISS** (Sakaridis et al., 2023) - Condition-invariant segmentation
3. **Retinexformer** (Cai et al., ICCV 2023) - Low-light enhancement
4. **EdgeTAM** (Zhou et al., 2025) - Mobile NPU deployment patterns

### C. Open Questions

1. Should we prioritize accuracy over speed, or balance both?
2. What is the minimum acceptable FPS for real-time use?
3. Should auditory alerts be customizable by hazard type?
4. What level of low-light performance is acceptable?

---

**Document Version:** 1.0  
**Created:** March 22, 2026  
**Last Review:** March 22, 2026
