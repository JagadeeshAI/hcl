README

🧠 Tongue Diagnosis System

This project automates tongue feature analysis using image processing and deep learning. It extracts clinically significant features from tongue images to compute Nutritional and Mantle Health Scores, which may assist in traditional and modern health diagnostics.

---

📸 Dataset

We use the [BioHit Tongue Image Dataset](https://www.kaggle.com/datasets) from Kaggle, which contains high-quality images labeled for various tongue conditions.

---

🧪 Features Extracted

The system analyzes 5 key features from tongue images:

1. Coating  
   - Measures the amount of white or yellowish coating.  
   - Detected using classical image processing: white pixel ratio from the segmented tongue mask.

2. Jagged Shape  
   - Measures the degree of serrated or uneven tongue edges.  
   - Calculated using contour curvature ratios on CLAHE-enhanced edges.

3. Cracks  
   - Quantifies number and depth of fissures on the tongue surface.  
   - Detected using Canny edge detection and a sliding window edge density approach.

4. Papillae Size  
   - Detects the fine hair-like projections (fungiform papillae).  
   - Extracted using a fine-tuned EfficientNet classifier.

5. Redness  
   - Measures color intensity at red, round papillae.  
   - Also extracted using the EfficientNet model, trained on redness patterns.

---

🧬 Final Scores Computed

- Refined Nutrition Score Logic (Weighted) 
  Nutrition = 0.5 × Redness + 0.3 × PapillaeSize + 0.2 × (10 − Coating)
  Indicates deficiencies or imbalances in diet based on features like coating, cracks, and redness.
  Justification:

  Redness (50%)*: Indicates blood circulation and energy flow; most important in determining nutrition absorption.
  Papillae Size (30%)*: Reflects vitality of tongue tissue and metabolic health.
  Coating (Inverted, 20%)*: Too much coating is bad; minimal coating is good for nutrient uptake.



- Refined Mantle Score Logic (Weighted)
  Mantle = 0.6 × Cracks + 0.4 × JaggedShape  
  Reflects protective barriers and surface health, linked to jagged edges and papillae structure.
  Justification:

  Cracks (60%)*: Deep or numerous cracks reflect chronic dryness or systemic weakness.
  Jagged Shape (40%)*: Often associated with Qi deficiency or internal stress.



---

🧠 Methods Overview

🔍 1. Tongue Detection & Cropping
- Uses a YOLOv8 model trained to detect tongues.
- Crops the tongue region from raw images.

🧼 2. Segmentation
- Applies Mask2Former (Swin Transformer backbone) for semantic segmentation of the tongue area.
- Removes background and focuses on tongue surface only.

### ⚙ 3. Normalization
- Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) for better visibility of surface texture and edges.

### 🧾 4. Feature Extraction
| Feature        | Method                                                                 |
|----------------|------------------------------------------------------------------------|
| Coating        | Classical CV (white pixel ratio in segmented region)                   |
| Jagged Shape   | Contour curvature on CLAHE-enhanced edges                              |
| Cracks         | Canny edge detector + sliding window pixel density                     |
| Papillae Size  | EfficientNet-based classifier for texture/pattern recognition          |
| Redness        | EfficientNet model trained to detect red round papillae regions        |

---


<!-- export PYTHONPATH="$PYTHONPATH:/media/jag/volD/hcl" -->
