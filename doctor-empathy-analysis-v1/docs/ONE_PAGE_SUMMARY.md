# Doctor Empathy Language Detection – One-Page Summary
## Empathy Detection in Chinese Medical Consultations

**Author**: Evelyn Du | **Date**: August 2025 | **Project**: Empathy language detection in Chinese medical consultations

---

## **Methodology**

**Dual analysis approach**: Combine traditional linguistic analysis (lexicon-based empathy features) with machine learning enhancement (linguistic feature modeling) to deliver comprehensive empathy detection.

**Technical flow**: Medical dialogue → Text preprocessing → Feature extraction → Dual analysis → Result output

---

## **Feature Strategy**

### **Empathy Lexicon (8 Dimensions)**
| Category | Weight | Core Vocabulary Examples |
|----------|--------|--------------------------|
| **Understanding & Empathy** | 1.5 | 能够理解, 理解您, 确实 |
| **Professional Care** | 1.3 | 建议, 推荐, 最好, 定期 |
| **Comfort & Encouragement** | 1.4 | 别担心, 会好的, 慢慢来 |
| **Care & Consideration** | 1.3 | 密切观察, 定期复查, 注意 |
| **Patient Explanation** | 1.3 | 详细, 具体, 详细说明 |
| **Support & Help** | 1.2 | 帮助您, 协助您, 一起 |
| **Emotional Response** | 1.1 | 嗯, 好的, 可以, 我明白 |
| **Thanking for Trust** | 1.2 | 感谢您的信任, 谢谢您 |

### **Linguistic Features (27 metrics)**
- **Text statistics**: Length, word count, character count, average word length, sentence count
- **Syntactic features**: Interrogatives, suggestion sentences, punctuation counts
- **Emotional cues**: Modal particles, intensifiers, emotion patterns
- **Stylistic signals**: Personal pronouns, repeated words, lexical diversity

---

## **Machine Learning Models**

**Model suite**: RandomForest (200 trees), LogisticRegression (2,000 iterations), GradientBoosting (150 estimators)

**Multi-label classification**: Supports six empathy dimensions – emotional_acknowledgment, reassurance_comfort, encouragement, shared_responsibility, positive_reframing, apology

---

## **Preliminary Results**

### **Rule-Based Analysis (11 Ophthalmology Cases)**
- **Average empathy score**: 1.532
- **Average empathy density**: 2.454 per 100 characters
- **Average empathy intensity**: 0.043
- **Highest score**: Pediatric cataract case (4.267)
- **Lowest score**: Cataract case (0.367)

### **Model Performance**
- **RandomForest**: F1 micro 0.745, F1 macro 0.702
- **LogisticRegression**: F1 micro 0.747, F1 macro 0.712
- **GradientBoosting**: F1 micro 0.715, F1 macro 0.667

### **High-Frequency Empathy Vocabulary**
1. **建议** (10) – Professional care | 2. **可以** (7) – Emotional response  
3. **定期** (5) – Professional care | 4. **感谢** (4) – Thanking for trust  
5. **详细** (4) – Patient explanation

### **Empathy Category Distribution**
- **Professional care** (19) – Highest share | **Emotional response** (5) – Foundational expressions  
- **Patient explanation** (4) – Communication strength | **Understanding & empathy** (3) – Core validation

---

## **Technical Highlights**

**Chinese medical text specialization**: Native Chinese tokenization, medical terminology recognition, visualization tuned for Chinese fonts.

**Intelligent data augmentation**: Dynamic synthetic samples, edge-case coverage, label-noise handling.

**Ensemble prediction**: Weighted voting, probability outputs, explainability support.

---

## **Key File Locations**

**Executable script**: `src/empathy_analysis.py` | **Method overview**: `docs/METHODS_SUMMARY.md`

**Visualization output**: `outputs/figures/empathy_analysis_results.png` (empathy frequency comparison across consultations)

**How to run**: `pip install -r requirements.txt` → `python src/empathy_analysis.py`

---

## **Application Value**

**Healthcare communication assessment**: Quantify empathetic expression, surface improvement areas, support physician training.

**Service quality improvement**: Monitor empathy trends, evaluate interventions, enhance patient satisfaction.

---

**Keywords**: empathy detection, NLP, doctor–patient communication, Chinese medical text, machine learning

**Project status**: Core functionality complete, ongoing optimization
