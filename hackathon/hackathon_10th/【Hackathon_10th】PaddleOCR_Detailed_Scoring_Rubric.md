# PaddleOCR Global Derivative Model Challenge — Detailed Scoring Rubric

**Total: 100 points across 6 dimensions**

Important Note: ‼️ If synthetic data accounts for an excessively high proportion of the evaluation set, once discovered, the evaluation set will be directly scored as 0, resulting in a **one‑vote veto**. Under these circumstances, the other scoring dimensions will no longer carry any meaning, and the participant will be disqualified from any subsequent rankings or awards. Please ensure the authenticity and diversity of your evaluation set.

## Summary of Point Allocation

| Dimension | Points |
|-----------|--------|
| 1. Evaluation Dataset Quality | 20 |
| 2. Scenario Scarcity | 15 |
| 3. Task Complexity | 15 |
| 4. Training Data Construction Rigor | 20 |
| 5. Model Fine-tuning Strategy & Innovation | 10 |
| 6. Technical Documentation & Open Source Contribution | 20 |
| **Total** | **100** |

---

## Dimension 1: Evaluation Dataset Quality (20 pts)

**Objective:** Evaluation datasets should reliably, stably, and objectively assess model capabilities. High-quality datasets require: clear task definition, high annotation quality, sufficient sample size, and realistic distribution. Recommended scale: >=100 document pages (500+ recommended); >=300 OCR instances (1000+ recommended). Excellent datasets typically feature real-world captured data, complex visual conditions, and real noise (e.g., photographed documents, scanned documents, blurred images, occlusions, real handwriting). Common deductions: only downloading public data from the internet, insufficient data volume, annotation errors, severe data duplication, and low data diversity.

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 1.1 Data Scale — Whether data volume is sufficient for stable evaluation | 5 |**High-score tendency**: Data volume ≥ 1000, all real collected data (no copyright issues), strong evaluation capability <br> **Low-score tendency**: Data volume ≤ 100, all low-quality synthetic data, weak evaluation capability (insufficient to measure the model's true capability) |
| 1.2 Annotation Accuracy — Whether annotations are correct and free of obvious errors | 5 | **High-score tendency**: No obvious annotation errors, provides annotation quality report or visualization report <br> **Low-score tendency**: Many annotation errors exist that cannot be filtered out, affecting model evaluation |
| 1.3 Data Diversity — Whether it covers multiple real-world visual conditions | 5 | **High-score tendency**: Data type variance is large enough, includes multiple real-world capture scenarios, can effectively evaluate the model's true capability and generalization <br> **Low-score tendency**: Data type is single, variance is extremely small, cannot evaluate the model's true capability and generalization |
| 1.4 Difficulty Reasonableness — Whether it includes easy/medium/hard levels | 5 | **High-score tendency**: Data difficulty distribution is balanced, conforms to the data distribution in real vertical domain scenarios, can effectively evaluate the model's capability at different difficulty levels <br> **Low-score tendency**: Data difficulty is single (all easy or all hard samples), does not conform to the distribution in real vertical domain scenarios, cannot evaluate the model's true capability |

---

## Dimension 2: Scenario Scarcity (15 pts)

**Objective:** Encourage exploration of real-world application scenarios that current OCR/document parsing research rarely addresses. Scenario examples — Common (not recommended): regular text recognition, regular table recognition. More valuable: medical report recognition, handwritten form recognition, handwritten formula recognition, flowchart recognition, organic chemistry formula recognition, ID card key info extraction, Tibetan recognition, and Arabic recognition. Excellent cases usually feature: industry demand + current OCR cannot solve effectively (e.g., medical prescriptions, ancient texts, handwriting, minority languages).

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 2.1 Research Scarcity — Whether the scenario lacks public benchmarks or related research in academia | 6 | **High-score tendency**: No public dataset exists for this scenario in academia or industry; research field is scarce <br> **Low-score tendency**: Sufficient public datasets already exist in academia and industry; poor scarcity |
| 2.2 Industrial Demand Value — Whether the scenario solves real industry problems with practical application value | 7 | **High-score tendency**: This scenario addresses a core industry need with numerous real-world pain points to be solved; extremely high application value <br> **Low-score tendency**: This scenario is fictitious; the pain points do not actually exist; no industry need at all |
| 2.3 Scenario Uniqueness — Whether the scenario is significantly different from existing OCR tasks | 2 | **High-score tendency**: Significantly unique compared to existing OCR tasks (e.g., entirely new document types, new task definitions, new interaction methods) <br> **Low-score tendency**: Identical to existing OCR tasks; no uniqueness whatsoever. |

---

## Dimension 3: Task Complexity (15 pts)

**Objective:** Tasks should have a reasonable challenge. Not encouraged: simple text recognition. Encouraged: multi-task joint tasks, key info extraction, document VQA, and chart QA. High-scoring cases usually contain multiple subtasks, involve document structure understanding (e.g., chart recognition + chart QA, multi-ethnic language recognition such as Tibetan + Mongolian + Uyghur, multilingual table recognition).

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 3.1 Visual Complexity — Whether document structure, image background, etc., are complex | 7 | **High-score tendency**: Contains ≥ 4 types of real-capture scenarios (blur, occlusion, lighting, handwriting, creases, etc.) with reasonable proportions <br> **Low-score tendency**: No visual complexity challenges at all (e.g., all clear PDF images) |
| 3.2 Structural Complexity — Whether multi-task joint optimization is needed | 5 | **High-score tendency**: Task requires multi-task joint optimization (e.g., table recognition + KIE, OCR + information extraction), provides clear structural design explanation, and clarifies how multiple tasks are collaboratively optimized <br> **Low-score tendency**: Task is single-task, with no mention of information related to task structural complexity, or the provided description contains obvious errors |
| 3.3 Understanding Complexity — Whether semantic understanding and reasoning are needed | 3 | **High-score tendency**: Task requires semantic reasoning (e.g., document visual question answering, chart KIE), involving document content understanding <br> **Low-score tendency**: Task does not involve any content understanding, only recognizing layout elements. |

---

## Dimension 4: Training Data Construction Rigor (20 pts)

**Objective:** Ensure training dataset construction is scientific, reproducible, and extensible.

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 4.1 Collection Process Standardization — Whether data source is clear | 5 | **High-score tendency**: Clear data source, detailed collection method, no copyright issues, provides reproducible key code/tools <br> **Low-score tendency**: Unclear or missing data source, copyright disputes, does not provide reproducible key code/tools |
| 4.2 Annotation Standard Completeness — Whether the annotation guideline is complete | 5 | **High-score tendency**: Provides detailed annotation guidelines and processes, including handling rules for various cases, with reasonable content <br> **Low-score tendency**: No annotation-related explanation, or annotation rules are unreasonable |
| 4.3 Quality Control Mechanism — Whether audit process exists | 5 | **High-score tendency**: Has a clear, systematic quality control process with explanation (e.g., manual inspection, rule-based automatic inspection, etc.), detailed description, provides relevant QC data <br> **Low-score tendency**: No quality control performed or explained, serious deficiencies in the QC process |
| 4.4 Data Statistical Analysis — Whether dataset analysis is provided | 5 | **High-score tendency**: Provides a detailed data analysis report including data distribution, category statistics, difficulty analysis, etc., with visual explanations <br> **Low-score tendency**: No data statistical analysis performed or explained, or the provided report is seriously inconsistent with the actual situation |

---

## Dimension 5: Model Fine-tuning Strategy & Innovation (10 pts)

**Objective:** Encourage exploration of effective fine-tuning methods. Baseline: full-parameter SFT, LoRA. Possible directions: multi-stage training, reinforcement learning, etc.

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 5.1 Fine-tuning Strategy Reasonableness — Whether the training process is scientific | 5 | **High-score tendency**: Strategy is scientifically designed with comparative experiments, selects the most appropriate fine-tuning method (e.g., full-parameter, LoRA, etc.), and features task-specific, unique fine-tuning strategies <br> **Low-score tendency**: Strategy is unreasonable without explanation, and no dedicated research for the task has been conducted |
| 5.2 Experiment Thoroughness — Whether systematic experiments were conducted | 3 | **High-score tendency**: Systematic experiments are performed, including but not limited to comparisons of different hyperparameters, data ratios, and experimental strategies, with detailed result analysis <br> **Low-score tendency**: No systematic experiments or explanation provided; only simple fine-tuning attempts |
| 5.3 Technical Innovation — Whether new methods are proposed | 2 | **High-score tendency**: Proposes a unique and effective fine-tuning method tailored to the specific task, validated through sufficient comparative experiments <br> **Low-score tendency**: No methodological innovation; only standard fine-tuning processes are used

 |

---

## Dimension 6: Technical Documentation & Open Source Contribution (20 pts)

**Objective:** Encourage open ecosystem building.

| Sub-item | Max Score | Scoring Tendencies |
|----------|-----------|-----------------|
| 6.1 Documentation Quality — Whether documentation is complete and clear | 5 | **5 pts:** Clear and complete documentation including project intro, installation, usage, training, evaluation — all steps, easy to understand <br> **4 pts:** Fairly complete but missing some details <br> **3 pts:** Basically usable but not detailed enough <br> **2 pts:** Simple documentation, only main steps <br> **1 pt:** Chaotic documentation, hard to understand <br> **0 pts:** No documentation |
| 6.2 Code Reproducibility — Whether experiments can be reproduced | 5 | **5 pts:** Clear code structure, complete training and evaluation scripts, necessary environment config, results reproducible <br> **4 pts:** Basically complete code but needs some debugging <br> **3 pts:** Partially missing code or cannot run directly <br> **2 pts:** Chaotic code, hard to use <br> **1 pt:** Only code fragments <br> **0 pts:** No code provided |
| 6.3 Demo Completeness — Whether a visualization or online demo is provided | 5 | **5 pts:** Interactive online demo (e.g., Hugging Face Spaces) or local GUI for easy user experience <br> **4 pts:** Command-line demo or screenshots <br> **3 pts:** Example code but no intuitive demonstration <br> **2 pts:** Only describes demo effects <br> **1 pt:** Mentions demo but not provided <br> **0 pts:** No demonstration |
| 6.4 Community Contribution Value — Whether it has long-term value for the community | 5 | **5 pts:** Long-term community value, solves universal problems, widely reusable, high model downloads <br> **4 pts:** Some value but limited audience <br> **3 pts:** Has value but narrow scope <br> **2 pts:** Average value <br> **1 pt:** Low value <br> **0 pts:** No value |

---
