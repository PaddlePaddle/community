# PaddleOCR Global Derivative Model Challenge — Detailed Scoring Rubric

**Total: 100 points across 6 dimensions**

---

## Dimension 1: Evaluation Dataset Quality (20 pts)

**Objective:** Evaluation datasets should reliably, stably, and objectively assess model capabilities. High-quality datasets require: clear task definition, high annotation quality, sufficient sample size, and realistic distribution. Recommended scale: >=100 document pages (500+ recommended); >=300 OCR instances (1000+ recommended). Excellent datasets typically feature real-world captured data, complex visual conditions, and real noise (e.g., photographed documents, scanned documents, blurred images, occlusions, real handwriting). Common deductions: only downloading public data from the internet, insufficient data volume, annotation errors, severe data duplication, low data diversity.

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 1.1 Data Scale — Whether data volume is sufficient for stable evaluation | 5 | **5 pts:** Images >=500 AND annotations >=1000; or images >=800; or annotations >=2000 <br> **4 pts:** Images >=300 AND annotations >=600; or images >=500; or annotations >=1000 <br> **3 pts:** Images >=200 AND annotations >=400; or images >=300; or annotations >=600 <br> **2 pts:** Images >=100 AND annotations >=300; or images >=200; or annotations >=400 <br> **1 pt:** Images >=50 AND annotations >=150; or images >=100; or annotations >=300 <br> **0 pts:** Images <50 AND annotations <150 |
| 1.2 Annotation Accuracy — Whether annotations are correct and free of obvious errors | 5 | **5 pts:** Accuracy >=98%, no obvious errors, with quality audit report <br> **4 pts:** Accuracy >=95%, few errors not affecting evaluation <br> **3 pts:** Accuracy >=90%, some errors but acceptable <br> **2 pts:** Accuracy >=80%, many errors <br> **1 pt:** Accuracy >=70%, severely affecting evaluation <br> **0 pts:** Accuracy <70% or no annotation documentation |
| 1.3 Data Diversity — Whether it covers multiple real-world visual conditions | 5 | **5 pts:** Covers >=4 real visual conditions (e.g., photographed, scanned, blurred, occluded, handwritten, lighting variations) with reasonable distribution <br> **4 pts:** Covers 3 visual conditions <br> **3 pts:** Covers 2 visual conditions <br> **2 pts:** Covers 1 visual condition <br> **1 pt:** Basically no diversity (e.g., all clear scanned documents) <br> **0 pts:** No diversity description |
| 1.4 Difficulty Reasonableness — Whether it includes easy/medium/hard levels | 5 | **5 pts:** Clear difficulty levels (easy/medium/hard), reasonable distribution (e.g., 30% easy, 40% medium, 30% hard) <br> **4 pts:** Has difficulty levels but unbalanced distribution <br> **3 pts:** Has difficulty levels but significant distribution bias (e.g., 90% easy) <br> **2 pts:** Mentions difficulty but no classification or distribution <br> **1 pt:** No difficulty consideration, but samples naturally vary <br> **0 pts:** Samples too simple or too difficult, cannot reflect true model capability |

---

## Dimension 2: Scenario Scarcity (15 pts)

**Objective:** Encourage exploration of real-world application scenarios that current OCR/document parsing research rarely addresses. Scenario examples — Common (not recommended): regular text recognition, regular table recognition. More valuable: medical report recognition, handwritten form recognition, handwritten formula recognition, flowchart recognition, organic chemistry formula recognition, ID card key info extraction, Tibetan recognition, Arabic recognition. Excellent cases usually feature: industry demand + current OCR cannot solve effectively (e.g., medical prescriptions, ancient texts, handwriting, minority languages).

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 2.1 Research Scarcity — Whether the scenario lacks public benchmarks or related research in academia | 6 | **6 pts:** No public benchmark at all, <2 related papers, emerging blank field <br> **5 pts:** Almost no public benchmark, only 2–4 related papers, no recognized evaluation standard <br> **4 pts:** A few related benchmarks exist but don't fully match (different domain/task definition), requiring significant adaptation <br> **3 pts:** Some related benchmarks, partially reusable, but still need substantial custom data <br> **2 pts:** Fairly close benchmarks exist, but with some task differences (different language, document type) <br> **1 pt:** Mature benchmarks exist and task is highly similar, only fine-tuning needed <br> **0 pts:** Standard benchmarks exist and fully overlap with existing OCR tasks |
| 2.2 Industrial Demand Value — Whether the scenario solves real industry problems with practical application value | 7 | **7 pts:** Core industry demand, numerous business pain points, extremely high application value, clear commercialization potential <br> **6 pts:** Clear industry demand, multiple enterprises/institutions have urgent needs, but not yet a broad market <br> **5 pts:** Real industry demand but narrower scope (specific vertical domain), solving it significantly improves efficiency <br> **4 pts:** Potential demand exists but not fully validated, needs exploration <br> **3 pts:** Demand is vague, only a few users mention it, or "nice-to-have" application <br> **2 pts:** Theoretical application possible but difficult to implement or low value <br> **1 pt:** Unclear demand or high overlap with existing solutions, no significant incremental value <br> **0 pts:** No industrial demand, purely academic/fictional scenario |
| 2.3 Scenario Uniqueness — Whether the scenario is significantly different from existing OCR tasks | 2 | **2 pts:** Significantly unique (new document type, new task definition, new interaction method) <br> **1 pt:** Some uniqueness, but largely within existing task scope (e.g., variant of existing task, new language but same task) <br> **0 pts:** Identical to existing OCR tasks, no uniqueness |

---

## Dimension 3: Task Complexity (15 pts)

**Objective:** Tasks should have reasonable challenge. Not encouraged: simple text recognition. Encouraged: multi-task joint tasks, key info extraction, document VQA, chart QA. High-scoring cases usually: contain multiple subtasks, involve document structure understanding (e.g., chart recognition + chart QA, multi-ethnic language recognition such as Tibetan + Mongolian + Uyghur, multilingual table recognition).

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 3.1 Visual Complexity — Whether document structure, image background, etc. are complex | 7 | **7 pts:** Contains >=4 image challenges (blur, occlusion, lighting, perspective, handwriting, cluttered background) with high proportion <br> **6 pts:** 3 challenges <br> **5 pts:** 3 challenges but some with low proportion <br> **4 pts:** 2 challenges <br> **3 pts:** 2 challenges but weak degree <br> **2 pts:** 1 challenge <br> **1 pt:** 1 challenge but weak <br> **0 pts:** No challenge (e.g., clear black text on white background) |
| 3.2 Structural Complexity — Whether multi-task joint optimization is needed | 5 | **5 pts:** Requires multi-task joint optimization (e.g., table recognition + QA, document structure understanding + info extraction), clear structural design with demonstration <br> **4 pts:** Single structural understanding task (e.g., table structure restoration, layout analysis), clear and complete description <br> **3 pts:** Some structural understanding (e.g., identifying titles vs. body text), basic but incomplete description <br> **2 pts:** Very simple structure (e.g., single-line text), or has structure but unclear description <br> **1 pt:** Attempts to describe structure but with obvious errors/contradictions <br> **0 pts:** No mention of structural complexity |
| 3.3 Understanding Complexity — Whether semantic understanding and reasoning are needed | 3 | **3 pts:** Requires semantic reasoning (e.g., document VQA, chart QA) <br> **2 pts:** Requires some semantic understanding (e.g., key info extraction) <br> **1 pt:** Simple semantics (e.g., text classification) <br> **0 pts:** No understanding, recognition only |

---

## Dimension 4: Training Data Construction Rigor (20 pts)

**Objective:** Ensure training dataset construction is scientific, reproducible, and extensible.

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 4.1 Collection Process Standardization — Whether data source is clear | 5 | **5 pts:** Clear data source, detailed collection method, provides key code/tools, reproducible <br> **4 pts:** Clear source, fairly complete description, missing some details <br> **3 pts:** Basically clear source, average description <br> **2 pts:** Vague source, incomplete description <br> **1 pt:** Unclear source, simple statement only <br> **0 pts:** Not provided |
| 4.2 Annotation Standard Completeness — Whether annotation guideline is complete | 5 | **5 pts:** Detailed annotation guideline with handling rules for various situations, reasonable <br> **4 pts:** Has guideline but not detailed enough <br> **3 pts:** Has annotation notes but not formalized <br> **2 pts:** Only simple rules <br> **1 pt:** No annotation standards, data only <br> **0 pts:** No standards and chaotic annotations |
| 4.3 Quality Control Mechanism — Whether audit process exists | 5 | **5 pts:** Clear systematic audit process including multi-round QC, error rate statistics, sampling ratios, with QC reports or data proof <br> **4 pts:** Has audit process but not systematic (e.g., single manual check), fairly complete description lacking quantitative data <br> **3 pts:** Mentions quality control but simple description (e.g., "we did manual sampling"), no specific process or data <br> **2 pts:** Only mentions "quality control was performed" without any specific measures <br> **1 pt:** Claims quality control but measures clearly unreasonable or insufficient <br> **0 pts:** No mention of quality control |
| 4.4 Data Statistical Analysis — Whether dataset analysis is provided | 5 | **5 pts:** Detailed analysis report with data distribution, category statistics, difficulty analysis, visualizations, in-depth analysis <br> **4 pts:** Basic statistics (counts, category distribution) but incomplete or lacking visualization <br> **3 pts:** Some statistics but not comprehensive (just totals and simple categories), simple description <br> **2 pts:** Only total count (e.g., "500 images total") or a few numbers without distribution <br> **1 pt:** Claims "data analysis was performed" but no results or irrelevant data <br> **0 pts:** No mention of data statistical analysis |

---

## Dimension 5: Model Fine-tuning Strategy & Innovation (10 pts)

**Objective:** Encourage exploration of effective fine-tuning methods. Baseline: full-parameter SFT, LoRA. Possible directions: multi-stage training, reinforcement learning, etc.

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 5.1 Fine-tuning Strategy Reasonableness — Whether training process is scientific | 5 | **5 pts:** Scientific strategy design, comparative experiments, optimal method selection (full-param, LoRA, etc.) with sufficient justification <br> **4 pts:** Reasonable strategy but insufficient justification or lacking comparisons <br> **3 pts:** Basically reasonable strategy but no selection rationale <br> **2 pts:** Strategy has flaws (e.g., overfitting or underfitting) <br> **1 pt:** Unreasonable strategy <br> **0 pts:** Not described |
| 5.2 Experiment Thoroughness — Whether systematic experiments were conducted | 3 | **3 pts:** Systematic experiments including different hyperparameters, data volumes, thorough result analysis <br> **2 pts:** Partial experiments but not systematic <br> **1 pt:** Only one result, no comparison <br> **0 pts:** No experiment description |
| 5.3 Technical Innovation — Whether new methods are proposed | 2 | **2 pts:** Proposed new fine-tuning method or improvement, verified effective <br> **1 pt:** Minor innovation or improvement <br> **0 pts:** No innovation, standard methods only |

---

## Dimension 6: Technical Documentation & Open Source Contribution (20 pts)

**Objective:** Encourage open ecosystem building.

| Sub-item | Max Score | Scoring Criteria |
|----------|-----------|-----------------|
| 6.1 Documentation Quality — Whether documentation is complete and clear | 5 | **5 pts:** Clear and complete documentation including project intro, installation, usage, training, evaluation — all steps, easy to understand <br> **4 pts:** Fairly complete but missing some details <br> **3 pts:** Basically usable but not detailed enough <br> **2 pts:** Simple documentation, only main steps <br> **1 pt:** Chaotic documentation, hard to understand <br> **0 pts:** No documentation |
| 6.2 Code Reproducibility — Whether experiments can be reproduced | 5 | **5 pts:** Clear code structure, complete training and evaluation scripts, necessary environment config, results reproducible <br> **4 pts:** Basically complete code but needs some debugging <br> **3 pts:** Partially missing code or cannot run directly <br> **2 pts:** Chaotic code, hard to use <br> **1 pt:** Only code fragments <br> **0 pts:** No code provided |
| 6.3 Demo Completeness — Whether visualization or online demo is provided | 5 | **5 pts:** Interactive online demo (e.g., Hugging Face Spaces) or local GUI for easy user experience <br> **4 pts:** Command-line demo or screenshots <br> **3 pts:** Example code but no intuitive demonstration <br> **2 pts:** Only describes demo effects <br> **1 pt:** Mentions demo but not provided <br> **0 pts:** No demonstration |
| 6.4 Community Contribution Value — Whether it has long-term value for the community | 5 | **5 pts:** Long-term community value, solves universal problems, widely reusable, high model downloads <br> **4 pts:** Some value but limited audience <br> **3 pts:** Has value but narrow scope <br> **2 pts:** Average value <br> **1 pt:** Low value <br> **0 pts:** No value |

---

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
