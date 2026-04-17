# 【Hackathon_10th】PaddleOCR Global Derivative Model Challenge

## Competition Adjustments (April 17)

Dear participants, Thank you for your support and interest in this competition. The following adjustments have been made to the schedule and rules:
1. **Schedule Extension**: The overall timeline has been postponed by half a month. The submission deadline for the preliminary round is extended to May 29.
1. **Evaluation Set Submission Rules**:
   - The previous deadline of April 24 is no longer applicable. You may submit or update your evaluation set at any time before the preliminary deadline (May 29).
   - Important Note: ‼️ If synthetic data accounts for an excessively high proportion of the evaluation set, once discovered, the evaluation set will be directly scored as 0, resulting in a **one‑vote veto**. Under these circumstances, the other scoring dimensions will no longer carry any meaning, and the participant will be disqualified from any subsequent rankings or awards. Please ensure the authenticity and diversity of your evaluation set.
1. **Leaderboard Release and Feedback Mechanism**:
   - When each leaderboard is released, your overall score will be published, and each participant will receive a detailed dimension score along with personalized improvement suggestions from mentors via email.
   - The earlier you submit your work, the earlier you will receive feedback, helping you continuously optimize your model and dataset to improve your ranking.
1. **Scoring Sheet Optimization**: The detailed scoring sheet has been revised to use descriptive labels (e.g., "high‑score tendencies" and "low‑score tendencies") instead of purely numeric values, making it easier to understand the evaluation criteria and areas for improvement.
1. **Award Score Threshold Adjustments**:
   - Top 10 finalists: Minimum score thresholds are set for all awards. Your total score must be ≥ 60 points to enter the Top 10 ranking and be eligible for corresponding awards.
   - High‑Quality Evaluation Set Contribution Award: This award is evaluated independently. Your score in the "Evaluation Set Quality" dimension must be ≥ 12 points (out of 20) to be eligible.

For the latest competition process, complete rules, and detailed scoring criteria, please refer to the content below. Thank you for your understanding and support. We wish you excellent results!

## 1. Competition Background
With the rapid advancement of large model technologies, the OCR field is undergoing a new wave of technological transformation. ERNIE, as Baidu's core AI capability, already possesses powerful general visual understanding abilities, but there remains enormous room for optimization and customization in specialized scenarios.

The industry currently faces three major challenges:

* **Strong demand for long-tail scenarios**: Specialized fields such as ancient manuscript recognition, minority language OCR, handwritten formulas, and medical receipts lack high-quality dedicated models;
* **Weak derivative model ecosystem**: The barrier for developers to build upon base models is high, with a lack of systematic guidance and support;
* **Scarce evaluation data**: High-quality scenario-specific evaluation datasets are in short supply, severely limiting the pace of model iteration and optimization.

To address these challenges and drive the evolution of OCR technology from "generally usable" to "scenario-excellent", this competition was created. We encourage participants to build on **PaddleOCR-VL series models** (e.g., PaddleOCR-VL, PaddleOCR-VL-1.5, etc.), focus on real-world scenarios, construct new **document parsing / OCR / document understanding** tasks, and solve practical problems through model fine-tuning.

## 2. Challenge Topics
This competition focuses on developing derivative models in OCR, encouraging participants to define their own task directions based on real-world scenarios and to solve practical problems through model fine-tuning.

* **Reference scenario pool** (example directions provided by the organizer, for reference only, not mandatory): Ancient manuscript recognition, chart question answering, ethnic language recognition, minority language recognition, handwritten text recognition, handwritten formula recognition, artistic text recognition, regionOCR, ID information extraction, complex table recognition, receipt recognition, K12 exam paper recognition, flowchart recognition, code text recognition, organic chemical formula recognition, complex document reading order restoration, medical table recognition, SVG recognition, etc.
* **Open topic selection**: Participants are not limited to the above examples and may independently choose any OCR sub-field of interest for model fine-tuning. Innovation and exploration are encouraged.

### Scenario Value Guidance
* **Common scenarios (not recommended)**: Scenarios with mature existing solutions, such as standard text recognition and standard table recognition.
* **Higher-value scenarios**: Medical report recognition, handwritten form recognition, handwritten formula recognition, flowchart recognition, organic chemical formula recognition, key information extraction from licenses/IDs, Tibetan text recognition, Arabic text recognition, etc.
* **Characteristics of outstanding submissions**: They typically address industry-critical needs that existing OCR methods struggle to solve effectively, such as medical prescription recognition, ancient manuscript recognition, handwritten text recognition, and minority language recognition.

We look forward to participants focusing on real-world application areas not yet fully covered by academia or industry, tackling these "hard nut" scenarios, and producing derivative models with both technical depth and practical value.

## 3. Timeline

| Phase | Date | Key Tasks | Submission Notes |
|---|---|---|---|
| **Preliminary Round** | April 1 – May 29 | Participants submit their work; rankings are published every two weeks | **Evaluation set、Training data report and open-source project** (GitHub + Hugging Face): Must be submitted by **May 29**. |
|Leaderboard Release (Round 1)|Friday, April 17|Official leaderboard: Top 50 Overall Scores (including total points)||
|Leaderboard Release (Round 2)|Monday, May 11|Official leaderboard: Top 50 Overall Scores (including total points)||
|Leaderboard Release (Round 3)|Monday, May 25|Official leaderboard: Top 50 Overall Scores (including total points)||
|**Preliminary Review**|May 31 – June 9|Review completed based on 6-dimension scoring; advancement list determined||
|**Preliminary Results Announcement**|Monday, June 9|**Final preliminary leaderboard** published (Top 10 advancement list + 10 High-Quality Data Contribution Award winners)||
|**Finals Preparation**|June 10– June 20|Advancing teams prepare presentation materials and submit final versions|Advancing teams must submit the final training data construction report, open-source project (GitHub + Hugging Face), and presentation slides by **June 20**|
|**Finals Defense**|June 24 – June 26|In-person or online defense with on-site scoring||
|**Results Announcement**|June 30|**Award winners list** published||

### Notes
* **All materials** (Evaluation set、training data construction report, complete open-source project): May be submitted early to participate in bi-weekly leaderboard rankings, and can be continuously iterated until **May 29**; the final version is determined by the last submission before the deadline.
* After each leaderboard release, participants will receive an email containing detailed scores and mentor feedback. The earlier you submit, the sooner you receive feedback, helping you iteratively improve your model and dataset.
* If advancing to the finals, the final optimised version of the above materials may be submitted during the finals preparation phase (before June 20).
* All submitted materials must comply with the specifications in the "Submission Requirements" section.
* The format of the finals defence (in-person/online) will be announced separately based on actual circumstances.

## 4. Prizes
The total cash prize pool for this competition is **70,000 RMB**, divided into preliminary round incentives and finals prizes. Cash prizes are pre-tax amounts; personal income tax will be withheld and remitted by the organiser before disbursement. In addition, participants have the opportunity to receive physical gifts and various ecosystem benefits.

### 1. Preliminary Round Incentives

| Incentive | Quantity | Description |
|---|---|---|
| **Award Certificate (Digital)** | Unlimited | All participants who submit a complete work (open-source model link) will receive a digital participation certificate. |
| **Baidu Merchandise** | 50 sets | Participants ranked in the Top 50 by composite score will receive Baidu custom merchandise. |
| **High-Quality Evaluation Set Contribution Award** | 10 people | 1,000 RMB cash reward per person; evaluated by the R&D team based on the "Evaluation Set Quality" dimension and awarded to the top 10 participants who score **≥ 12 points** in this dimension and achieve the highest overall quality. |
| **Universal Computing Resource Package** | Unlimited | Participants who submit a project link will receive GPU computing credits worth 100 RMB from PaddlePaddle AI Studio, to support project creation and model tuning. |

### 2. Finals Prizes

The Top 10 teams advance to the finals defence, and the following cash prizes (total **60,000 RMB**) are awarded:

| Prize | Winners | Prize per Person | Subtotal | Minimum Total Score (out of 100) |Description|
|---|:--:|--:|--:|--:|--:|
| 1st Place (Champion) | 1 | 20,000 RMB | 20,000 RMB | ≥ 75 | Demonstrates outstanding technical innovation, complete open‑source contributions, and a high‑quality evaluation set.| 
| 2nd Place (Runner-up) | 1 | 12,000 RMB | 12,000 RMB |≥ 70 |Excellent overall capability with outstanding performance across all dimensions.| 
| 3rd Place | 1 | 8,000 RMB | 8,000 RMB |≥ 65 |Strong overall capability with no obvious weaknesses in any dimension.| 
| 4th–6th Place | 3 | 4,000 RMB | 12,000 RMB |≥ 60 |High completeness with good performance in most dimensions.| 
| 7th–10th Place | 4 | 2,000 RMB | 8,000 RMB |≥ 60 |Complete submission with notable strengths in at least one dimension.| 
| **Total** | **10** | | **60,000 RMB** | || 

**Dynamic Award Distribution Rules for the Final Round**:
* The final ranking is determined by the jury based on the participants’ final submissions and oral defence presentations, evaluated across six dimensions (a total of 100 points).
* If the **highest score among the top 6 finalists is below 65 points** (i.e., no participant reaches the 3rd‑place minimum threshold), then the Champion, Runner‑Up, and 3rd Place awards will all be cancelled. The top 6 participants will instead each receive the award amount for 4th–6th place: RMB 4,000 per person.
* If at least one of the top 6 finalists scores **≥ 65 points**, the original award structure applies as usual (Champion ≥ 75, Runner‑Up ≥ 70, 3rd Place ≥ 65, 4th–10th Place ≥ 60).
* Awards for 4th–10th place also require a score of at least 60 points; otherwise, the corresponding awards will be left vacant.
* The **High‑Quality Evaluation Set Contribution Award** is evaluated independently and is not affected by the downgrading rule above.

### 3. Ecosystem Benefits

To encourage open-source co-creation, outstanding works and their authors will also receive the following ecosystem benefits:

* **Official Showcase and Promotion**: Outstanding award-winning works will be featured on the Hugging Face ERNIE Community topic page and Baidu's domestic and overseas social media channels. Additionally, some developers with notable contributions will receive public acknowledgment in release notes.
* **In-Depth Collaboration Opportunities**: Authors of outstanding works may have the opportunity to receive further guidance and collaboration from Baidu engineers to jointly explore technology application scenarios; particularly outstanding teams may have the opportunity to co-publish papers at top conferences.
* **ERNIE Community Ambassador Title**: Participants who perform exceptionally well will be awarded the "ERNIE Large Model Community Honorary Ambassador" title and included in Baidu's key developer ecosystem contact list.
* **AIStudio Community Computing and API Resources**: Outstanding participants can receive free GPU computing credits from PaddlePaddle AI Studio's AIStudio community and ERNIE API Token quotas, helping participants continue AI development and model iteration after the competition and reducing subsequent R&D hardware costs.
* **Internship Opportunities and Fast Track**: Outstanding student teams will have the opportunity to receive a **fast-track interview channel for the PaddleOCR team**, with excellent members given priority consideration to join the **PaddleOCR team** and participate in core technology R&D and deployment.

**Notes**:

* The High-Quality Data Contribution Award and finals prizes can be awarded cumulatively.
* All award results will be publicly announced on the official competition website and official communities. Please stay tuned.

## 5. Scoring Criteria
Both the preliminary round and finals use a unified scoring system that comprehensively evaluates participants' work across the following six dimensions, with a total of **100 points**. In addition, a separate **"High-Quality Evaluation Set Contribution Award"** is set, which is evaluated solely based on the "evaluation set quality" dimension and awarded to the top 10 participants who score **≥ 12 points** in this dimension and achieve the highest overall quality.

| Dimension | Full Score | Sub-items and Scores |
|---|:--:|---|
| **Evaluation Set Quality** | 20 pts | Data scale (5 pts), annotation accuracy (5 pts), data diversity (5 pts), difficulty reasonableness (5 pts) |
| **Scenario Scarcity** | 15 pts | Research scarcity (6 pts), industrial demand value (7 pts), scenario uniqueness (2 pts) |
| **Task Complexity** | 15 pts | Visual complexity (7 pts), structural complexity (5 pts), comprehension complexity (3 pts) |
| **Training Dataset Construction Rigor** | 20 pts | Collection process standardization (5 pts), annotation standard completeness (5 pts), quality control mechanisms (5 pts), data statistical analysis (5 pts) |
| **Model Fine-tuning Strategy and Innovation** | 10 pts | Fine-tuning strategy appropriateness (5 pts), experiment thoroughness (3 pts), technical innovation (2 pts) |
| **Technical Documentation and Open-Source Contribution** | 20 pts | Documentation quality (5 pts), code reproducibility (5 pts), demo completeness (5 pts), community contribution value (5 pts) |

For detailed scoring criteria for each dimension, please refer to the [Detailed Scoring Rubric](./【Hackathon_10th】PaddleOCR_Detailed_Scoring_Rubric.md).

### Special Note
The final submission deadline is **June 20**, which is more generous than the preliminary round deadline (May 29). Therefore, after the top 10 preliminary round participants advance to the finals, **the final rankings in the finals will be re-evaluated based on the quality of the finals submissions**; preliminary round scores will not be carried over to the finals.

## 6. Technical Support

To ensure participants can compete smoothly, the organizer provides the following technical support:

* **Official Documentation**: The R&D team will provide detailed official documentation, including model fine-tuning methods, training workflows, evaluation standards, etc., to ensure participants can operate independently. See the [PaddleOCR-VL Best Practice Cases](https://github.com/PaddlePaddle/PaddleFormers/tree/develop/examples/best_practices/PaddleOCR-VL) for reference.
* **Remote Technical Guidance**: If participants encounter issues not covered by the documentation, they can contact the official community or email, and the R&D team will provide remote technical guidance.

## 7. Submission Requirements
All submitted materials (except finals defense materials) should be sent via email to: **ext_paddle_oss@baidu.com + paddleocr@baidu.com + cuicheng01@baidu.com + liujiaxuan01@baidu.com**, with the subject formatted as: `PaddleOCR Derivative Model Challenge - [Material Name] - [GitHub ID]` (e.g., `PaddleOCR Derivative Model Challenge - Evaluation Set - zhangsan`).

| Submission Content | Deadline | Public? | Requirements |
|---|---|:--:|---|
| **Evaluation Set** | May 29 (Preliminary) / June 20 (Finals)  | Not required | images/documents + annotations + task description + evaluation script + dataset description (data sources, scale, category distribution, difficulty analysis). Host on Baidu Netdisk or [AI Studio Open Datasets](https://aistudio.baidu.com/datasetoverview) and submit the link. |
| **Training Data Construction Report** | May 29 (Preliminary) / June 20 (Finals) | Not required | PDF/Markdown. Must include: data collection methods (with key code), annotation specifications, annotation tools, and quality control workflows. |
| **Complete Open-Source Project** |May 29 (Preliminary) / June 20 (Finals)  | **Must be public** | **GitHub repository**: training/evaluation code, documentation, demo (training data not required to be open-sourced). **Hugging Face model**: fine-tuned model with complete model card; reference: [PaddleOCR-VL-For-Manga](https://huggingface.co/jzhang533/PaddleOCR-VL-For-Manga). |
| **Finals Defense Materials** | June 20 | — | Advancing teams only: presentation slides (10 min) + optional demo video (3–5 min). Submission method to be communicated separately. |

**Important Notes**:

* All submitted materials may be continuously iterated and improved before the deadline; the final version is determined by the last submission.
* All submitted materials must be original; plagiarism or infringement of others' intellectual property rights is strictly prohibited and will result in disqualification upon discovery.
* For any questions, please contact us via the official mailbox or discussion area.

## 8. Computing Resources
This competition does not provide computing resources (machines); participants must provide their own computing power for model fine-tuning and evaluation. Participants are advised to prioritize lightweight models in the preliminary round or use quantization techniques to reduce resource requirements.

## 9. Intellectual Property
The intellectual property of competition submissions (including but not limited to algorithms, models, etc.) is jointly owned by Baidu and the participants. The organizing committee has the right to use competition submissions, related works, and participating team information for promotional materials, related publications, media releases (designated and authorized), official website browsing and downloads, exhibitions (including touring exhibitions), and other activities. The competition organizing body retains priority partnership rights.

## 10. Important Notes
1. **Eligibility**: The competition is open to developers worldwide; individuals or teams (up to 5 members) may register. Baidu internal employees may receive rankings and honors normally, but do not participate in cash prize distribution per company policy.
2. **Participation Agreement**: Registration constitutes agreement to the participation agreement, which defines terms such as intellectual property ownership and data usage. Please read it carefully.
3. **International Participants**: Prize distribution for overseas participants involves foreign exchange management; participants will need to cooperate with the organizer in providing the necessary information. Please be aware of this in advance.
4. **Academic Integrity**:
   * Works must be original; plagiarism or infringement of others' intellectual property rights is strictly prohibited and will result in disqualification.
   * Participants are prohibited from exploiting rule loopholes or technical vulnerabilities to improve rankings; violations will result in score cancellation and serious disciplinary action.
   * Personnel who have had access to competition-related data will have their submissions excluded from leaderboard rankings and prize evaluation.
5. **Rule Adjustments**: The organizing committee reserves the right to adjust competition rules and event arrangements, determine and handle cheating behavior, and revoke or refuse to award prizes to teams that affect the organization and fairness of the competition.

## 11. Contact

* **Official Email**: ext_paddle_oss@baidu.com (competition inquiries)
* **GitHub Issue**: [PaddlePaddle/PaddleOCR#17858](https://github.com/PaddlePaddle/PaddleOCR/issues/17858)
* **GitHub Discussions**: [Derivative Model Challenge](https://github.com/PaddlePaddle/PaddleOCR/discussions/categories/derivative-model-challenge)
* **Official Website / PaddleOCR**: [paddleocr.com](https://paddleocr.com/)
* **PaddleOCR GitHub**: [github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
* **Domestic Community (China)**: [Click to Join](https://paddle.wjx.cn/vm/mVlHfvq.aspx#) <img src="https://github.com/user-attachments/assets/289b2f8b-9a87-4ef2-a187-47505d0b71c6" width="200px">
* **International Community**: [Discord](https://discord.gg/BzUBS6VXss)
