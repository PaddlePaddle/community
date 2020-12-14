# PPSIG Serving Charter

This charter adheres to the conventions described in the [PaddlePaddle Charter Guide](/ppsigs/toc-ppsig-guide/PPSIG_CHARTER_GUIDE.md) and uses
the Roles and Organization Management outlined in [PPSIG-governance](/GOVERNANCE.md).

## Scope

PPSIG Serving’s mission is to simplify, develop, deployg and maintain PaddlePaddle Serving project.

### In scope

#### Areas of Focus

- When you have trained a deep neural net with PaddlePaddle, you are also capable to deploy the model online easily. 
- Industrial serving features supported, such as models management, online loading, online A/B testing etc.
- Support large scale sparse features as model inputs.
- Multiple programming languages supported on client side, such as Golang, C++ and python.

#### Code, Binaries and Services

- all [Serving repositories](https://github.com/PaddlePaddle/Serving) under the PaddlePaddle organization
- all the PPSUBSIGs formerly owned by [PPSUBSIG-Benchmark](), [PPSUBSIG-Documentation](), [PPSIG-Dashboard](), [PPSIG-Model]().
- any new PPSUBSIG that is Serving specific, unless there is another PPSIG already sponsoring it.

#### Cross-cutting and Externally Facing Processes

- This PPSIG works with PPSUBSIG Benchmark to ensure that Servings are actively testing, benchmarking & reporting results to PaddlePaddle Serving.
- This PPSIG works with PPSUBSIG Docs to provide user-facing documentation on configuring PaddlePaddle Serving with integration enabled.
- This PPSIG works with new Serving components in the PaddlePaddle ecosystem that want to host their code in the PaddlePaddle organization and have an interest in contributing back.
- This PPSIG actively engages with PPSIGs owning other external components of PaddlePaddle (Pipelines, OCR, Detection) to ensure a consistent integration story for users.

## Roles and Organization Management

This PPSIG follows adheres to the Roles and Organization Management outlined in [PPSIG-governance](/GOVERNANCE.md)
and opts-in to updates and modifications to [PPSIG-governance](/GOVERNANCE.md).

### Additional responsibilities of Chairs/PMs

- Selecting/prioritizing work to be done for a milestone
- Hosting the weekly or biweekly PPSIG meeting, ensure that recordings are uploaded.
- Organizing PPSIG sessions at PaddlePaddle events (intro/deep dive sessions).
- Creating roadmaps for a given year or release, or reviewing and approving technical implementation plans in coordination with other PPSIGs.

### Deviations from [PPSIG-governance](/GOVERNANCE.md)

- As PPSIG Serving contains a number of PPSUBSIGs, the PPSIG has empowered PPSUBSIG leads with a number of additional responsibilities, including but not limited to:
    * Releases: The PPSUBSIG owners are responsible for determining the PPSUBSIG release cycle, producing releases, and communicating releases with PPSIG Release and any other relevant PPSIG.
    * Backlog: The PPSUBSIG owners are responsible for ensuring that the issues for the PPSUBSIG are correctly associated with milestones and that bugs are resolved.
PR timeliness: The PPSUBSIG owners are responsible for ensuring that active pull requests for the PPSUBSIG are addressed.
    * Repository ownership: The PPSUBSIG owners are given admin permissions to repositories under the PPSUBSIG. 
