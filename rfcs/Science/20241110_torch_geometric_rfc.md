# Torch-Geometric 模块适配飞桨 RFC

|              |                                 |
| ------------ | ------------------------------- |
| 提交作者     | LilaKen                         |
| 提交时间     | 2024-11-10                      |
| RFC 版本号   | v1.0                            |
| 依赖飞桨版本 | develop                         |
| 文件名       | 20241110_torch_geometric_rfc.md |

---

## 1. 概述

### 1.1 相关背景

Torch-Geometric 是一个强大的图机器学习框架，提供了高效的数据操作和图神经网络模块，用于处理结构化数据任务。本 RFC 旨在为 Torch-Geometric 提供适配飞桨框架的支持，扩展其在 Paddle 生态系统中的使用场景，提升 Paddle 在图神经网络领域的功能覆盖。

### 1.2 功能目标

- 提供与 Torch-Geometric 相同的 API 接口，便于用户迁移。
- 优化数据加载、转换和图神经网络的实现，以适配 Paddle 的计算特性。
- 保障所有必要模块的单测通过，完成性能验证。

---

## 2. Torch-Geometric 的架构

Torch-Geometric 的架构设计清晰，按照功能分为以下主要模块：

1. **`data` 模块**：提供数据处理和存储的基本功能，包括图数据、异构图数据的定义和管理。
2. **`datasets` 模块**：提供主流数据集的封装和加载，支持多种公开数据集。
3. **`nn` 模块**：包括图神经网络的基本层、聚合操作、正则化操作、以及相关的模块化模型。
4. **`transforms` 模块**：用于对图数据进行变换，如增加自环、归一化、生成特征等。
5. **`loader` 模块**：支持批量加载数据的功能，包括常见的采样方法和多种场景的加载器。
6. **`sampler` 模块**：提供高效的邻居采样、边采样等功能，用于图数据的随机化处理。
7. **`utils` 模块**：包含常用的图操作函数，如图归并、节点采样、自环操作等。
8. **`profile` 模块**：提供性能分析工具，包括统计内存、时间等的辅助函数。
9. **`explain` 模块**：支持图神经网络的可解释性研究。

---

## 3. 公共 API 调研情况

以下为各模块的公开 API 列表以及是否需要反向传播支持：
表格最终结合第三part和第四part定档情况如下：

| **模块**   | **公开 API**                  | **需要反向** | **转换代码API** | **转换代码状态** | **单元测试代码文件名** |**单元测试代码状态** | **最终状态** |

### `utils` 模块

| **模块**   | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|-------------------------------|--------------|-----------------|------------------|------------------------|----------------------|--------------|
| utils      | scatter                       | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | group_argsort                 | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | segment                       | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | index_sort                    | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | cumsum                        | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | degree                        | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | softmax                       | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | lexsort                       | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | sort_edge_index               | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | coalesce                      | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | is_undirected                 | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | to_undirected                 | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | contains_self_loops           | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | remove_self_loops             | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | segregate_self_loops          | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | add_self_loops                | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | add_remaining_self_loops      | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | get_self_loop_attr            | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | contains_isolated_nodes       | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | remove_isolated_nodes         | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | get_num_hops                  | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | subgraph                      | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | bipartite_subgraph            | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | k_hop_subgraph                | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | dropout_node                  | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | dropout_edge                  | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | dropout_path                  | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | dropout_adj                   | 否           |                 | :heavy_check_mark:                |                        |                      |              |
| utils      | homophily                     | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | assortativity                 | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | get_laplacian                 | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | get_mesh_laplacian            | 否           |                 |  :heavy_check_mark:                |                        |                      |              |
| utils      | mask_select                   | 否           |                 | :heavy_check_mark:                 |                        |                      |              |
| utils      | index_to_mask                 | 否           |                 | :heavy_check_mark:                 |                        |                      |              |


### `data` 模块

| **模块**   | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| data       | batch                  | 否           |                  |     :heavy_check_mark:         |                |                |              |
| data       | collate                    | 否           |                  |    :heavy_check_mark:          |                |                |              |
| data       | data                    | 否           |                 |    :heavy_check_mark:          |                |                |              |
| data       | database                      | 否           |                  |  :heavy_check_mark:            |                |                |              |
| data       | datapipes                          | 否           |                 |   :heavy_check_mark:           |                |                |              |
| data       | dataset                    | 否           |                 |  :heavy_check_mark:            |                |                |              |
| data       | download                         | 否           |                  | :heavy_check_mark:             |                |                |              |
| data       | extract                  | 否           |                 |:heavy_check_mark:              |                |                |              |
| data       | feature_store                      | 否           |                  | :heavy_check_mark:             |                |                |              |
| data       | graph_store                | 否           |               |   :heavy_check_mark:           |                |                |              |
| data       | hetero                 | 否           |                |   :heavy_check_mark:           |                |                |              |
| data       | hypergraph_data                       | 否           |                  |  :heavy_check_mark:            |                |                |              |
| data       | in_memory_dataset               | 否           |                 |   :heavy_check_mark:           |                |                |              |
| data       | on_disk_dataset                 | 否           |                  |   :heavy_check_mark:           |                |                |              |
| data       | remote_backend_utils                      | 否           |                 |:heavy_check_mark:              |                |                |              |
| data       | separate                  | 否           |                 | :heavy_check_mark:             |                |                |              |
| data       | summary           | 否           |                  |:heavy_check_mark:              |                |                |              |
| data       | temporal                   | 否           |                  | :heavy_check_mark:             |                |                |              |
| data       | view                   | 否           |                  | :heavy_check_mark:             |                |                |              |



### `sampler` 模块

| **模块**   | **公开 API**               | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|----------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| sampler    | BaseSampler                | 否           |                 |:heavy_check_mark:              |                |                |              |
| sampler    | NodeSamplerInput           | 否           |                 |:heavy_check_mark:              |                |                |              |
| sampler    | EdgeSamplerInput           | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | SamplerOutput              | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | HeteroSamplerOutput        | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | NumNeighbors               | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | NegativeSampling           | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | NeighborSampler            | 否           |                 | :heavy_check_mark:             |                |                |              |
| sampler    | HGTSampler                 | 否           |                 |:heavy_check_mark:              |                |                |              |


### `loader` 模块

| **模块**   | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| loader     | DataLoader                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | NodeLoader                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | LinkLoader                    | 否           |                 |:heavy_check_mark:              |                |                |              |
| loader     | NeighborLoader                | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | LinkNeighborLoader            | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | HGTLoader                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | ClusterData                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | ClusterLoader                 | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | GraphSAINTSampler             | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | GraphSAINTNodeSampler         | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | GraphSAINTEdgeSampler         | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | GraphSAINTRandomWalkSampler   | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | ShaDowKHopSampler             | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | RandomNodeLoader              | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | ZipLoader                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| loader     | DataListLoader                | 否           |                 | :heavy_check_mark:            |                |                |              |
| loader     | DenseDataLoader               | 否           |                 |  :heavy_check_mark:            |                |                |              |
| loader     | TemporalDataLoader            | 否           |                 | :heavy_check_mark:             |                |                |              |


### `transforms` 模块

| **模块**      | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|---------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| transforms    | BaseTransform                 | 否           |                 |              |                |                |              |
| transforms    | Compose                       | 否           |                 |              |                |                |              |
| transforms    | ComposeFilters                | 否           |                 |              |                |                |              |
| transforms    | ToDevice                      | 否           |                 |              |                |                |              |
| transforms    | ToSparseTensor                | 否           |                 |              |                |                |              |
| transforms    | Constant                      | 否           |                 |              |                |                |              |
| transforms    | NormalizeFeatures             | 否           |                 |              |                |                |              |
| transforms    | SVDFeatureReduction           | 否           |                 |              |                |                |              |
| transforms    | RemoveTrainingClasses         | 否           |                 |              |                |                |              |
| transforms    | RandomNodeSplit               | 否           |                 |              |                |                |              |
| transforms    | RandomLinkSplit               | 否           |                 |              |                |                |              |
| transforms    | NodePropertySplit             | 否           |                 |              |                |                |              |
| transforms    | IndexToMask                   | 否           |                 |              |                |                |              |
| transforms    | MaskToIndex                   | 否           |                 |              |                |                |              |
| transforms    | Pad                           | 否           |                 |              |                |                |              |
| transforms    | ToUndirected                  | 否           |                 |              |                |                |              |
| transforms    | OneHotDegree                  | 否           |                 |              |                |                |              |
| transforms    | TargetIndegree                | 否           |                 |              |                |                |              |
| transforms    | LocalDegreeProfile            | 否           |                 |              |                |                |              |
| transforms    | AddSelfLoops                  | 否           |                 |              |                |                |              |
| transforms    | AddRemainingSelfLoops         | 否           |                 |              |                |                |              |
| transforms    | RemoveIsolatedNodes           | 否           |                 |              |                |                |              |
| transforms    | RemoveDuplicatedEdges         | 否           |                 |              |                |                |              |
| transforms    | KNNGraph                      | 否           |                 |              |                |                |              |
| transforms    | RadiusGraph                   | 否           |                 |              |                |                |              |
| transforms    | ToDense                       | 否           |                 |              |                |                |              |
| transforms    | TwoHop                        | 否           |                 |              |                |                |              |
| transforms    | LineGraph                     | 否           |                 |              |                |                |              |
| transforms    | LaplacianLambdaMax            | 否           |                 |              |                |                |              |
| transforms    | GDC                           | 否           |                 |              |                |                |              |
| transforms    | SIGN                          | 否           |                 |              |                |                |              |
| transforms    | GCNNorm                       | 否           |                 |              |                |                |              |
| transforms    | AddMetaPaths                  | 否           |                 |              |                |                |              |
| transforms    | AddRandomMetaPaths            | 否           |                 |              |                |                |              |
| transforms    | RootedEgoNets                 | 否           |                 |              |                |                |              |
| transforms    | RootedRWSubgraph              | 否           |                 |              |                |                |              |
| transforms    | LargestConnectedComponents    | 否           |                 |              |                |                |              |
| transforms    | VirtualNode                   | 否           |                 |              |                |                |              |
| transforms    | AddLaplacianEigenvectorPE     | 否           |                 |              |                |                |              |
| transforms    | AddRandomWalkPE               | 否           |                 |              |                |                |              |
| transforms    | FeaturePropagation            | 否           |                 |              |                |                |              |
| transforms    | HalfHop                       | 否           |                 |              |                |                |              |
| transforms    | Distance                      | 否           |                 |              |                |                |              |
| transforms    | Cartesian                     | 否           |                 |              |                |                |              |
| transforms    | LocalCartesian                | 否           |                 |              |                |                |              |
| transforms    | Polar                         | 否           |                 |              |                |                |              |
| transforms    | Spherical                     | 否           |                 |              |                |                |              |
| transforms    | PointPairFeatures             | 否           |                 |              |                |                |              |
| transforms    | Center                        | 否           |                 |              |                |                |              |
| transforms    | NormalizeRotation             | 否           |                 |              |                |                |              |
| transforms    | NormalizeScale                | 否           |                 |              |                |                |              |
| transforms    | RandomJitter                  | 否           |                 |              |                |                |              |
| transforms    | RandomFlip                    | 否           |                 |              |                |                |              |
| transforms    | LinearTransformation          | 否           |                 |              |                |                |              |
| transforms    | RandomScale                   | 否           |                 |              |                |                |              |
| transforms    | RandomRotate                  | 否           |                 |              |                |                |              |
| transforms    | RandomShear                   | 否           |                 |              |                |                |              |
| transforms    | FaceToEdge                    | 否           |                 |              |                |                |              |
| transforms    | SamplePoints                  | 否           |                 |              |                |                |              |
| transforms    | FixedPoints                   | 否           |                 |              |                |                |              |
| transforms    | GenerateMeshNormals           | 否           |                 |              |                |                |              |
| transforms    | Delaunay                      | 否           |                 |              |                |                |              |
| transforms    | ToSLIC                        | 否           |                 |              |                |                |              |
| transforms    | GridSampling                  | 否           |                 |              |                |                |              |


### `datasets` 模块

| **模块**      | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|---------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| datasets      | KarateClub                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | TUDataset                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | GNNBenchmarkDataset           | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | Planetoid                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | NELL                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | CitationFull                  | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | CoraFull                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Coauthor                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Amazon                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | PPI                           | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Reddit                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Reddit2                       | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Flickr                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Yelp                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | AmazonProducts                | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | QM7b                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | QM9                           | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MD17                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | ZINC                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | AQSOL                         | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | MoleculeNet                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | PCQM4Mv2                      | 否           |                 |  :heavy_check_mark:            |                |                |              |
| datasets      | Entities                      | 否           |                 | :heavy_check_mark:            |                |                |              |
| datasets      | RelLinkPredDataset            | 否           |                 |:heavy_check_mark:             |                |                |              |
| datasets      | GEDDataset                    | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | AttributedGraphDataset        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MNISTSuperpixels              | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | FAUST                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | DynamicFAUST                  | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | ShapeNet                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | ModelNet                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | CoMA                          | 否           |                 |  :heavy_check_mark:            |                |                |              |
| datasets      | SHREC2016                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | TOSCA                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | PCPNetDataset                 | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | S3DIS                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | GeometricShapes               | 否           |                 |  :heavy_check_mark:            |                |                |              |
| datasets      | BitcoinOTC                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | GDELTLite                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | ICEWS18                       | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | GDELT                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | WILLOWObjectClass             | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | PascalVOCKeypoints            | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | PascalPF                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | SNAPDataset                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | SuiteSparseMatrixCollection   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | WordNet18                     | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | WordNet18RR                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | FB15k_237                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | WikiCS                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | WebKB                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | WikipediaNetwork              | 否           |                 |:heavy_check_mark:             |                |                |              |
| datasets      | HeterophilousGraphDataset     | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | Actor                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | UPFD                          | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | GitHub                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | FacebookPagePage              | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | LastFMAsia                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | DeezerEurope                  | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | GemsecDeezer                  | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | Twitch                        | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | Airports                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | LRGBDataset                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MalNetTiny                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | OMDB                          | 否           |                 | :heavy_check_mark:            |                |                |              |
| datasets      | PolBlogs                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | EmailEUCore                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | LINKXDataset                  | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | EllipticBitcoinDataset        | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | EllipticBitcoinTemporalDataset| 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | DGraphFin                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | HydroNet                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | AirfRANS                      | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | JODIEDataset                  | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Wikidata5M                    | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | MyketDataset                  | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | BrcaTcga                      | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | DBP15K                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | AMiner                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | OGB_MAG                       | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | DBLP                          | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | MovieLens                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MovieLens100K                 | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MovieLens1M                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | IMDB                          | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | LastFM                        | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | HGBDataset                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | Taobao                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | IGMCDataset                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | AmazonBook                    | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | HM                            | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | OSE_GVCS                      | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | RCDD                          | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | FakeDataset                   | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | FakeHeteroDataset             | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | StochasticBlockModelDataset   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | RandomPartitionGraphDataset   | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | MixHopSyntheticDataset        | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | ExplainerDataset              | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | InfectionDataset              | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | BA2MotifDataset               | 否           |                 | :heavy_check_mark:             |                |                |              |
| datasets      | BAMultiShapesDataset          | 否           |                 |:heavy_check_mark:              |                |                |              |
| datasets      | BAShapes                      | 否           |                 | :heavy_check_mark:             |                |                |              |

### `explain` 模块

| **模块**    | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|-------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| explain     | ExplainerConfig           | 否           |                 | :heavy_check_mark:             |                |                |              |
| explain     | ModelConfig               | 否           |                 |  :heavy_check_mark:            |                |                |              |
| explain     | ThresholdConfig           | 否           |                 | :heavy_check_mark:             |                |                |              |
| explain     | Explanation               | 否           |                 |  :heavy_check_mark:            |                |                |              |
| explain     | HeteroExplanation         | 否           |                 |  :heavy_check_mark:            |                |                |              |
| explain     | Explainer                 | 否           |                 | :heavy_check_mark:             |                |                |              |


### `profile` 模块

| **模块**    | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|-------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| profile     | profileit                     | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | timeit                        | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | get_stats_summary             | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | trace_handler                 | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | print_time_total              | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | rename_profile_file           | 否           |                 |  :heavy_check_mark:            |                |                |              |
| profile     | torch_profile                 | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | xpu_profile                   | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | count_parameters              | 否           |                 | :heavy_check_mark:            |                |                |              |
| profile     | get_model_size                | 否           |                 | :heavy_check_mark:            |                |                |              |
| profile     | get_data_size                 | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | get_cpu_memory_from_gc        | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | get_gpu_memory_from_gc        | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | get_gpu_memory_from_nvidia_smi | 否          |                 |:heavy_check_mark:             |                |                |              |
| profile     | get_gpu_memory_from_ipex      | 否           |                 | :heavy_check_mark:             |                |                |              |
| profile     | benchmark                     | 否           |                 | :heavy_check_mark:             |                |                |              |


### `nn.aggr` 模块

| **模块**    | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|-------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.aggr     | Aggregation                   | 否           |                 |              |                |                |              |
| nn.aggr     | MultiAggregation              | 否           |                 |              |                |                |              |
| nn.aggr     | SumAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | MeanAggregation               | 否           |                 |              |                |                |              |
| nn.aggr     | MaxAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | MinAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | MulAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | VarAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | StdAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | SoftmaxAggregation            | 否           |                 |              |                |                |              |
| nn.aggr     | PowerMeanAggregation          | 否           |                 |              |                |                |              |
| nn.aggr     | MedianAggregation             | 否           |                 |              |                |                |              |
| nn.aggr     | QuantileAggregation           | 否           |                 |              |                |                |              |
| nn.aggr     | LSTMAggregation               | 否           |                 |              |                |                |              |
| nn.aggr     | GRUAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | Set2Set                       | 否           |                 |              |                |                |              |
| nn.aggr     | DegreeScalerAggregation       | 否           |                 |              |                |                |              |
| nn.aggr     | SortAggregation               | 否           |                 |              |                |                |              |
| nn.aggr     | GraphMultisetTransformer      | 否           |                 |              |                |                |              |
| nn.aggr     | AttentionalAggregation        | 否           |                 |              |                |                |              |
| nn.aggr     | EquilibriumAggregation        | 否           |                 |              |                |                |              |
| nn.aggr     | MLPAggregation                | 否           |                 |              |                |                |              |
| nn.aggr     | DeepSetsAggregation           | 否           |                 |              |                |                |              |
| nn.aggr     | SetTransformerAggregation     | 否           |                 |              |                |                |              |
| nn.aggr     | LCMAggregation                | 否           |                 |              |                |                |              |


### `nn.conv` 模块

| **模块**    | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|-------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.conv     | MessagePassing               | 否           |                 |              |                |                |              |
| nn.conv     | SimpleConv                   | 是           |                 |              |                |                |              |
| nn.conv     | GCNConv                      | 是           |                 |              |                |                |              |
| nn.conv     | ChebConv                     | 是           |                 |              |                |                |              |
| nn.conv     | SAGEConv                     | 是           |                 |              |                |                |              |
| nn.conv     | CuGraphSAGEConv              | 是           |                 |              |                |                |              |
| nn.conv     | GraphConv                    | 是           |                 |              |                |                |              |
| nn.conv     | GravNetConv                  | 是           |                 |              |                |                |              |
| nn.conv     | GatedGraphConv               | 是           |                 |              |                |                |              |
| nn.conv     | ResGatedGraphConv            | 是           |                 |              |                |                |              |
| nn.conv     | GATConv                      | 是           |                 |              |                |                |              |
| nn.conv     | CuGraphGATConv               | 是           |                 |              |                |                |              |
| nn.conv     | FusedGATConv                 | 是           |                 |              |                |                |              |
| nn.conv     | GATv2Conv                    | 是           |                 |              |                |                |              |
| nn.conv     | TransformerConv              | 是           |                 |              |                |                |              |
| nn.conv     | AGNNConv                     | 是           |                 |              |                |                |              |
| nn.conv     | TAGConv                      | 是           |                 |              |                |                |              |
| nn.conv     | GINConv                      | 是           |                 |              |                |                |              |
| nn.conv     | GINEConv                     | 是           |                 |              |                |                |              |
| nn.conv     | ARMAConv                     | 是           |                 |              |                |                |              |
| nn.conv     | SGConv                       | 是           |                 |              |                |                |              |
| nn.conv     | APPNP                        | 是           |                 |              |                |                |              |
| nn.conv     | MFConv                       | 是           |                 |              |                |                |              |
| nn.conv     | RGCNConv                     | 是           |                 |              |                |                |              |
| nn.conv     | FastRGCNConv                 | 是           |                 |              |                |                |              |
| nn.conv     | CuGraphRGCNConv              | 是           |                 |              |                |                |              |
| nn.conv     | RGATConv                     | 是           |                 |              |                |                |              |
| nn.conv     | SignedConv                   | 是           |                 |              |                |                |              |
| nn.conv     | DNAConv                      | 是           |                 |              |                |                |              |
| nn.conv     | PointNetConv                 | 是           |                 |              |                |                |              |
| nn.conv     | GMMConv                      | 是           |                 |              |                |                |              |
| nn.conv     | SplineConv                   | 是           |                 |              |                |                |              |
| nn.conv     | NNConv                       | 是           |                 |              |                |                |              |
| nn.conv     | CGConv                       | 是           |                 |              |                |                |              |
| nn.conv     | EdgeConv                     | 是           |                 |              |                |                |              |
| nn.conv     | DynamicEdgeConv              | 是           |                 |              |                |                |              |
| nn.conv     | XConv                        | 是           |                 |              |                |                |              |
| nn.conv     | PPFConv                      | 是           |                 |              |                |                |              |
| nn.conv     | FeaStConv                    | 是           |                 |              |                |                |              |
| nn.conv     | PointTransformerConv         | 是           |                 |              |                |                |              |
| nn.conv     | HypergraphConv               | 是           |                 |              |                |                |              |
| nn.conv     | LEConv                       | 是           |                 |              |                |                |              |
| nn.conv     | PNAConv                      | 是           |                 |              |                |                |              |
| nn.conv     | ClusterGCNConv               | 是           |                 |              |                |                |              |
| nn.conv     | GENConv                      | 是           |                 |              |                |                |              |
| nn.conv     | GCN2Conv                     | 是           |                 |              |                |                |              |
| nn.conv     | PANConv                      | 是           |                 |              |                |                |              |
| nn.conv     | WLConv                       | 是           |                 |              |                |                |              |
| nn.conv     | WLConvContinuous             | 是           |                 |              |                |                |              |
| nn.conv     | FiLMConv                     | 是           |                 |              |                |                |              |
| nn.conv     | SuperGATConv                 | 是           |                 |              |                |                |              |
| nn.conv     | FAConv                       | 是           |                 |              |                |                |              |
| nn.conv     | EGConv                       | 是           |                 |              |                |                |              |
| nn.conv     | PDNConv                      | 是           |                 |              |                |                |              |
| nn.conv     | GeneralConv                  | 是           |                 |              |                |                |              |
| nn.conv     | HGTConv                      | 是           |                 |              |                |                |              |
| nn.conv     | HEATConv                     | 是           |                 |              |                |                |              |
| nn.conv     | HeteroConv                   | 是           |                 |              |                |                |              |
| nn.conv     | HANConv                      | 是           |                 |              |                |                |              |
| nn.conv     | LGConv                       | 是           |                 |              |                |                |              |
| nn.conv     | SSGConv                      | 是           |                 |              |                |                |              |
| nn.conv     | PointGNNConv                 | 是           |                 |              |                |                |              |
| nn.conv     | GPSConv                      | 是           |                 |              |                |                |              |
| nn.conv     | AntiSymmetricConv            | 是           |                 |              |                |                |              |
| nn.conv     | DirGNNConv                   | 是           |                 |              |                |                |              |
| nn.conv     | MixHopConv                   | 是           |                 |              |                |                |              |


### `nn.pool` 模块

| **模块**   | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.pool    | global_add_pool           | 否           |                 |              |                |                |              |
| nn.pool    | global_mean_pool          | 否           |                 |              |                |                |              |
| nn.pool    | global_max_pool           | 否           |                 |              |                |                |              |
| nn.pool    | KNNIndex                  | 否           |                 |              |                |                |              |
| nn.pool    | L2KNNIndex                | 否           |                 |              |                |                |              |
| nn.pool    | MIPSKNNIndex              | 否           |                 |              |                |                |              |
| nn.pool    | TopKPooling               | 是           |                 |              |                |                |              |
| nn.pool    | SAGPooling                | 是           |                 |              |                |                |              |
| nn.pool    | EdgePooling               | 是           |                 |              |                |                |              |
| nn.pool    | ASAPooling                | 是           |                 |              |                |                |              |
| nn.pool    | PANPooling                | 是           |                 |              |                |                |              |
| nn.pool    | MemPooling                | 是           |                 |              |                |                |              |
| nn.pool    | max_pool                  | 否           |                 |              |                |                |              |
| nn.pool    | avg_pool                  | 否           |                 |              |                |                |              |
| nn.pool    | max_pool_x                | 否           |                 |              |                |                |              |
| nn.pool    | max_pool_neighbor_x       | 否           |                 |              |                |                |              |
| nn.pool    | avg_pool_x                | 否           |                 |              |                |                |              |
| nn.pool    | avg_pool_neighbor_x       | 否           |                 |              |                |                |              |
| nn.pool    | graclus                   | 否           |                 |              |                |                |              |
| nn.pool    | voxel_grid                | 否           |                 |              |                |                |              |
| nn.pool    | fps                       | 否           |                 |              |                |                |              |
| nn.pool    | knn                       | 否           |                 |              |                |                |              |
| nn.pool    | knn_graph                 | 否           |                 |              |                |                |              |
| nn.pool    | approx_knn                | 否           |                 |              |                |                |              |
| nn.pool    | approx_knn_graph          | 否           |                 |              |                |                |              |
| nn.pool    | radius                    | 否           |                 |              |                |                |              |
| nn.pool    | radius_graph              | 否           |                 |              |                |                |              |
| nn.pool    | nearest                   | 否           |                 |              |                |                |              |


### `nn.glob` 模块

| **模块**   | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.glob    | GlobalAttention           | 是           |                 |              |                |                |              |
| nn.glob    | global_sort_pool          | 是           |                 |              |                |                |              |
| nn.glob    | global_add_pool           | 否           |                 |              |                |                |              |
| nn.glob    | global_max_pool           | 否           |                 |              |                |                |              |
| nn.glob    | global_mean_pool          | 否           |                 |              |                |                |              |


### `nn.norm` 模块

| **模块**   | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.norm    | BatchNorm                 | 是           |                 |              |                |                |              |
| nn.norm    | HeteroBatchNorm           | 是           |                 |              |                |                |              |
| nn.norm    | InstanceNorm              | 是           |                 |              |                |                |              |
| nn.norm    | LayerNorm                 | 是           |                 |              |                |                |              |
| nn.norm    | HeteroLayerNorm           | 是           |                 |              |                |                |              |
| nn.norm    | GraphNorm                 | 是           |                 |              |                |                |              |
| nn.norm    | GraphSizeNorm             | 否           |                 |              |                |                |              |
| nn.norm    | PairNorm                  | 否           |                 |              |                |                |              |
| nn.norm    | MeanSubtractionNorm       | 否           |                 |              |                |                |              |
| nn.norm    | MessageNorm               | 是           |                 |              |                |                |              |
| nn.norm    | DiffGroupNorm             | 否           |                 |              |                |                |              |


### `nn.dense` 模块

| **模块**   | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.dense   | Linear                    | 是           |                 |              |                |                |              |
| nn.dense   | HeteroLinear              | 是           |                 |              |                |                |              |
| nn.dense   | HeteroDictLinear          | 是           |                 |              |                |                |              |
| nn.dense   | DenseGCNConv              | 是           |                 |              |                |                |              |
| nn.dense   | DenseGINConv              | 是           |                 |              |                |                |              |
| nn.dense   | DenseGraphConv            | 是           |                 |              |                |                |              |
| nn.dense   | DenseSAGEConv             | 是           |                 |              |                |                |              |
| nn.dense   | DenseGATConv              | 是           |                 |              |                |                |              |
| nn.dense   | dense_diff_pool           | 是           |                 |              |                |                |              |
| nn.dense   | dense_mincut_pool         | 是           |                 |              |                |                |              |
| nn.dense   | DMoNPooling               | 是           |                 |              |                |                |              |


### `nn.kge` 模块

| **模块**   | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.kge     | KGEModel                  | 是           |                 |              |                |                |              |
| nn.kge     | TransE                    | 是           |                 |              |                |                |              |
| nn.kge     | ComplEx                   | 是           |                 |              |                |                |              |
| nn.kge     | DistMult                  | 是           |                 |              |                |                |              |
| nn.kge     | RotatE                    | 是           |                 |              |                |                |              |


### `nn.models` 模块

| **模块**     | **公开 API**              | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|--------------|---------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.models    | MLP                       | 是           |                 |              |                |                |              |
| nn.models    | GCN                       | 是           |                 |              |                |                |              |
| nn.models    | GraphSAGE                 | 是           |                 |              |                |                |              |
| nn.models    | GIN                       | 是           |                 |              |                |                |              |
| nn.models    | GAT                       | 是           |                 |              |                |                |              |
| nn.models    | PNA                       | 是           |                 |              |                |                |              |
| nn.models    | EdgeCNN                   | 是           |                 |              |                |                |              |
| nn.models    | JumpingKnowledge          | 是           |                 |              |                |                |              |
| nn.models    | MetaLayer                 | 是           |                 |              |                |                |              |
| nn.models    | Node2Vec                  | 是           |                 |              |                |                |              |
| nn.models    | DeepGraphInfomax          | 是           |                 |              |                |                |              |
| nn.models    | InnerProductDecoder       | 是           |                 |              |                |                |              |
| nn.models    | GAE                       | 是           |                 |              |                |                |              |
| nn.models    | VGAE                      | 是           |                 |              |                |                |              |
| nn.models    | ARGA                      | 是           |                 |              |                |                |              |
| nn.models    | ARGVA                     | 是           |                 |              |                |                |              |
| nn.models    | SignedGCN                 | 是           |                 |              |                |                |              |
| nn.models    | RENet                     | 是           |                 |              |                |                |              |
| nn.models    | GraphUNet                 | 是           |                 |              |                |                |              |
| nn.models    | SchNet                    | 是           |                 |              |                |                |              |
| nn.models    | DimeNet                   | 是           |                 |              |                |                |              |
| nn.models    | DimeNetPlusPlus           | 是           |                 |              |                |                |              |
| nn.models    | to_captum_model           | 否           |                 |              |                |                |              |
| nn.models    | to_captum_input           | 否           |                 |              |                |                |              |
| nn.models    | captum_output_to_dicts    | 否           |                 |              |                |                |              |
| nn.models    | MetaPath2Vec              | 是           |                 |              |                |                |              |
| nn.models    | DeepGCNLayer              | 是           |                 |              |                |                |              |
| nn.models    | TGNMemory                 | 是           |                 |              |                |                |              |
| nn.models    | LabelPropagation          | 是           |                 |              |                |                |              |
| nn.models    | CorrectAndSmooth          | 是           |                 |              |                |                |              |
| nn.models    | AttentiveFP               | 是           |                 |              |                |                |              |
| nn.models    | RECT_L                    | 是           |                 |              |                |                |              |
| nn.models    | LINKX                     | 是           |                 |              |                |                |              |
| nn.models    | LightGCN                  | 是           |                 |              |                |                |              |
| nn.models    | MaskLabel                 | 是           |                 |              |                |                |              |
| nn.models    | GroupAddRev               | 是           |                 |              |                |                |              |
| nn.models    | GNNFF                     | 是           |                 |              |                |                |              |
| nn.models    | PMLP                      | 是           |                 |              |                |                |              |
| nn.models    | NeuralFingerprint         | 是           |                 |              |                |                |              |
| nn.models    | ViSNet                    | 是           |                 |              |                |                |              |

### `nn.functional` 模块

| **模块**         | **公开 API**            | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|-------------------|-------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn.functional     | bro                    | 是           |                 |              |                |                |              |
| nn.functional     | gini                   | 是           |                 |              |                |                |              |


### `nn` 模块

| **模块**   | **公开 API**            | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|------------|-------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| nn         | Reshape                 | 否           |                 |              |                |                |              |
| nn         | Sequential              | 否           |                 |              |                |                |              |
| nn         | DataParallel            | 否           |                 |              |                |                |              |
| nn         | to_hetero               | 否           |                 |              |                |                |              |
| nn         | to_hetero_with_bases    | 否           |                 |              |                |                |              |
| nn         | to_fixed_size           | 否           |                 |              |                |                |              |
| nn         | PositionalEncoding      | 否           |                 |              |                |                |              |
| nn         | TemporalEncoding        | 否           |                 |              |                |                |              |
| nn         | summary                 | 否           |                 |              |                |                |              |


### `torch_geometric` 模块

| **模块**           | **公开 API**                  | **需要反向** | **转换代码API** | **转码状态** | **单测码文件** | **单测码状态** | **最终状态** |
|--------------------|-------------------------------|--------------|-----------------|--------------|----------------|----------------|--------------|
| torch_geometric    | EdgeIndex                     | 否           |                 |              |                |                |              |
| torch_geometric    | seed_everything               | 否           |                 |              |                |                |              |
| torch_geometric    | get_home_dir                  | 否           |                 |              |                |                |              |
| torch_geometric    | set_home_dir                  | 否           |                 |              |                |                |              |
| torch_geometric    | compile                       | 否           |                 |              |                |                |              |
| torch_geometric    | is_compiling                  | 否           |                 |              |                |                |              |
| torch_geometric    | is_torch_instance             | 否           |                 |              |                |                |              |
| torch_geometric    | is_debug_enabled              | 否           |                 |              |                |                |              |
| torch_geometric    | debug                         | 否           |                 |              |                |                |              |
| torch_geometric    | set_debug                     | 否           |                 |              |                |                |              |
| torch_geometric    | is_experimental_mode_enabled  | 否           |                 |              |                |                |              |
| torch_geometric    | experimental_mode             | 否           |                 |              |                |                |              |
| torch_geometric    | set_experimental_mode         | 否           |                 |              |                |                |              |
| torch_geometric    | torch_geometric               | 否           |                 |              |                |                |              |
| torch_geometric    | version                       | 否           |                 |              |                |                |              |
| torch_geometric    | _compile                      | 否           |                 |:heavy_check_mark:              |                |                |              |
| torch_geometric    | _onnx                         | 否           |                 | :heavy_check_mark:             |                |                |              |
| torch_geometric    | backend                       | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | config_mixin                  | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | config_store                  | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | debug                         | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | deprecation                   | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | device                        | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | edge_index                    | 否           |                 |              |                |                |              |
| torch_geometric    | experimental                  | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | home                          | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | index                         | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | inspector                     | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | isinstance                    | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | lazy_loader                   | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | logging                       | 否           |                |:heavy_check_mark:              |                |                |              |
| torch_geometric    | resolver                      | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | seed                          | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | template                      | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | typing                        | 否           |                | :heavy_check_mark:             |                |                |              |
| torch_geometric    | warnings                      | 否           |               | :heavy_check_mark:             |                |                |              |


## 4. 公共 API 单元测试

待定

## 5. 下一步计划

根据调研结果，下一步重点完成以下任务：

1. **实现核心 API**：优先完成对 `nn`、`data` 和 `datasets` 模块的核心 API 迁移。
2. **补充单测**：为所有迁移的 API 提供完备的单元测试。
3. **性能优化**：优化关键算子的性能，确保与 Torch-Geometric 实现的性能一致。
4. **文档编写**：补充相关 API 的中文文档和使用示例。

---

## 6. 影响面与风险评估

- **影响面**：适配不会破坏现有 Paddle 生态，仅对 Torch-Geometric 提供后端支持。
- **风险**：重点在于图神经网络相关算子的迁移，其实现复杂度较高。

---
