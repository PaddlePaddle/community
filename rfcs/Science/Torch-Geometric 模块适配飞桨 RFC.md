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


### `utils` 模块

| 模块   | 公开 API                  | 需要反向 |
|--------|---------------------------|----------|
| utils  | scatter                   | 否       |
| utils  | group_argsort             | 否       |
| utils  | segment                   | 否       |
| utils  | index_sort                | 否       |
| utils  | cumsum                    | 否       |
| utils  | degree                    | 否       |
| utils  | softmax                   | 否       |
| utils  | lexsort                   | 否       |
| utils  | sort_edge_index           | 否       |
| utils  | coalesce                  | 否       |
| utils  | is_undirected             | 否       |
| utils  | to_undirected             | 否       |
| utils  | contains_self_loops       | 否       |
| utils  | remove_self_loops         | 否       |
| utils  | segregate_self_loops      | 否       |
| utils  | add_self_loops            | 否       |
| utils  | add_remaining_self_loops  | 否       |
| utils  | get_self_loop_attr        | 否       |
| utils  | contains_isolated_nodes   | 否       |
| utils  | remove_isolated_nodes     | 否       |
| utils  | get_num_hops              | 否       |
| utils  | subgraph                  | 否       |
| utils  | bipartite_subgraph        | 否       |
| utils  | k_hop_subgraph            | 否       |
| utils  | dropout_node              | 否       |
| utils  | dropout_edge              | 否       |
| utils  | dropout_path              | 否       |
| utils  | dropout_adj               | 否       |
| utils  | homophily                 | 否       |
| utils  | assortativity             | 否       |
| utils  | get_laplacian             | 否       |
| utils  | get_mesh_laplacian        | 否       |
| utils  | mask_select               | 否       |
| utils  | index_to_mask             | 否       |

### `data` 模块

| 模块   | 公开 API                  | 需要反向 |
|--------|---------------------------|----------|
| data   | FeatureStore              | 否       |
| data   | TensorAttr                | 否       |
| data   | GraphStore                | 否       |
| data   | EdgeAttr                  | 否       |
| data   | Data                      | 否       |
| data   | HeteroData                | 否       |
| data   | Batch                     | 否       |
| data   | TemporalData              | 否       |
| data   | Database                  | 否       |
| data   | SQLiteDatabase            | 否       |
| data   | RocksDatabase             | 否       |
| data   | Dataset                   | 否       |
| data   | InMemoryDataset           | 否       |
| data   | OnDiskDataset             | 否       |
| data   | makedirs                  | 否       |
| data   | download_url              | 否       |
| data   | download_google_url       | 否       |
| data   | extract_tar               | 否       |
| data   | extract_zip               | 否       |
| data   | extract_bz2               | 否       |
| data   | extract_gz                | 否       |

### `sampler` 模块

| 模块      | 公开 API                  | 需要反向 |
|-----------|---------------------------|----------|
| sampler   | BaseSampler               | 否       |
| sampler   | NodeSamplerInput          | 否       |
| sampler   | EdgeSamplerInput          | 否       |
| sampler   | SamplerOutput             | 否       |
| sampler   | HeteroSamplerOutput       | 否       |
| sampler   | NumNeighbors              | 否       |
| sampler   | NegativeSampling          | 否       |
| sampler   | NeighborSampler           | 否       |
| sampler   | HGTSampler                | 否       |

### `loader` 模块

| 模块      | 公开 API                  | 需要反向 |
|-----------|---------------------------|----------|
| loader    | DataLoader                | 否       |
| loader    | NodeLoader                | 否       |
| loader    | LinkLoader                | 否       |
| loader    | NeighborLoader            | 否       |
| loader    | LinkNeighborLoader        | 否       |
| loader    | HGTLoader                 | 否       |
| loader    | ClusterData               | 否       |
| loader    | ClusterLoader             | 否       |
| loader    | GraphSAINTSampler         | 否       |
| loader    | GraphSAINTNodeSampler     | 否       |
| loader    | GraphSAINTEdgeSampler     | 否       |
| loader    | GraphSAINTRandomWalkSampler | 否     |
| loader    | ShaDowKHopSampler         | 否       |
| loader    | RandomNodeLoader          | 否       |
| loader    | ZipLoader                 | 否       |
| loader    | DataListLoader            | 否       |
| loader    | DenseDataLoader           | 否       |
| loader    | TemporalDataLoader        | 否       |

### `transforms` 模块

| 模块         | 公开 API                  | 需要反向 |
|--------------|---------------------------|----------|
| transforms   | BaseTransform             | 否       |
| transforms   | Compose                   | 否       |
| transforms   | ComposeFilters            | 否       |
| transforms   | ToDevice                  | 否       |
| transforms   | ToSparseTensor            | 否       |
| transforms   | Constant                  | 否       |
| transforms   | NormalizeFeatures         | 否       |
| transforms   | SVDFeatureReduction       | 否       |
| transforms   | RemoveTrainingClasses     | 否       |
| transforms   | RandomNodeSplit           | 否       |
| transforms   | RandomLinkSplit           | 否       |
| transforms   | NodePropertySplit         | 否       |
| transforms   | IndexToMask               | 否       |
| transforms   | MaskToIndex               | 否       |
| transforms   | Pad                       | 否       |
| transforms   | ToUndirected              | 否       |
| transforms   | OneHotDegree              | 否       |
| transforms   | TargetIndegree            | 否       |
| transforms   | LocalDegreeProfile        | 否       |
| transforms   | AddSelfLoops              | 否       |
| transforms   | AddRemainingSelfLoops     | 否       |
| transforms   | RemoveIsolatedNodes       | 否       |
| transforms   | RemoveDuplicatedEdges     | 否       |
| transforms   | KNNGraph                  | 否       |
| transforms   | RadiusGraph               | 否       |
| transforms   | ToDense                   | 否       |
| transforms   | TwoHop                    | 否       |
| transforms   | LineGraph                 | 否       |
| transforms   | LaplacianLambdaMax        | 否       |
| transforms   | GDC                       | 否       |
| transforms   | SIGN                      | 否       |
| transforms   | GCNNorm                   | 否       |
| transforms   | AddMetaPaths              | 否       |
| transforms   | AddRandomMetaPaths        | 否       |
| transforms   | RootedEgoNets             | 否       |
| transforms   | RootedRWSubgraph          | 否       |
| transforms   | LargestConnectedComponents | 否      |
| transforms   | VirtualNode               | 否       |
| transforms   | AddLaplacianEigenvectorPE | 否       |
| transforms   | AddRandomWalkPE           | 否       |
| transforms   | FeaturePropagation        | 否       |
| transforms   | HalfHop                   | 否       |
| transforms   | Distance                  | 否       |
| transforms   | Cartesian                 | 否       |
| transforms   | LocalCartesian            | 否       |
| transforms   | Polar                     | 否       |
| transforms   | Spherical                 | 否       |
| transforms   | PointPairFeatures         | 否       |
| transforms   | Center                    | 否       |
| transforms   | NormalizeRotation         | 否       |
| transforms   | NormalizeScale            | 否       |
| transforms   | RandomJitter              | 否       |
| transforms   | RandomFlip                | 否       |
| transforms   | LinearTransformation      | 否       |
| transforms   | RandomScale               | 否       |
| transforms   | RandomRotate              | 否       |
| transforms   | RandomShear               | 否       |
| transforms   | FaceToEdge                | 否       |
| transforms   | SamplePoints              | 否       |
| transforms   | FixedPoints               | 否       |
| transforms   | GenerateMeshNormals       | 否       |
| transforms   | Delaunay                  | 否       |
| transforms   | ToSLIC                    | 否       |
| transforms   | GridSampling              | 否       |

### `datasets` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| datasets   | KarateClub                | 否       |
| datasets   | TUDataset                 | 否       |
| datasets   | GNNBenchmarkDataset       | 否       |
| datasets   | Planetoid                 | 否       |
| datasets   | NELL                      | 否       |
| datasets   | CitationFull              | 否       |
| datasets   | CoraFull                  | 否       |
| datasets   | Coauthor                  | 否       |
| datasets   | Amazon                    | 否       |
| datasets   | PPI                       | 否       |
| datasets   | Reddit                    | 否       |
| datasets   | Reddit2                   | 否       |
| datasets   | Flickr                    | 否       |
| datasets   | Yelp                      | 否       |
| datasets   | AmazonProducts            | 否       |
| datasets   | QM7b                      | 否       |
| datasets   | QM9                       | 否       |
| datasets   | MD17                      | 否       |
| datasets   | ZINC                      | 否       |
| datasets   | AQSOL                     | 否       |
| datasets   | MoleculeNet               | 否       |
| datasets   | PCQM4Mv2                  | 否       |
| datasets   | Entities                  | 否       |
| datasets   | RelLinkPredDataset        | 否       |
| datasets   | GEDDataset                | 否       |
| datasets   | AttributedGraphDataset    | 否       |
| datasets   | MNISTSuperpixels          | 否       |
| datasets   | FAUST                     | 否       |
| datasets   | DynamicFAUST              | 否       |
| datasets   | ShapeNet                  | 否       |
| datasets   | ModelNet                  | 否       |
| datasets   | CoMA                      | 否       |
| datasets   | SHREC2016                 | 否       |
| datasets   | TOSCA                     | 否       |
| datasets   | PCPNetDataset             | 否       |
| datasets   | S3DIS                     | 否       |
| datasets   | GeometricShapes           | 否       |
| datasets   | BitcoinOTC                | 否       |
| datasets   | GDELTLite                 | 否       |
| datasets   | ICEWS18                   | 否       |
| datasets   | GDELT                     | 否       |
| datasets   | WILLOWObjectClass         | 否       |
| datasets   | PascalVOCKeypoints        | 否       |
| datasets   | PascalPF                  | 否       |
| datasets   | SNAPDataset               | 否       |
| datasets   | SuiteSparseMatrixCollection | 否      |
| datasets   | WordNet18                 | 否       |
| datasets   | WordNet18RR               | 否       |
| datasets   | FB15k_237                 | 否       |
| datasets   | WikiCS                    | 否       |
| datasets   | WebKB                     | 否       |
| datasets   | WikipediaNetwork          | 否       |
| datasets   | HeterophilousGraphDataset | 否       |
| datasets   | Actor                     | 否       |
| datasets   | UPFD                      | 否       |
| datasets   | GitHub                    | 否       |
| datasets   | FacebookPagePage          | 否       |
| datasets   | LastFMAsia                | 否       |
| datasets   | DeezerEurope              | 否       |
| datasets   | GemsecDeezer              | 否       |
| datasets   | Twitch                    | 否       |
| datasets   | Airports                  | 否       |
| datasets   | LRGBDataset               | 否       |
| datasets   | MalNetTiny                | 否       |
| datasets   | OMDB                      | 否       |
| datasets   | PolBlogs                  | 否       |
| datasets   | EmailEUCore               | 否       |
| datasets   | LINKXDataset              | 否       |
| datasets   | EllipticBitcoinDataset    | 否       |
| datasets   | EllipticBitcoinTemporalDataset | 否    |
| datasets   | DGraphFin                 | 否       |
| datasets   | HydroNet                  | 否       |
| datasets   | AirfRANS                  | 否       |
| datasets   | JODIEDataset              | 否       |
| datasets   | Wikidata5M                | 否       |
| datasets   | MyketDataset              | 否       |
| datasets   | BrcaTcga                  | 否       |
| datasets   | DBP15K                    | 否       |
| datasets   | AMiner                    | 否       |
| datasets   | OGB_MAG                   | 否       |
| datasets   | DBLP                      | 否       |
| datasets   | MovieLens                 | 否       |
| datasets   | MovieLens100K             | 否       |
| datasets   | MovieLens1M               | 否       |
| datasets   | IMDB                      | 否       |
| datasets   | LastFM                    | 否       |
| datasets   | HGBDataset                | 否       |
| datasets   | Taobao                    | 否       |
| datasets   | IGMCDataset               | 否       |
| datasets   | AmazonBook                | 否       |
| datasets   | HM                        | 否       |
| datasets   | OSE_GVCS                  | 否       |
| datasets   | RCDD                      | 否       |
| datasets   | FakeDataset               | 否       |
| datasets   | FakeHeteroDataset         | 否       |
| datasets   | StochasticBlockModelDataset | 否      |
| datasets   | RandomPartitionGraphDataset | 否      |
| datasets   | MixHopSyntheticDataset    | 否       |
| datasets   | ExplainerDataset          | 否       |
| datasets   | InfectionDataset          | 否       |
| datasets   | BA2MotifDataset           | 否       |
| datasets   | BAMultiShapesDataset      | 否       |
| datasets   | BAShapes                  | 否       |

### `explain` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| explain    | ExplainerConfig           | 否       |
| explain    | ModelConfig               | 否       |
| explain    | ThresholdConfig           | 否       |
| explain    | Explanation               | 否       |
| explain    | HeteroExplanation         | 否       |
| explain    | Explainer                 | 否       |

### `profile` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| profile    | profileit                 | 否       |
| profile    | timeit                    | 否       |
| profile    | get_stats_summary         | 否       |
| profile    | trace_handler             | 否       |
| profile    | print_time_total          | 否       |
| profile    | rename_profile_file       | 否       |
| profile    | torch_profile             | 否       |
| profile    | xpu_profile               | 否       |
| profile    | count_parameters          | 否       |
| profile    | get_model_size            | 否       |
| profile    | get_data_size             | 否       |
| profile    | get_cpu_memory_from_gc    | 否       |
| profile    | get_gpu_memory_from_gc    | 否       |
| profile    | get_gpu_memory_from_nvidia_smi | 否     |
| profile    | get_gpu_memory_from_ipex  | 否       |
| profile    | benchmark                 | 否       |

### `nn.aggr` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.aggr    | Aggregation               | 否       |
| nn.aggr    | MultiAggregation          | 否       |
| nn.aggr    | SumAggregation            | 否       |
| nn.aggr    | MeanAggregation           | 否       |
| nn.aggr    | MaxAggregation            | 否       |
| nn.aggr    | MinAggregation            | 否       |
| nn.aggr    | MulAggregation            | 否       |
| nn.aggr    | VarAggregation            | 否       |
| nn.aggr    | StdAggregation            | 否       |
| nn.aggr    | SoftmaxAggregation        | 否       |
| nn.aggr    | PowerMeanAggregation      | 否       |
| nn.aggr    | MedianAggregation         | 否       |
| nn.aggr    | QuantileAggregation       | 否       |
| nn.aggr    | LSTMAggregation           | 否       |
| nn.aggr    | GRUAggregation            | 否       |
| nn.aggr    | Set2Set                   | 否       |
| nn.aggr    | DegreeScalerAggregation   | 否       |
| nn.aggr    | SortAggregation           | 否       |
| nn.aggr    | GraphMultisetTransformer  | 否       |
| nn.aggr    | AttentionalAggregation    | 否       |
| nn.aggr    | EquilibriumAggregation    | 否       |
| nn.aggr    | MLPAggregation            | 否       |
| nn.aggr    | DeepSetsAggregation       | 否       |
| nn.aggr    | SetTransformerAggregation | 否       |
| nn.aggr    | LCMAggregation            | 否       |

### `nn.conv` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.conv    | MessagePassing            | 否       |
| nn.conv    | SimpleConv                | 是       |
| nn.conv    | GCNConv                   | 是       |
| nn.conv    | ChebConv                  | 是       |
| nn.conv    | SAGEConv                  | 是       |
| nn.conv    | CuGraphSAGEConv           | 是       |
| nn.conv    | GraphConv                 | 是       |
| nn.conv    | GravNetConv               | 是       |
| nn.conv    | GatedGraphConv            | 是       |
| nn.conv    | ResGatedGraphConv         | 是       |
| nn.conv    | GATConv                   | 是       |
| nn.conv    | CuGraphGATConv            | 是       |
| nn.conv    | FusedGATConv              | 是       |
| nn.conv    | GATv2Conv                 | 是       |
| nn.conv    | TransformerConv           | 是       |
| nn.conv    | AGNNConv                  | 是       |
| nn.conv    | TAGConv                   | 是       |
| nn.conv    | GINConv                   | 是       |
| nn.conv    | GINEConv                  | 是       |
| nn.conv    | ARMAConv                  | 是       |
| nn.conv    | SGConv                    | 是       |
| nn.conv    | APPNP                     | 是       |
| nn.conv    | MFConv                    | 是       |
| nn.conv    | RGCNConv                  | 是       |
| nn.conv    | FastRGCNConv              | 是       |
| nn.conv    | CuGraphRGCNConv           | 是       |
| nn.conv    | RGATConv                  | 是       |
| nn.conv    | SignedConv                | 是       |
| nn.conv    | DNAConv                   | 是       |
| nn.conv    | PointNetConv              | 是       |
| nn.conv    | GMMConv                   | 是       |
| nn.conv    | SplineConv                | 是       |
| nn.conv    | NNConv                    | 是       |
| nn.conv    | CGConv                    | 是       |
| nn.conv    | EdgeConv                  | 是       |
| nn.conv    | DynamicEdgeConv           | 是       |
| nn.conv    | XConv                     | 是       |
| nn.conv    | PPFConv                   | 是       |
| nn.conv    | FeaStConv                 | 是       |
| nn.conv    | PointTransformerConv      | 是       |
| nn.conv    | HypergraphConv            | 是       |
| nn.conv    | LEConv                    | 是       |
| nn.conv    | PNAConv                   | 是       |
| nn.conv    | ClusterGCNConv            | 是       |
| nn.conv    | GENConv                   | 是       |
| nn.conv    | GCN2Conv                  | 是       |
| nn.conv    | PANConv                   | 是       |
| nn.conv    | WLConv                    | 是       |
| nn.conv    | WLConvContinuous          | 是       |
| nn.conv    | FiLMConv                  | 是       |
| nn.conv    | SuperGATConv              | 是       |
| nn.conv    | FAConv                    | 是       |
| nn.conv    | EGConv                    | 是       |
| nn.conv    | PDNConv                   | 是       |
| nn.conv    | GeneralConv               | 是       |
| nn.conv    | HGTConv                   | 是       |
| nn.conv    | HEATConv                  | 是       |
| nn.conv    | HeteroConv                | 是       |
| nn.conv    | HANConv                   | 是       |
| nn.conv    | LGConv                    | 是       |
| nn.conv    | SSGConv                   | 是       |
| nn.conv    | PointGNNConv              | 是       |
| nn.conv    | GPSConv                   | 是       |
| nn.conv    | AntiSymmetricConv         | 是       |
| nn.conv    | DirGNNConv                | 是       |
| nn.conv    | MixHopConv                | 是       |

### `nn.pool` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.pool    | global_add_pool           | 否       |
| nn.pool    | global_mean_pool          | 否       |
| nn.pool    | global_max_pool           | 否       |
| nn.pool    | KNNIndex                  | 否       |
| nn.pool    | L2KNNIndex                | 否       |
| nn.pool    | MIPSKNNIndex              | 否       |
| nn.pool    | TopKPooling               | 是       |
| nn.pool    | SAGPooling                | 是       |
| nn.pool    | EdgePooling               | 是       |
| nn.pool    | ASAPooling                | 是       |
| nn.pool    | PANPooling                | 是       |
| nn.pool    | MemPooling                | 是       |
| nn.pool    | max_pool                  | 否       |
| nn.pool    | avg_pool                  | 否       |
| nn.pool    | max_pool_x                | 否       |
| nn.pool    | max_pool_neighbor_x       | 否       |
| nn.pool    | avg_pool_x                | 否       |
| nn.pool    | avg_pool_neighbor_x       | 否       |
| nn.pool    | graclus                   | 否       |
| nn.pool    | voxel_grid                | 否       |
| nn.pool    | fps                       | 否       |
| nn.pool    | knn                       | 否       |
| nn.pool    | knn_graph                 | 否       |
| nn.pool    | approx_knn                | 否       |
| nn.pool    | approx_knn_graph          | 否       |
| nn.pool    | radius                    | 否       |
| nn.pool    | radius_graph              | 否       |
| nn.pool    | nearest                   | 否       |

### `nn.glob` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.glob    | GlobalAttention           | 是       |
| nn.glob    | global_sort_pool          | 是       |
| nn.glob    | global_add_pool           | 否       |
| nn.glob    | global_max_pool           | 否       |
| nn.glob    | global_mean_pool          | 否       |

### `nn.norm` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.norm    | BatchNorm                 | 是       |
| nn.norm    | HeteroBatchNorm           | 是       |
| nn.norm    | InstanceNorm              | 是       |
| nn.norm    | LayerNorm                 | 是       |
| nn.norm    | HeteroLayerNorm           | 是       |
| nn.norm    | GraphNorm                 | 是       |
| nn.norm    | GraphSizeNorm             | 否       |
| nn.norm    | PairNorm                  | 否       |
| nn.norm    | MeanSubtractionNorm       | 否       |
| nn.norm    | MessageNorm               | 是       |
| nn.norm    | DiffGroupNorm             | 否       |

### `nn.dense` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.dense   | Linear                    | 是       |
| nn.dense   | HeteroLinear              | 是       |
| nn.dense   | HeteroDictLinear          | 是       |
| nn.dense   | DenseGCNConv              | 是       |
| nn.dense   | DenseGINConv              | 是       |
| nn.dense   | DenseGraphConv            | 是       |
| nn.dense   | DenseSAGEConv             | 是       |
| nn.dense   | DenseGATConv              | 是       |
| nn.dense   | dense_diff_pool           | 是       |
| nn.dense   | dense_mincut_pool         | 是       |
| nn.dense   | DMoNPooling               | 是       |

### `nn.kge` 模块

| 模块       | 公开 API                  | 需要反向 |
|------------|---------------------------|----------|
| nn.kge     | KGEModel                  | 是       |
| nn.kge     | TransE                    | 是       |
| nn.kge     | ComplEx                   | 是       |
| nn.kge     | DistMult                  | 是       |
| nn.kge     | RotatE                    | 是       |

### `nn.models` 模块

| 模块         | 公开 API                  | 需要反向 |
|--------------|---------------------------|----------|
| nn.models    | MLP                       | 是       |
| nn.models    | GCN                       | 是       |
| nn.models    | GraphSAGE                 | 是       |
| nn.models    | GIN                       | 是       |
| nn.models    | GAT                       | 是       |
| nn.models    | PNA                       | 是       |
| nn.models    | EdgeCNN                   | 是       |
| nn.models    | JumpingKnowledge          | 是       |
| nn.models    | MetaLayer                 | 是       |
| nn.models    | Node2Vec                  | 是       |
| nn.models    | DeepGraphInfomax          | 是       |
| nn.models    | InnerProductDecoder       | 是       |
| nn.models    | GAE                       | 是       |
| nn.models    | VGAE                      | 是       |
| nn.models    | ARGA                      | 是       |
| nn.models    | ARGVA                     | 是       |
| nn.models    | SignedGCN                 | 是       |
| nn.models    | RENet                     | 是       |
| nn.models    | GraphUNet                 | 是       |
| nn.models    | SchNet                    | 是       |
| nn.models    | DimeNet                   | 是       |
| nn.models    | DimeNetPlusPlus           | 是       |
| nn.models    | to_captum_model           | 否       |
| nn.models    | to_captum_input           | 否       |
| nn.models    | captum_output_to_dicts    | 否       |
| nn.models    | MetaPath2Vec              | 是       |
| nn.models    | DeepGCNLayer              | 是       |
| nn.models    | TGNMemory                 | 是       |
| nn.models    | LabelPropagation          | 是       |
| nn.models    | CorrectAndSmooth          | 是       |
| nn.models    | AttentiveFP               | 是       |
| nn.models    | RECT_L                    | 是       |
| nn.models    | LINKX                     | 是       |
| nn.models    | LightGCN                  | 是       |
| nn.models    | MaskLabel                 | 是       |
| nn.models    | GroupAddRev               | 是       |
| nn.models    | GNNFF                     | 是       |
| nn.models    | PMLP                      | 是       |
| nn.models    | NeuralFingerprint         | 是       |
| nn.models    | ViSNet                    | 是       |

### `nn.functional` 模块

| 模块         | 公开 API                  | 需要反向 |
|--------------|---------------------------|----------|
| nn.functional | bro                       | 是       |
| nn.functional | gini                      | 是       |

### `nn` 模块

| 模块   | 公开 API                  | 需要反向 |
|--------|---------------------------|----------|
| nn     | Reshape                   | 否       |
| nn     | Sequential                | 否       |
| nn     | DataParallel              | 否       |
| nn     | to_hetero                 | 否       |
| nn     | to_hetero_with_bases      | 否       |
| nn     | to_fixed_size             | 否       |
| nn     | PositionalEncoding        | 否       |
| nn     | TemporalEncoding          | 否       |
| nn     | summary                   | 否       |

### `torch_geometric` 模块

| 模块           | 公开 API                  | 需要反向 |
|----------------|---------------------------|----------|
| torch_geometric | EdgeIndex                 | 否       |
| torch_geometric | seed_everything           | 否       |
| torch_geometric | get_home_dir              | 否       |
| torch_geometric | set_home_dir              | 否       |
| torch_geometric | compile                   | 否       |
| torch_geometric | is_compiling              | 否       |
| torch_geometric | is_torch_instance         | 否       |
| torch_geometric | is_debug_enabled          | 否       |
| torch_geometric | debug                     | 否       |
| torch_geometric | set_debug                 | 否       |
| torch_geometric | is_experimental_mode_enabled | 否    |
| torch_geometric | experimental_mode         | 否       |
| torch_geometric | set_experimental_mode     | 否       |
| torch_geometric | torch_geometric           | 否       |
| torch_geometric | version                   | 否       |


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