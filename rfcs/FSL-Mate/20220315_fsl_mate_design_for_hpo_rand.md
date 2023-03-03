# paddlefsl.hpo.rand 设计文档
|API名称 | paddlefsl.hpo.rand | 
|---|---|
|提交作者 | jinyouzhi | 
|提交时间 | 2022-03-31 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | develop | 
|文件名 | 20220316_api_design_for_hop_rand.md | 

# 一、概述
## 1、相关背景
为提升飞桨社区活跃程度，促进生态繁荣，第二期Hackathon设置的任务。（https://github.com/tata1661/FSL-Mate/issues/19）
PaddleFSL是基于飞桨框架开发的小样本学习工具包，旨在降低小样本学习实践成本，提供了一系列常用小样本学习常用算法。

## 2、功能目标
随机搜索即在搜索空间随机的搜索超参数。它是一种不需要优化问题梯度的数值优化方法，也是常用的基线超参数搜索算法。

## 3、意义。
小样本学习是为了解决碎片化场景无法获取大量数据训练模型的问题，而小样本条件下模型的经验风险最小化不可靠，有必要引入 AutoML 帮助建立模型。
超参数搜索（HPO，Hyper-Parameter Optimization）是 AutoML 的重要手段，随机搜索是一种朴素的搜索方法，是最基本的超参数搜索模式。

# 二、飞桨现状
FSL-Mate 框架中目前还不具备该能力。

# 三、业内方案调研

## NNI
NNI 是 Microsoft 推出的 AutoML 框架，基于 Python 开发，提供了比较全的功能和训练框架支持。NNI 提供了超参数调优模块 Tuner，其中随机搜索方法实现在 `RandomTuner` 类，[文档](https://nni.readthedocs.io/zh/stable/Tuner/RandomTuner.html)，[源码](https://github.com/microsoft/nni/blob/master/nni/algorithms/hpo/random_tuner.py)。

NNI 框架的使用流程，在被训练代码中与 NNI 交互，NNI 根据 `search_space.json` 的参数空间生成参数字典，传入被训练代码，被训练代码解析该参数字典。

### 实现方式
在 NNI 框架中，超参数生成模块抽象出 Tuner 的基类，不同的方法在基础上派生不同的子类。我们首先分析随机方法的具体实现，然后分析模块的实现。

#### 随机搜索
随机搜索的思想并不复杂，并且 trial 中一系列的参数组合是独立生成的，并无关联，分别根据 `spec` 指定的参数空间，根据各个参数的类型和指定的随机分布生成随机参数。采用 `numpy.random.default_rng` 随机数生成器产生参数。
```python
class RandomTuner(Tuner):

    def __init__(self, seed: int | None = None): # 初始化随机数种子
        self.space = None
        if seed is None:  # explicitly generate a seed to make the experiment reproducible
            seed = np.random.default_rng().integers(2 ** 31)
        self.rng = np.random.default_rng(seed)
        self.dedup = None
        _logger.info(f'Using random seed {seed}')

    def update_search_space(self, space): # 从config中提取参数搜索空间和设置
        self.space = format_search_space(space)
        self.dedup = Deduplicator(self.space)

    def generate_parameters(self, *args, **kwargs): # 生成一组参数
        params = suggest(self.rng, self.space)
        params = self.dedup(params)
        return deformat_parameters(params, self.space)

    def receive_trial_result(self, *args, **kwargs): # 由于独立生成参数，所以无需接收其他参数组合结果
        pass


def suggest(rng, space):
    params = {}
    for key, spec in space.items():
        if spec.is_activated_in(params):
            params[key] = suggest_parameter(rng, spec)
    return params

def suggest_parameter(rng, spec):
    if spec.categorical: # 离散型的参数
        return rng.integers(spec.size)
    if spec.normal_distributed: # 连续型的参数
        return rng.normal(spec.mu, spec.sigma)
    else:
        return rng.uniform(spec.low, spec.high)
```
#### 参数空间
NNI 从文件 `search_space.json` 导入参数空间配置，传递给 `spec`。主要有三种：
- categorical：离散型，列举若干选项，对于随机搜索，即 `randint` 序号；
- normal_distributed：连续型，高斯分布，根据均值和方差随机生成；
- uniform_distributed：连续型，均匀分布，根据上下界随机生成。

#### 超参生成模块

NNI 的超参生成模块封装为 Tuner 基类，包含一组抽象方法，提供了设置参数空间格式、生成参数、接收已有参数等方法。

```python
class Tuner(Recoverable):
    def generate_parameters(self, parameter_id: int, **kwargs) -> Parameters:
    # 生成一组超参数

    def generate_multiple_parameters(self, parameter_id_list: list[int], **kwargs) -> list[Parameters]:
    # 生成若干组超参数

    def receive_trial_result(self, parameter_id: int, parameters: Parameters, value: TrialMetric, **kwargs) -> None:
    # 对于依赖已有参数结果的方法，接收已经搜索的参数的结果

    def trial_end(self, parameter_id: int, success: bool, **kwargs) -> None:
    # 一次 trial 终止

    def update_search_space(self, search_space: SearchSpace) -> None:
    # 设置参数搜索空间，包括参数种类、范围等，NNI 建议支持在运行时更新搜索空间
```

## HyperOpt
HyperOpt 是一种基于 Python 开发的超参数搜索库，提供了支持 sklearn 的分支版本。其提供了三种超参搜索方法，该库较为简洁。
根据入门，随机搜索可以如下使用，`fmin` 实现了参数搜索功能，随机参数生成方法 `rand` 作为函数传入。
```python
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, rand, space_eval
best = fmin(objective, space, algo=rand.suggest, max_evals=100)


```
### 实现方法
在 hyperopt 目录下提供了`rand`和`tpe`的参数搜索方法。
#### 随机搜索
`rand` 中提供 `suggest` 函数，在 `fmin` 直接传入作为 `algo` 参数。
```python
def suggest(new_ids, domain, trials, seed):
    rng = np.random.default_rng(seed)
    rval = []
    for ii, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs, vals = pyll.rec_eval(
            domain.s_idxs_vals, memo={domain.s_new_ids: [new_id], domain.s_rng: rng}
        )
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id], [None], [new_result], [new_misc]))
    return rval
```
#### 参数空间的配置
HyperOpt 在 `hp` 中封装了一套参数空间配置，参数空间以 Python 代码方式配置。

```python
# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])
```

## Optuna
Optuna 是一个特别为机器学习设计的自动超参数优化软件框架，基于 Python 的实现并兼容大多数机器学习框架。该框架也提供了随机超参搜索功能，文档参见[RadomSampler](https://github.com/optuna/doc-optuna-zh-cn/blob/master/source/reference/generated/optuna.samplers.RandomSampler.rst)。

该方法的示例：
```python
import optuna
from optuna.samplers import RandomSampler


def objective(trial):
    x = trial.suggest_float("x", -5, 5)
    return x ** 2


study = optuna.create_study(sampler=RandomSampler())
study.optimize(objective, n_trials=10)
```
### 实现方法

#### 随机搜索
参考[文档](https://optuna.readthedocs.io/zh_CN/latest/reference/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler)给出的[源码](https://optuna.readthedocs.io/zh_CN/latest/_modules/optuna/samplers/_random.html#RandomSampler)，随机生成依赖于 `numpy.random`。
```python
class RandomSampler(BaseSampler):
    def __init__(self, seed: Optional[int] = None) -> None:

        self._rng = numpy.random.RandomState(seed)

    def reseed_rng(self) -> None:

        self._rng = numpy.random.RandomState()

    def infer_relative_search_space(
        self, study: Study, trial: FrozenTrial
    ) -> Dict[str, BaseDistribution]:

        return {}

    def sample_relative(
        self, study: Study, trial: FrozenTrial, search_space: Dict[str, BaseDistribution]
    ) -> Dict[str, Any]:

        return {}

    def sample_independent(
        self,
        study: Study,
        trial: FrozenTrial,
        param_name: str,
        param_distribution: distributions.BaseDistribution,
    ) -> Any:

        search_space = {param_name: param_distribution}
        trans = _SearchSpaceTransform(search_space)
        trans_params = self._rng.uniform(trans.bounds[:, 0], trans.bounds[:, 1])

        return trans.untransform(trans_params)[param_name]
```
#### 参数空间
Optuna 的参数空间配置更加 Pythonic，具体参见[文档](https://optuna.readthedocs.io/zh_CN/latest/tutorial/10_key_features/002_configurations.html#)。
在搜索时，调用 `_SearchSpaceTransform` 生成采样。
```python
def objective(trial):
    # Categorical parameter
    optimizer = trial.suggest_categorical("optimizer", ["MomentumSGD", "Adam"])

    # Integer parameter
    num_layers = trial.suggest_int("num_layers", 1, 3)

    # Integer parameter (log)
    num_channels = trial.suggest_int("num_channels", 32, 512, log=True)

    # Integer parameter (discretized)
    num_units = trial.suggest_int("num_units", 10, 100, step=5)

    # Floating point parameter
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 1.0)

    # Floating point parameter (log)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # Floating point parameter (discretized)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 1.0, step=0.1)
```

#### 超参生成模块
参考[文档](https://optuna.readthedocs.io/zh_CN/latest/reference/generated/optuna.samplers.BaseSampler.html#optuna.samplers.BaseSampler)的说明，主要提供了如下几种方法：

|method 名称| 用途| 
|---|---|
|after_trial（study, trial, state, values) | 实验后处理，从目标函数返回后调用 | 
|infer_relative_search_space(study, trial) | 推导相关搜索空间，`sample_relative()`前调用，生成所需的搜索空间 | 
|reseed_rng() | 再生成随机因子 | 
|sample_relative(study, trial, search_space)| 相关采样，从给定搜索空间采样 |
|sample_independent(study, trial, param_name, ...) | 独立无关采样，从给定分布中采样，适用于无需参数间关系的采样方法，例如随机搜索 | 

# 四、对比分析
## 评价
### NNI
作为一种全功能的 AutoML 学习框架，NNI 的功能全面，但对于当下需求显得有些笨重，就超参数搜索来说，NNI 的优点在于，功能完善，封装良好，在算法实现和功能设计值得学习。
NNI 的使用需要在被训练代码侵入增加参数交互命令，搜索空间从文件导入，使用略显复杂，不适合引入到本项目中。

### HyperOpt
HyperOpt 专注于超参搜索功能，设计简洁，开箱可用，但仅提供了三种搜索方法，且兼容的框架较少。
该框架的设计简单，没有进行模块封装，对于超参搜索方法，实现在函数`rand`中，搜索功能也封装在`fmin`中，仅仅提供了搜索最小值的功能，是一种较简化的方案。
HyperOpt 配置超参数搜索空间的方法

### Optuna
Optuna 是专用的超参搜索库，提供了全面的超参采样优化算法，兼容了大量框架。
该框架文档清晰完善，提供了非常全面的优化算法实现，封装较为良好。

## 对比分析
首先，就随机搜索算法本身，方法并不复杂，调研的框架基本都采用了 NumPy 实现随机采样功能，区别在于如何建立参数空间。
关键在于，模型的抽象封装，这一点也涉及到其他后续算法实现的开发，这方面 NNI 和 Optuna 都对参数生成模块进行了良好的封装，并且这两个框架也基于此提供了多种算法，表明其设计合理性。
最后，也要考虑到当前模块的需求并不复杂，本 API 仅应提供参数生成的方法。那么为了测试验证和使用，可以参考 HyperOpt 和 Optuna 实现一个朴素的搜索过程调用本 API 实现若干轮参数超训练实验。
综上，参考 NNI 和 Optuna 设计思路搭建模块，参考 HyperOpt 的实现提供基本的算法功能。

# 五、设计思路与实现方案

## 命名与参数设计
随机参数优化的 API 定为 `paddlefsl.hpo.rand`。

参数搜索空间假设参考 NNI 设计提供 categorical、normal、uniform 三类。
```python
class BaseSampler():
    def __init__(self):
    	pass

	def generate_parameters(search_space):
	#......

class RandomSampler(BaseSampler):
	def __init__(self, seed):

	def generate_parameters(search_space):
	
```

## 底层OP设计
不涉及。

## API实现方案

```python
class RandomSampler(BaseSampler):
	def __init__(self, seed):
		self._rng = numpy.random.RandomState(seed)

	def generate_parameters(search_space):
    	params = {}
    	for key, spec in search_space.items():
        	if spec.is_activated_in(params):
            params[key] = suggest_parameter(self._rng, spec)
    	return params

def suggest_parameter(rng, spec):
    if spec.categorical: # 离散型的参数
        return rng.integers(spec.size)
    if spec.normal_distributed: # 连续型的参数
        return rng.normal(spec.mu, spec.sigma)
    else:
        return rng.uniform(spec.low, spec.high)
```


# 六、测试和验收的考量
## 验收要求
在Omniglot和miniImageNet数据集的5-way 1-shot任务和5-way 5-shot任务进行测试。使MAML, ANIL, ProtoNet and RelationNet使用Bayesian optimization搜索出的超参数能达到比原汇报结果更高的效果。
## 测试方式
如上文所述，需要开发一个超参数搜索 examples 实际调用本 API 提供的参数搜索功能，验证实验效果。本项目中已有 MAML 等任务测试方法可供测试脚本集成调用。

# 七、可行性分析和排期规划
已有广泛实现，无需论证可行性。

1-2nd week，开发随机搜索；
3-4th week，开发单测；
5th week，完善 API 文档。

# 八、影响面
涉及到其他同类方法的引入，例如 TPE 方法。
测试涉及的 MAML 等功能，现依赖项目中已提供的脚本开发，任务93涉及到其 API 化，待后续完善后考虑兼容。

# 名词解释
无。

# 附件及参考资料
[NNI Random Search API Document](https://nni.readthedocs.io/zh/stable/Tuner/BuiltinTuner.html#Random)
[HyperOpt Document](http://hyperopt.github.io/hyperopt/#algorithms)
[Optuna Random Search API Document](https://optuna.readthedocs.io/zh_CN/latest/reference/generated/optuna.samplers.RandomSampler.html#optuna.samplers.RandomSampler)
[Random Search for Hyper-Parameter Optimization (origin paper)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)