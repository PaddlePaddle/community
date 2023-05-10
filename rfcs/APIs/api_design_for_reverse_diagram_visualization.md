# 实现飞桨框架动态图反向图的可视化 设计文档


|API名称 | 新增API名称                                         | 
|---|-------------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | 丘文波                                             | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-05-10                                      | 
|版本号 | V0.1                                            | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | 如无特殊情况，都应基于develop版本开发                          | 
|文件名 | api_design_for_reverse_diagram_visualization.md | 


# 一、概述
## 1、相关背景
飞桨深度学习框架提供了动态图编程的模式来开发深度学习模型（方便开发与调试），但动态图的反向图调试能力仍存在不足。
为飞桨动态图框架添加反向节点在 Python 端的访问机制。并在该机制基础上，为飞桨框架扩展反向图可视化能力。

pytorch框架下,有一个第三方库[pytorchviz](https://github.com/szagoruyko/pytorchviz)可以对基于pytorch框架实现的网络的反向图进行可视化
![img.png](img.png)

为了提高paddle下动态图的反向调试能力,需要对paddle下动态图的反向图进行可视化.

## 2、功能目标
- 为飞桨动态图框架扩展 Python 端访问反向节点的能力，包括不限于 grad_fn、next_functions等
- 参考 PyTorchViz 实现飞桨动态图反向图的可视化
- 丰富反向图信息：如 Tensor 名、Tensor dtype、Tensor shape、前向堆栈等

## 3、意义
为飞桨动态图框架添加反向节点在 Python 端的访问机制。并在该机制基础上，为飞桨框架扩展反向图可视化能力。方便进行调试。

# 二、飞桨现状
飞浆的前向图的可视化可以通过飞桨的可视化工具[VisualDL](https://www.paddlepaddle.org.cn/documentation/docs/zh/2.2/guides/03_VisualDL/visualdl_cn.html)实现.
但是飞浆框架目前还不支持对于反向图的可视化,也没有提供反向图的访问机制. 

飞浆构建的模型前向计算的时候会同时将[反向图的节点构建好](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/Dygraph/20221201_dygraph_backward.md#%E5%89%8D%E5%90%91%E6%89%A7%E8%A1%8C%E5%90%8E%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E5%9B%BE), 将这些节点信息暴露给pythonAPI 基于这些节点信息可以构建反向图.最后进行可视化



# 三、业内方案调研
pytorch 中的反向图可视化,可以通过第三方库[pytorchviz](https://github.com/szagoruyko/pytorchviz) 实现.
pytorchviz的实现原理是通过获取pytorch的反向图节点信息构建反向图,然后通过graphviz将反向图可视化.

pytorch张量中包含了一个属性grad_fn 用于表示反向图的节点信息. 通过next_functions函数来获取向下一个节点的信息


pytorchviz通过如下的[核心函数grad_fn](https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py#L146)来获取到pytorch构建的网络的反向图的节点信息

```python
def add_base_tensor(var, color='darkolivegreen1'):
    if var in seen:
        return
    seen.add(var)
    dot.node(str(id(var)), get_var_name(var), fillcolor=color)
    if (var.grad_fn):
        add_nodes(var.grad_fn)  # 获取张量var的反向图节点信息
        dot.edge(str(id(var.grad_fn)), str(id(var)))
    if var._is_view():
        add_base_tensor(var._base, color='darkolivegreen3')
        dot.edge(str(id(var._base)), str(id(var)), style="dotted")
```

通过[核心函数next_functions](https://github.com/szagoruyko/pytorchviz/blob/master/torchviz/dot.py#L131)来获取向下一个节点的信息
```python
def add_nodes(fn):
    assert not torch.is_tensor(fn)
    if fn in seen:
        return
    seen.add(fn)

    if hasattr(fn, 'variable'):
        # if grad_accumulator, add the node for `.variable`
        var = fn.variable
        seen.add(var)
        dot.node(str(id(var)), get_var_name(var), fillcolor='lightblue')
        dot.edge(str(id(var)), str(id(fn)))

    # add the node for this grad_fn
    dot.node(str(id(fn)), get_fn_name(fn, show_attrs, max_attr_chars))

    # recurse
    if hasattr(fn, 'next_functions'):
        for u in fn.next_functions: # 获取fn的下一个节点信息
            if u[0] is not None:
                dot.edge(str(id(u[0])), str(id(fn)))
                add_nodes(u[0])
```

# 四、对比分析
paddle和pytorch都会创建好反向图的节点信息,但是paddle没有提供反向图的访问机制, pytorch提供了反向图的访问机制.
所以可以参考pytorch的实现,在paddle中添加反向图的访问机制,
最后将反向图的节点信息通过graphviz的实现思路进行可视化.

# 五、设计思路与实现方案
基于上述的分析,要实现paddle的反向图可视化, 需要实现以下几个功能:
1. 获取paddle的反向图的节点信息
2. 将反向图的节点信息暴露给pythonAPI
3. 将反向图的节点信息通过graphviz进行可视化

## 1、获取paddle的反向图的节点信息
[前向过程执行结束后，反向节点Grad_node创建](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/Dygraph/20221201_dygraph_backward.md#%E5%89%8D%E5%90%91%E6%89%A7%E8%A1%8C%E5%90%8E%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84%E5%9B%BE) ，其中包含反向输入tensor信息 bwd_in_meta_，反向输出信息 bwd_out_meta_
GradSlotMeta中包含 adj_edge_
Edge中包含 in_slot_id ,in_rank, grad_node_等信息
![img_1.png](img_1.png)

基于以上数据结构，反向图的建立过程本质上是数据结构的实例化，[其中7个调用函数对应在数据结构上的构造关系如下图所示](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/Dygraph/20221201_dygraph_forward.md#23-dygraph-function%E4%BB%A3%E7%A0%81%E8%A7%A3%E6%9E%90)：
![img_2.png](img_2.png)
图中以C_OP节点的反向图结构建立为例。在执行C_OP代码后首先1创建C_G_OP反向节点；2,3设置其中attribute,tensorwrapper变量；4设置输出的meta信息。 5对输出的tensor设置autogradMeta信息，绑定反向节点,6设置meta中edge的下一结点，7设置输入的meta信息。

举个反向图节点创建的例子:
```c++
// Node Creation
  if(require_any_grad) {
    paddle::platform::RecordEvent node_creation_record_event("matmul node_creation", paddle::platform::TracerEventType::OperatorInner, 1);

    egr::EagerUtils::PassStopGradient(false,out_autograd_meta);

    // Node Construction
    auto grad_node = std::shared_ptr<MatmulGradNodeFinal>(new MatmulGradNodeFinal(1, 2));
    // SetAttributes if needed
    grad_node->SetAttributetranspose_x(transpose_x);
    grad_node->SetAttributetranspose_y(transpose_y);
    // Set TensorWrappers for Forward Inputs if needed
    grad_node->SetTensorWrapperx(x);
    grad_node->SetTensorWrappery(y);
    // SetGradOutMeta & SetEdges
    grad_node->SetGradOutMeta(x, 0);
    grad_node->SetGradOutMeta(y, 1);
    // SetOutRank & SetHistory & SetGradInMeta & RetainGrad
    if (out_autograd_meta) {
      egr::EagerUtils::SetOutRankWithSlot(out_autograd_meta, 0);
    }
    if (out_autograd_meta) {
      egr::EagerUtils::SetHistory(out_autograd_meta, grad_node);
    }
    grad_node->SetGradInMeta(out, 0);
    egr::EagerUtils::CheckAndRetainGrad(out);
    // Set TensorWrappers for Forward Outputs if needed

  }
```

前向过程后 可以创建好反向图中的节点, 然后将这些节点信息读取出来. 通过网络输出张良tensor 获取到对应的反向图节点,然后再基于该节点寻找下一个节点,以此类推获取到全部方向图节点.
```c++
* 遍历输出tensor的vector变量 tensors ，获取反向节点grad_node放入队列，更新node_input_buffers_dict map

const paddle::experimental::Tensor& tensor = tensors[i];
// 对每个tensor创建反向信息：auto_grad_meta

AutogradMeta* auto_grad_meta = EagerUtils::nullable_autograd_meta(tensor);
if (auto_grad_meta == nullptr) {}

// 获取输出tensor是第几个输出out_slot_id_；第几个tensor out_rank_
auto input_info = auto_grad_meta->OutRankInfo();

// 获取该tensor作为输入的反向节点
auto shared_grad_node = auto_grad_meta->GetMutableGradNode();
if (shared_grad_node == nullptr || shared_grad_node.get() == nullptr ||
    auto_grad_meta->StopGradient()) {}
    
// 获得普通grad_node的指针变量
GradNodeBase* grad_node = shared_grad_node.get();
if (is_general_grad) {
  // Save orig grad node
  orig_queue.push_back(grad_node);
  // Replace grad_node with copied grad_node
  grad_node = GeneralGrad::Instance().CopyGradNode(shared_grad_node);
  // Record potential startup grad node
  GeneralGrad::Instance().GetPotentialStartupNodes()->insert(grad_node);
}
```    

## 2. 将反向图的节点信息暴露给pythonAPI
首先在c++语言下创建一个函数 该函数的输入是一个输出张量,基于该张量获取到所有的反向图节点信息,并进行返回,
然后将该函数通过工具pybind11暴露给python

## 3.将反向图的节点信息通过graphviz进行可视化
基于上面的步骤获取到所有反向图节点信息后 参考pytorchvize的实现 将所有的节点信息组织成一个拓扑图进行显示.

获取到所有的节点信息之后使用 工具 [pygraphviz](https://pypi.org/project/pygraphviz/)将拓扑图进行可视化






## 命名与参数设计
参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)
## 底层OP设计
## API实现方案

# 六、测试和验收的考量
构建一个全连接网络,RNN网络,测试反向图的可视化效果,并保证通过全部的单元测试

# 七、可行性分析和排期规划
* 5.30 实现反向图节点信息获取
* 6.30 将反向图的节点信息暴露给pythonAPI
* 7.30 将反向图的节点信息通过graphviz进行可视化

# 八、影响面
需要进一步讨论的问题，开放性问题，有争议问题；对其他模块是否有影响

# 名词解释

# 附件及参考资料
- 飞桨开源框架：[https://github.com/PaddlePaddle/Paddle](https://github.com/PaddlePaddle/Paddle)
- 飞桨动态图阅读笔记：[https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/Dygraph](https://github.com/PaddlePaddle/community/tree/master/pfcc/paddle-code-reading/Dygraph)
- PyTorchViz： [https://github.com/szagoruyko/pytorchviz](https://github.com/szagoruyko/pytorchviz)