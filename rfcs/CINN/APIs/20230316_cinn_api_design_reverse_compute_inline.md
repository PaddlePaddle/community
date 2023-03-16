# CINN ReverseComputeInline 设计文档
|API名称 | ReverseComputeInline | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | zrr1999 |
|提交时间<input type="checkbox" class="rowselector hidden"> | 2023-03-16 |
|版本号 | V1.0 | 
|依赖CINN版本<input type="checkbox" class="rowselector hidden"> | develop | 
|文件名 | 20230226_cinn_api_design_reverse_compute_inline.md<br> |


# 一、概述

## 1、相关背景
CINN是一种在不改变模型代码的条件下加速飞桨模型运行速度的深度学习编译器。
在对接上层框架时，编译器会将上层的框架算子进一步拆分为若干基础算子，这样做的目的一方面是为了减少算子开发的工作量，
仅实现有限的基础算子便可以组合出大量的上层框架算子；
另一方面便于算子融合技术在编译器中可以实现跨算子自动融合，减少最终执行时的kernel数目和访存开销，达到更好的性能。

Schedule 原语是 CINN 编译器优化算子计算实现的接口，目前已经实现了Split、Fuse、Reorder等常用原语，
其中 ComputeInline 原语操作是将一个 tensor 的计算过程内联到其消费者中完成，简化计算过程。

## 2、名词解释
NCHW ：一种图的数据格式。N 指 Batch，C 指 Channel，H 指 Height，W 指 width。

## 3、功能目标
参考已有的 ComputeInline 操作和 CINN 调度原语开发说明文档，添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。

## 4、意义
添加 ReverseComputeInline 原语，实现将一个 tensor 的计算内联到其生产者中。

# 二、CINN现状
CINN框架暂不支持 `ReverseComputeInline` 原语，需要实现。

# 三、业内方案调研
**TVM 的 `ReverseComputeInline` 原语**

在 TVM 中，核心代码如下：
```c++
class ReverseComputeInliner : public BaseInliner {
  class Substituter : public StmtExprMutator {
   public:
    explicit Substituter(ReverseComputeInliner* self) : self_(self) {}

   private:
    PrimExpr VisitExpr_(const VarNode* var) final {
      auto it = self_->idx_sub_.find(var);
      ICHECK(it != self_->idx_sub_.end());
      return (*it).second;
    }

    PrimExpr VisitExpr_(const BufferLoadNode* _load) final {
      BufferLoad load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(_load));
      return load->buffer.same_as(self_->inlined_buffer_) ? self_->producer_rhs_ : load;
    }

    ReverseComputeInliner* self_;
  };

 public:
  explicit ReverseComputeInliner(const Buffer& inlined_buffer, const BlockNode* producer_block,
                                 const BlockRealize& consumer_block_realize,
                                 const StmtSRef& scope_root_sref, const IRModule& mod)
      : BaseInliner(inlined_buffer, consumer_block_realize->block, scope_root_sref),
        producer_block_(producer_block),
        consumer_block_(consumer_block_realize->block.get()),
        mod_(mod) {
    // Initialize the predicates to ensure consumer block iters are in-bound
    consumer_iter_in_bound_ = Bool(true);
    for (const IterVar& iter : consumer_block_realize->block->iter_vars) {
      consumer_iter_in_bound_ =
          consumer_iter_in_bound_ &&
          (iter->var >= iter->dom->min && iter->var < iter->dom->min + iter->dom->extent);
    }
  }

  bool BodyPatternAllowInline(const BlockRealize& consumer_block_realize) {
    const Block& consumer_block = consumer_block_realize->block;

    if (!is_one(consumer_block_realize->predicate)) {
      // Failure: Predicate is the consumer block is not supported
      return false;
    }
    if (inlined_store_ == nullptr) {
      // Failure: block body is not BufferStore
      return false;
    }
    std::vector<const BufferLoadNode*> loads = ExtractBufferLoad(inlined_buffer_, inlined_store_);
    if (loads.size() == 0) {
      // Failure: no BufferLoad from the `inlined_buffer_`
      return false;
    }

    // Collect block iter domains and update the substition map
    Map<Var, Range> consumer_iter_doms;
    for (const auto& iter_var : consumer_block->iter_vars) {
      consumer_iter_doms.Set(iter_var->var, iter_var->dom);
      // Set default mapping for unit iters
      if (is_const_int(iter_var->dom->extent, 1) && is_const_int(iter_var->dom->min)) {
        idx_sub_[iter_var->var.get()] = iter_var->dom->min;
      }
    }

    for (const BufferLoadNode* load : loads) {
      if (!UpdateAndCheckIndexExprs(load->indices)) {
        return false;
      }
    }

    auto res = arith::DetectIterMap(
        /*indices=*/buffer_load_indices_,
        /*input_iters=*/consumer_iter_doms,
        /*predicate=*/true,
        /*check_level=*/arith::IterMapLevel::NoCheck,
        /*analyzer=*/&analyzer_,
        /*simplify_trivial_iterators=*/false);
    buffer_load_iter_map_ = res->indices;
    if (buffer_load_iter_map_.empty()) {
      // Failure: indices of BufferLoad are not bijective affine
      return false;
    }

    const BufferStoreNode* producer_store = producer_block_->body.as<BufferStoreNode>();
    if (producer_store == nullptr) {
      // Failure: producer block body is not BufferStore
      return false;
    }
    CreateInverseMapping(producer_store->indices);
    if (!CheckConsumerCovered()) {
      // Failure: consumer block iter domains are not covered by the producer block
      return false;
    }

    return true;
  }

 private:
  using BaseInliner::VisitExpr_;
  using BaseInliner::VisitStmt_;

  /*! \brief Generate the predicate after inlining based on the consumer predicate */
  PrimExpr BuildInlinedConsumerPredicate(const BlockRealizeNode* producer_block_realize) {
    // Bind the producer block iter domains for simplification
    Map<Var, PrimExpr> subst_map;
    for (int i = 0, n = producer_block_realize->iter_values.size(); i < n; ++i) {
      const IterVar& iter = producer_block_realize->block->iter_vars[i];
      analyzer_.Bind(iter->var, Range::FromMinExtent(iter->dom->min, iter->dom->extent));
      subst_map.Set(iter->var, producer_block_realize->iter_values[i]);
    }
    // Substitute the consumer block iters with the corresponding iters in the producer blocks
    PrimExpr predicate = Substituter(this)(consumer_iter_in_bound_);
    // Simplify the predicate using the producer block iter domains
    predicate = analyzer_.Simplify(predicate);
    // Substitute the producer block iters with the its bindings since the predicate in BlockRealize
    // should not contain the block iters
    predicate = Substitute(predicate, subst_map);
    predicate = analyzer_.Simplify(predicate);
    return predicate;
  }

  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    BlockRealize new_block_realize = Downcast<BlockRealize>(StmtMutator::VisitStmt_(op));
    if (op->block.get() == producer_block_) {
      auto new_predicate = BuildInlinedConsumerPredicate(new_block_realize.get());

      With<arith::ConstraintContext> ctx(&analyzer_, new_predicate);
      if (!analyzer_.CanProve(op->predicate)) {
        // We do not allow cases where the new predicate for the inlined block cannot
        // imply the original predicate in the producer block.
        throw ProducerHasNonTrivialPredicateError(mod_, GetRef<BlockRealize>(op), new_predicate);
      }
      new_block_realize.CopyOnWrite()->predicate = new_predicate;
    }
    return std::move(new_block_realize);
  }

  Stmt VisitStmt_(const BufferStoreNode* _store) final {
    BufferStore store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(_store));
    if (!store->buffer.same_as(inlined_buffer_)) {
      return std::move(store);
    }
    return ReplaceInlinedBuffer(std::move(store));
  }

  /*!
   * \brief Check the consumer block iter domains are covered by the producer block iter domains
   * \return Whether the consumer block iter domains are covered
   */
  bool CheckConsumerCovered() {
    Map<IterVar, arith::IntSet> producer_iter_doms;
    for (const IterVar& iter_var : producer_block_->iter_vars) {
      producer_iter_doms.Set(iter_var, arith::IntSet::FromRange(iter_var->dom));
    }
    // For each block iter in the consumer block, find the corresponding expression in the producer
    for (const IterVar& iter : consumer_block_->iter_vars) {
      if (auto it = idx_sub_.find(iter->var.get()); it != idx_sub_.end()) {
        const PrimExpr& producer_iter = it->second;
        arith::IntSet producer_iter_range = arith::EvalSet(producer_iter, producer_iter_doms);
        if (analyzer_.CanProve(producer_iter_range.min() > iter->dom->min) ||
            analyzer_.CanProve(producer_iter_range.max() <
                               iter->dom->min + iter->dom->extent - 1)) {
          return false;
        }
      } else {
        return false;
      }
    }
    return true;
  }

  /*!
   * \brief Apply the inverse of `buffer_load_iter_map_` to producer indices. Update `idx_sub_` with
   *        the result. It will be later used to transform the BufferStore indices of the producer.
   * \param producer_indices The BufferStore indices of the producer.
   */
  void CreateInverseMapping(const Array<PrimExpr> producer_indices) {
    auto inverse_iter_map = arith::InverseAffineIterMap(buffer_load_iter_map_, producer_indices);
    for (const auto& pair : inverse_iter_map) {
      idx_sub_[pair.first.get()] = pair.second;
    }
  }

  Stmt ReplaceInlinedBuffer(BufferStore producer) {
    producer_rhs_ = producer->value;
    return Substituter(this)(GetRef<BufferStore>(inlined_store_));
  }

  /*!
   * \brief Extracts expressions that loads a specific buffer
   * \param buffer The buffer to be loaded from
   * \param from The BufferStore statement to be extracted from
   * \return A list of `BufferLoad` expressions
   */
  static std::vector<const BufferLoadNode*> ExtractBufferLoad(const Buffer& buffer,
                                                              const BufferStoreNode* from) {
    struct Extractor : public ExprVisitor {
      void VisitExpr_(const BufferLoadNode* load) final {
        if (load->buffer.get() == buffer) {
          result.push_back(load);
        }
        ExprVisitor::VisitExpr_(load);
      }
      const BufferNode* buffer;
      std::vector<const BufferLoadNode*> result;
    } extractor;
    extractor.buffer = buffer.get();
    for (const PrimExpr& expr : from->indices) {
      extractor(expr);
    }
    extractor(from->value);
    return std::move(extractor.result);
  }

  /*!
   * \brief Update `buffer_load_indices_` with the given indices. If `buffer_load_indices_` is
   *        already non-empty, check it is consistent with the given indices.
   * \param indices The indices
   * \param expected_ndim The expected ndim of the access
   * \return A boolean flag indicating if the check is successful
   */
  bool UpdateAndCheckIndexExprs(const Array<PrimExpr>& indices) {
    if (buffer_load_indices_.empty()) {
      buffer_load_indices_ = indices;
    } else if (!std::equal(buffer_load_indices_.begin(), buffer_load_indices_.end(),
                           indices.begin(), indices.end(), ExprDeepEqual())) {
      // Failure: indices are not consistent in different BufferLoads
      return false;
    }
    return true;
  }

  /*! \brief The RHS value of the producer's BufferStore statement */
  PrimExpr producer_rhs_{nullptr};
  /*! \brief The indices of the consumer's BufferLoad */
  Array<PrimExpr> buffer_load_indices_;
  /*! \brief The IterMap representing the indices of the consumer's BufferLoad */
  Array<arith::IterSumExpr> buffer_load_iter_map_{nullptr};
  /*! \brief The producer block */
  const BlockNode* producer_block_{nullptr};
  /* \brief The consumer block */
  const BlockNode* consumer_block_{nullptr};
  /*! \brief The predicate to ensure the consumer block iters are in-bound. It will be inserted
   * as the predicate of the producer block after inlining.
   */
  PrimExpr consumer_iter_in_bound_{nullptr};
  /*! \brief The arithmetic analyzer */
  arith::Analyzer analyzer_;
  /*! \brief The target module, only used for error reporting. */
  const IRModule& mod_;
};
```

[ReverseComputeInline的核心代码](https://github.com/apache/tvm/blob/422ca2855a74bf0d0d88f1aa66343015f4326ac1/src/tir/schedule/primitive/compute_inline.cc)

# 四、对比分析
TVM 的 `ReverseComputeInline` 原语实现较为简单，可作为参考。本次任务计划参考已有的 ComputeInline 操作和 CINN 调度原语开发说明文档，实现 ReverseComputeInline

# 五、设计思路与实现方案

## 原语API设计
在 `cinn/ir/ir_schedule.h` 中新增 `ReverseComputeInline` 原语。
```c++
  /**
   * \brief Mark a previously inlined schedule block as no longer inlined. This function undoes the effects of
   * ComputeInline on the given schedule block.
   * @param schedule_block the previously inlined schedule block.
   */
  void ReverseComputeInline(const Expr& schedule_block);
```

## API实现方案
ComputeInline 原语：分别添加接口及实现至 cinn/ir/ir_schedule.h、cinn/ir/ir_schedule.cc
支持新增原语 Trace 重放：在 cinn/ir/schedule_desc.cc 中使用CINN_BUILD_STEP_KIND 注册 ComputeInline 原语的重放函数

# 六、测试和验收的考量。
ComputeInline 原语单测添加至 cinn/backends/ir_schedule_test.cc
新增原语 Trace 重放单测添加至 cinn/ir/schedule_desc_test.cc

# 七、可行性分析和排期规划
- 可行性分析

CINN中已经实现了许多其他原语，在现有的框架基础上能够很好地添加其他原语功能。

- 排期规划

3月17日 ~ 3月21日完成基本开发。

3月21日 ~ 3月31日完成调试和测试代码的编写。

# 八、影响面
本次任务影响模块如下，

`cinn\backends`，`cinn\ir`，`cinn\hlir`。

均是在原模块内增加代码，不影响原模块的已有功能。

# 附件及参考资料
1. [CINN项目贡献指南](https://github.com/PaddlePaddle/CINN/pull/810)  
2. [CINN IR抽象语法树](https://github.com/PaddlePaddle/CINN/pull/775)  
3. [CINN调度原语开发](https://github.com/PaddlePaddle/community/blob/master/pfcc/call-for-contributions/CINN/CINN_ir_schedule.md) 
