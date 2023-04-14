#  paddle.linalg.lu_solve 设计文档

|API名称 | lu_solve | 
|---|---|
|提交作者<input type="checkbox" class="rowselector hidden"> | strugglejx | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-04-14 | 
|版本号 | v1.0| 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop版本 | 
|文件名 | 20220414_api_design_for_lu_solve.md<br> | 

##一、概述
###1. 相关背景

LU分解是一种常用的矩阵分解方法，可以将一个方阵分解为一个下三角矩阵和一个上三角矩阵的乘积，同时记录一个置换矩阵。LU分解可以用于求解线性方程组，即给定一个方程组Ax=b，其中A是一个方阵，b是一个向量，可以先对A进行LU分解，得到A=PLU，其中P是置换矩阵，L是下三角矩阵，U是上三角矩阵，然后利用LU分解求解方程组，即先求解Ly=Pb，再求解Ux=y。这样可以避免直接对A进行高斯消元法，减少计算量和舍入误差。

lu是paddle中实现LU分解的API。但是，paddle目前没有提供用于基于LU分解求解线性方程组的API，用户需要自己实现LU分解和回代的过程。为了提高paddle的功能完善度和用户体验，我们想要开发这个API，并且使其与paddle的其他API保持一致的风格和接口。API的调用路径为：paddle.linalg.lu_solve 和 Tensor.lu_solve。

###2. 功能目标

lu_solve的API接收三个参数：LU_data、LU_pivots和B。LU_data和LU_pivots是由lu_factor函数返回的A矩阵的LU分解结果，B是线性方程组Ax=b中的右侧向量或矩阵。lu_solve的API返回一个张量X，它是线性方程组Ax=b的解。lu_solve的API还支持批量处理多个线性方程组，并且可以指定是否求解转置或共轭转置的线性方程组。

###3. 意义

lu_solve API的意义在于提供了一种高效且稳定的求解线性方程组的方法，可以应用于各种涉及线性代数运算的深度学习任务，如最小二乘法、逆矩阵计算、线性回归等。该API也可以与现有的lu API配合使用，提高paddle框架的功能完善度和用户体验。

##二、飞桨现状

目前paddle框架不支持使用LU分解求解线性方程组的功能，只提供了lu API用于进行LU分解。如果用户想要使用LU分解求解线性方程组，需要自己编写代码实现下三角矩阵和上三角矩阵的前向和后向替换算法，并考虑置换矩阵的影响。这样不仅增加了用户的编码难度和时间成本，也可能导致代码效率低下和精度损失。

##三、业内方案调研

业内有一些深度学习框架或者科学计算库提供了使用LU分解求解线性方程组的API，例如：

- PyTorch：PyTorch提供了一个torch.linalg.lu_solve函数(1)，它接收三个参数：LU、pivots和B。LU和pivots是由torch.linalg.lu_factor函数返回的A矩阵的LU分解结果，B是线性方程组Ax=B中的右侧向量或矩阵。torch.linalg.lu_solve函数返回一个张量X，它是线性方程组Ax=b的解。torch.linalg.lu_solve函数还支持批量处理多个线性方程组，并且可以指定是否求解转置或共轭转置的线性方程组。以下代码为lu_solve函数的核心源代码，该函数针对矩阵的转置，以及上三角和下三角矩阵类型，作了相应的预处理。同时，还将矩阵转换为列主元模式，最后，按不同的数据类型调用lapack函数。

```torch.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False, out=None) → Tensor
Tensor& linalg_solve_triangular_out(
    const Tensor& A,
    const Tensor& B,
    bool upper,
    bool left,
    bool unitriangular,
    Tensor& out) {
  checkInputsSolver(A, B, left, "linalg.solve_triangular");
  Tensor A_, B_;
  std::tie(B_, A_) = _linalg_broadcast_batch_dims(B, A, /*don't check errors*/nullptr);

  const bool avoid_copy_A = A_.transpose(-2, -1).is_contiguous() && A_.is_conj();
  if (avoid_copy_A) {
    // See Note: [Cloning A]
    at::native::resize_output(out, B_.sizes());
  }
  else {
    // poorman's reimplementation of resize_output with result F-contig
    if (resize_output_check(out, B_.sizes())) {
      out.resize_(B_.transpose(-2, -1).sizes(), MemoryFormat::Contiguous);
      out.transpose_(-2, -1);  // make 'out' have Fortran contiguous memory layout
    }
  }
  // Invariant: out has the right size, so we'll be able to copy into it later on

  Tensor out_f; // the out that will go into fortran
  // We use C10_LIKELY mostly for documentation as it helps following what's the most likely path
  if C10_LIKELY (is_row_or_column_contiguous(out)) {
    out_f = out;
    if C10_LIKELY (!out.is_same(B_)) {
      out_f.copy_(B_);
    }
  } else {
    if (avoid_copy_A) {
      // See Note: [Cloning A]
      out_f = B_.clone(at::MemoryFormat::Contiguous);
    }
    else {
      out_f = cloneBatchedColumnMajor(B_);
    }
  }
  // Invariant: out_f F-ready and has B copied into it

  // out_f is F-transposed
  bool transpose_A = false;
  bool transpose_out_f = false;
  if (out_f.stride(-1) == 1) {
    left = !left;
    transpose_A = true;
    transpose_out_f = true;
    out_f.transpose_(-2 ,-1);
  }

  // No need to conjugate anything if out_f is conj as AX = conj(B) <=> conj(A)conj(X) = B
  // and X = B after the algortihm. We just anotate that A is conjugated later on
  // The solution will be written into out_f, so it'll be conjugated already

  Tensor A_f = std::move(A_);  // The A that will go into fortran

  bool A_is_conj = A_f.is_conj() != out_f.is_conj();
  bool A_is_neg = A_f.is_neg() != out_f.is_neg();
  bool A_is_f_contig = (A_f.stride(-1) == 1) == transpose_A;
  if C10_UNLIKELY (!is_row_or_column_contiguous(A_f)) {
    // We first anotate with flags on A_f all the conj / transpose / neg coming from out
    // and then we clone the resulting tensor to resolve all of them in memory
    if (out_f.is_conj()) {
      A_f = A_f.conj();
    }
    A_is_conj = false;

    if (out_f.is_neg()) {
      A_f = A_f._neg_view();
    }
    A_is_neg = false;

    // This choice is to be consistent with how we flip `upper` later on
    // Note that this is the same reasoning we apply for neg and conj below
    // If B has neg or out or transpose, then we need to resolve it in memory
    A_f = transpose_A ? A_f.clone(at::MemoryFormat::Contiguous)
                      : cloneBatchedColumnMajor(A_f);
    A_is_f_contig = true;
  } else if C10_UNLIKELY (A_is_f_contig && A_is_conj) {
    if C10_UNLIKELY (A_f.is_neg() || out_f.is_neg()) {
      // Cases A_is_neg (remember that B.is_neg() iff out_f.is_same(B))
      // -AX = -B => A(-X) = B. Swap neg of A_f. Nothing to do on X as X.is_same(B).
      // -AX = B. We resolve the neg in memory
      // AX = -B => -A -X = B. We resolve the neg in memory for A,
      //                       Since X.is_same(B), we already have that X.is_neg() == true

      // We do the neg with a view, as this will be resolved in the clone below
      if (out_f.is_neg()) {
        A_f = A_f._neg_view();
      }
      A_is_neg = false;
    }
    // We resolve the transpose if necessary and then leave A_f F-transposed,
    // as BLAS can handle the case F-transposed and conjugated
    A_f = at::clone(transpose_A ? A_f.mT() : A_f, at::MemoryFormat::Contiguous);
    A_is_f_contig = false;
    if (transpose_A) {
      upper = !upper;
    }
    // As we've already resolved the conj of A in the clone
    A_is_conj = out_f.is_conj();
  } else if C10_UNLIKELY (A_is_neg) {
    // We follow the same logic as above, only that in this case we need to perform the
    // negation in memory
    if (out_f.is_neg()) {
      A_f = -A_f;
    } else {
      A_f = A_f.resolve_neg();
    }
    A_is_neg = false;
    // As we've already resolved the conj of A in the negationa bove
    A_is_conj = out_f.is_conj();
  }
  // Invariant: out_f is F-contig and A_f is F-ready
  // neg has been resolved

  // If we pass the matrix physically F-transposed, we need to change the parity of upper
  if (A_f.stride(-1) == 1) {
    upper = !upper;
  }

  triangular_solve_stub(
    A_f.device().type(), A_f, out_f,
    /*left=*/left,
    /*upper=*/upper,
    /*transpose*/to_transpose_type(A_is_f_contig, A_is_conj),
    /*unitriangular=*/unitriangular);

  if (transpose_out_f) {
    out_f.transpose_(-2, -1);
  }

  if (!out_f.is_same(out)) {
    out.copy_(out_f);
  }
  return out;
}
```

- SciPy：SciPy提供了一个scipy.linalg.lu_solve函数(2)，它主要接收两个参数：lu_and_piv和b。lu_and_piv是一个tuple，包含LU分解矩阵lu以及置换矩阵piv，它们组成线性方程组PLUx=b中的系数矩阵P、L和U。b是线性方程组PLUx=b中的右侧向量或矩阵。numpy.linalg.solve函数返回一个数组x，它是线性方程组Ax=b的解。numpy.linalg.solve函数源代码如下所示，它并不直接使用LU分解求解线性方程组，而是使用LAPACK库中的getrs函数，它会根据a的特征选择合适的算法来求解。
```
def lu_solve(lu_and_piv, b, trans=0, overwrite_b=False, check_finite=True):
    """Solve an equation system, a x = b, given the LU factorization of a

    Parameters
    ----------
    (lu, piv)
        Factorization of the coefficient matrix a, as given by lu_factor
    b : array
        Right-hand side
    trans : {0, 1, 2}, optional
        Type of system to solve:

        =====  =========
        trans  system
        =====  =========
        0      a x   = b
        1      a^T x = b
        2      a^H x = b
        =====  =========
    overwrite_b : bool, optional
        Whether to overwrite data in b (may increase performance)
    check_finite : bool, optional
        Whether to check that the input matrices contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    x : array
        Solution to the system

    See also
    --------
    lu_factor : LU factorize a matrix

    Examples
    --------
    >>> from scipy.linalg import lu_factor, lu_solve
    >>> A = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
    >>> b = np.array([1, 1, 1, 1])
    >>> lu, piv = lu_factor(A)
    >>> x = lu_solve((lu, piv), b)
    >>> np.allclose(A @ x - b, np.zeros((4,)))
    True

    """
    (lu, piv) = lu_and_piv
    if check_finite:
        b1 = asarray_chkfinite(b)
    else:
        b1 = asarray(b)
    overwrite_b = overwrite_b or _datacopied(b1, b)
    if lu.shape[0] != b1.shape[0]:
        raise ValueError("incompatible dimensions.")

    getrs, = get_lapack_funcs(('getrs',), (lu, b1))
    x, info = getrs(lu, piv, b1, trans=trans, overwrite_b=overwrite_b)
    if info == 0:
        return x
    raise ValueError('illegal value in %dth argument of internal gesv|posv'
                     % -info)
```

##四、对比分析：

对于使用LU分解求解线性方程组的API，PyTorch和SciPy提供了比较完善和灵活的功能，它们都支持批量处理多个线性方程组。它们也都提供了对应的lu_factor函数来计算A矩阵的LU分解结果，并且可以复用这个结果来求解不同的右侧向量或矩阵。不同之处在于，Scipy支持指定方程组的类型（普通、转置或共轭转置），而PyTorch的API不支持。此外，Scipy借助于lapack中的getrs函数实现线性方程组的求解，而PyTorch则使用dtrsm、strsm等函数求解上三角或下三角系数矩阵的线性方程组，最终通过分步求得原线性方程组的解。paddle框架可以参考这些方案，设计一个既符合用户需求又具有自身特色的API。

##五、设计思路与实现方案：
###1. 命名与参数设计：

  共添加以下两个API：
  paddle.linalg.lu_solve(LU, pivots, B, *, left=True, adjoint=False) → Tensor
  - LU (Tensor) - 由L和U拼接的矩阵
  - LU_pivots (Tensor) - 由paddle.linalg.lu返回的置换矩阵
  - B (Tensor) - 线性方程组的右值
  - left (bool, 可选) - 是求解系统AX＝BAX＝B还是XA＝BXA＝B。默认值：True
  - adjoint (bool, 可选) - 是解决系统AX＝B还是A^{H}X＝B。默认值：False

###2. 底层OP设计：

借鉴Scipy的解决思路，底层OP直接调用lapack库中的getrs函数即可。

###3. API实现方案：

1. 检查参数：包括LU矩阵、LU_pivots矩阵以及B矩阵维度大小的合理性
2. 计算：首先根据left和adjoint参数对线性方程组进行变形，然后调用lapack中的getrs函数实现线性方程组的求解
3. 返回结果

###六、测试和验收的考量：

1. 参数left、adjoint的有效性
2. 与scipy结果一致
3. CPU、GPU结果一致
4. 动态图/静态图测试

##七、可行性分析和排期规划：

该任务主要包括以下几个步骤：1. 编写API设计文档；2. 编写lu_solve_op的C++代码和CUDA代码；3. 编写lu_solve函数的Python代码；4. 编写单元测试、性能测试、精度测试和文档测试用例；5. 运行测试并调试代码；6. 编写API文档。
排期规划：预计花费15天的时间完成此项任务。
