# paddle.sparse.add 设计文档

|API名称 | paddle.sparse.add                                         | 
|---|-----------------------------------------------------------|
|提交作者<input type="checkbox" class="rowselector hidden"> | PeachML                                                   | 
|提交时间<input type="checkbox" class="rowselector hidden"> | 2022-03-29                                                | 
|版本号 | V1.0                                                      | 
|依赖飞桨版本<input type="checkbox" class="rowselector hidden"> | develop                                                   | 
|文件名 | 20220329_api_design_for_sparse_add.md<br> | 

# 一、概述

## 1、相关背景

为了提升飞桨API丰富度，add 是一个基础加法运算操作，目前 Paddle 中还没有 sparse 的加法算子。 本任务的目标是在 Paddle 中添加 sparse.add 算子， 实现输入是两个 SparseCooTensor
或者两个 SparseCsrTensor 逐元素相加的功能。 Paddle需要扩充API,新增 sparse.add API， 调用路径为：`paddle.sparse.add`
实现稀疏Tensor相加的功能。

## 3、意义

支持稀疏tensor相加，提高空间利用效率，提升稀疏tensor的计算效率。

# 二、飞桨现状

目前paddle缺少相关功能实现。

# 三、业内方案调研

## Pytorch

Pytorch中有API`torch.add(input, other, *, alpha=1, out=None)` ， 在pytorch中，介绍为：

```
Adds other, scaled by alpha, to input. 
```

可以支持一般tensor和sparse tensor的相加

## Scipy

Scipy中有csr类型的稀疏矩阵，可以支持相加操作，通过`binary_op`实现。对coo类型的稀疏矩阵，先转换成csr再相加。

### 实现方法

代码如下

```c
/*
 * Compute C = A (binary_op) B for CSR matrices that are not
 * necessarily canonical CSR format.  Specifically, this method
 * works even when the input matrices have duplicate and/or
 * unsorted column indices within a given row.
 *
 * Refer to csr_binop_csr() for additional information
 *
 * Note:
 *   Output arrays Cp, Cj, and Cx must be preallocated
 *   If nnz(C) is not known a priori, a conservative bound is:
 *          nnz(C) <= nnz(A) + nnz(B)
 *
 * Note:
 *   Input:  A and B column indices are not assumed to be in sorted order
 *   Output: C column indices are not generally in sorted order
 *           C will not contain any duplicate entries or explicit zeros.
 *
 */
template <class I, class T, class T2, class binary_op>
void csr_binop_csr_general(const I n_row, const I n_col,
                           const I Ap[], const I Aj[], const T Ax[],
                           const I Bp[], const I Bj[], const T Bx[],
                                 I Cp[],       I Cj[],       T2 Cx[],
                           const binary_op& op)
{
    //Method that works for duplicate and/or unsorted indices

    std::vector<I>  next(n_col,-1);
    std::vector<T> A_row(n_col, 0);
    std::vector<T> B_row(n_col, 0);

    I nnz = 0;
    Cp[0] = 0;

    for(I i = 0; i < n_row; i++){
        I head   = -2;
        I length =  0;

        //add a row of A to A_row
        I i_start = Ap[i];
        I i_end   = Ap[i+1];
        for(I jj = i_start; jj < i_end; jj++){
            I j = Aj[jj];

            A_row[j] += Ax[jj];

            if(next[j] == -1){
                next[j] = head;
                head = j;
                length++;
            }
        }

        //add a row of B to B_row
        i_start = Bp[i];
        i_end   = Bp[i+1];
        for(I jj = i_start; jj < i_end; jj++){
            I j = Bj[jj];

            B_row[j] += Bx[jj];

            if(next[j] == -1){
                next[j] = head;
                head = j;
                length++;
            }
        }


        // scan through columns where A or B has
        // contributed a non-zero entry
        for(I jj = 0; jj < length; jj++){
            T result = op(A_row[head], B_row[head]);

            if(result != 0){
                Cj[nnz] = head;
                Cx[nnz] = result;
                nnz++;
            }

            I temp = head;
            head = next[head];

            next[temp]  = -1;
            A_row[temp] =  0;
            B_row[temp] =  0;
        }

        Cp[i + 1] = nnz;
    }
}
```

# 四、对比分析

torch设计结构复杂，为了适配paddle phi库的设计模式，故采用scipy的实现方式

# 五、方案设计

## 命名与参数设计

在paddle/phi/kernels/sparse/目录下， kernel设计为

```    
void AddCsrKernel(const Context& dev_ctx,
                  const SparseCsrTensor& x,
                  const SparseCsrTensor& y,
                  SparseCsrTensor* out);
```

```    
//暂定
void AddCsrGradKernel(const Context& dev_ctx,
                      const SparseCsrTensor& x,
                      const SparseCsrTensor& y,
                      const SparseCsrTensor& dout,
                      SparseCsrTensor* dx,
                      SparseCsrTensor* dy);
```
函数设计为

```    
SparseCooTensor Add(const Context& dev_ctx,
                    const SparseCooTensor& x,
                    const SparseCooTensor& y);
```

和

```
SparseCsrTensor Add(const Context& dev_ctx,
                    const SparseCsrTensor& x,
                    const SparseCsrTensor& y);
```

## 底层OP设计

新增一个sparse elementwise 的功能模块（暂定），然后使用已有op组合实现， 主要涉及`SparseCooToCsrKernel`和`SparseCsrToCooKernel`。

## API实现方案

主要参考scipy实现，将coo转换成csr再进行加法，然后转换回coo

# 六、测试和验收的考量

测试考虑的case如下：

- 数值正确性

# 七、可行性分析及规划排期

方案主要依赖paddle现有op组合而成

# 八、影响面

为独立新增op，对其他模块没有影响

# 名词解释

无

# 附件及参考资料

无
