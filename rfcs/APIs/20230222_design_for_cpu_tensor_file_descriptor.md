# paddle cpu tensor file_descriptor 设计文档

增加 linux 下 cpu tensor file_descriptor 传输方案

| API名称      | 新增API名称                                       |
| ------------ | ------------------------------------------------- |
| 提交作者     | 核心已转储                                        |
| 提交时间     | 2023-02-21                                        |
| 版本号       | V1.0                                              |
| 依赖飞桨版本 | develop                                           |
| 文件名       | 20230222_design_for_cpu_tensor_file_descriptor.md |

# 一、概述

## 1、相关背景

本文档，主要设计完善`paddle.multiprocessing`模块。通过自定义Tensor序列化、反序列化方式，使用共享内存技术，实现paddle 的cpu Tensor在进程间快速传输、共享。

## 2、功能目标

完善 `paddle.multiprocessing`模块，可在多进程间，方便快捷的传输Tensor。

## 3、意义

multiprocessing 是支持进程间 Tensor 传输的一种方式。

# 二、飞桨现状

- 可参考 [paddle.multiprocessing 设计文档](https://github.com/PaddlePaddle/Paddle/wiki/paddle进程间tensor传输设计文档-paddle.multiprocessing)。
- 目前 paddle 支持了 file_system 的 cpu 传输方式，以文档形式存储传输tensor 的中间态。file_descriptor 打开文件句柄之后立即删除，更加安全，不容易发生文件残留
- #[37302](https://github.com/PaddlePaddle/Paddle/pull/37302) 初步支持了paddle的tensor进程间传输，需要继续完善

# 三、业内方案调研

在pytorch中，一旦张量或者存储被移动到共享单元(share_memory_)，它可以不需要任何其他复制操作的发送到其他的进程中。

1. **文件描述符的传递**

```python
def reduce_storage(storage):
    ...
    # file_descriptor方案
    fd, size = storage._share_fd_cpu_()		
    df = multiprocessing.reduction.DupFd(fd)

    metadata = (df, size)
    rebuild = rebuild_storage_fd  # type: ignore[assignment]
    ...
    return (rebuild, (type(storage),) + metadata)

def rebuild_storage_fd(cls, df, size):
    fd = df.detach()
    try:
        ...
        storage = cls._new_shared_fd_cpu(fd, size)
        ...
        return storage
    finally:
        os.close(fd)
```

通过C++接口申请共享内存，返回fd和size，由于直接在进程间传递fd并无意义，所以需要利用python.multiprocessing模块进行fd的传输

2. **序列化时**

```c++
static PyObject* THPStorage_shareFd(PyObject* _self, PyObject* noargs) {
  auto self = (THPStorage*)_self;
    
  c10::StorageImpl* storage = self->cdata;

  // 类似paddle中的phi::Allocation
  at::MapAllocator* ctx;
  
  if ((ctx = at::MapAllocator::fromDataPtr(storage->data_ptr()))) {
    // 数据已经在shmem中了
  } else {
    at::Storage new_storage(THPStorage_newFdStorage(storage->nbytes()));
    at::Storage _self_aten = torch::createStorage(_self);
    {
      // Copying into shared memory can be slow, so release the GIL
      pybind11::gil_scoped_release no_gil;
      storage_copy(new_storage, _self_aten);
    }

    std::swap(*storage, *new_storage.unsafeGetStorageImpl());
    ctx = at::MapAllocator::fromDataPtr(storage->data_ptr());
    AT_ASSERT(ctx);
  }
  
  // 伪代码:
  // return tuple(ctx->fd(), size);
}
```

3. **反序列化时**

```c++
static PyObject* THPStorage_newSharedFd(PyObject* _unused, PyObject* args) {
  ...
  PyObject* _tmp_fd = PyTuple_GET_ITEM(args, 0);
  PyObject* _size = PyTuple_GET_ITEM(args, 1);

  int fd;
  int tmp_fd = (int)THPUtils_unpackLong(_tmp_fd);
  int64_t size = THPUtils_unpackLong(_size);
  if ((fd = dup(tmp_fd)) == -1) {
    THPUtils_setError("could not duplicate a shared memory file descriptor");
    return nullptr;
  }

  int flags = at::ALLOCATOR_MAPPED_SHAREDMEM | at::ALLOCATOR_MAPPED_NOCREATE |
      at::ALLOCATOR_MAPPED_KEEPFD | at::ALLOCATOR_MAPPED_FROMFD;
  return THPStorage_New(c10::make_intrusive<at::StorageImpl>(
      c10::StorageImpl::use_byte_size_t(),
      size,
      at::MapAllocator::makeDataPtr(at::WITH_FD, "", fd, flags, size, nullptr),
      /*allocator=*/nullptr,
      /*resizable=*/false));
  END_HANDLE_TH_ERRORS
}
```

# 四、对比分析

计划采用和pytorch类似的方案，实现file_descriptor策略

# 五、设计思路与实现方案

## 命名与参数设计

参考：[飞桨API 设计及命名规范](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/api_contributing_guides/api_design_guidelines_standard_cn.html)

## API实现方案

1. 在python/paddle/incubate/multiprocessing/__init__.py中增加set_sharing_strategy和get_sharing_strategy

   ```python
   if sys.platform == 'darwin' or sys.platform == 'win32':
       _sharing_strategy = 'file_system'
   else:
       _sharing_strategy = 'file_descriptor'
   
   def set_sharing_strategy(sharing_strategy):
       if sharing_strategy != "file_descriptor" and sharing_strategy != "file_system":
           raise RuntimeError("We only support file_system mode and file_descriptor mode")
       else:
           _sharing_strategy = sharing_strategy
   
   def get_sharing_strategy():
       return _sharing_strategy;
   ```

2. 增加python/paddle/incubate/multiprocessing/reductions.py中的_reduce_lodtensor_fd和_rebuild_lodtensor_fd，支持利用fd进行序列化。_reduce_lodtensor将根据共享策略选择序列化方法

   ```python
   def _reduce_lodtensor(lodtensor):
       if get_sharing_strategy() == "file_descriptor":
           return _reduce_lodtensor_fd(lodtensor)
       else:
           return _reduce_lodtensor_fs(lodtensor)
   
   def _reduce_lodtensor_fs(lodtensor):
       # 原来 file_system 的函数
   
   def _reduce_lodtensor_fd(lodtensor):
       if (lodtensor._place().is_cpu_place()):
           for dim in lodtensor.shape():
               if dim == 0:
                   # Empty tensors have nothing be mmapped.
                   return (_rebuild_lodtensor_empty, (type(lodtensor),))
   
           metadata = (lodtensor._share_file_descriptor())  # fd, size, type_idx, dims, lod
           metadata[0] = multiprocessing.reduction.DupFd(metadata[0]) # 利用multiprocessing传输fd
       else:
           raise RuntimeError("We only support pass cpu lodtensor using file_descriptor stratege for now!")
   
       return (_rebuild_lodtensor_fd, (type(lodtensor),) + metadata)
   
   
   def _rebuild_lodtensor_fd(cls, df, size, type_idx, dims, lod):
       fd = df.detach()
       lodtensor = cls._new_file_descriptor((fd, size, type_idx, dims, lod))
       os.close(fd)
       return lodtensor
   ```

3. 在paddle/fuild/memory/allocation/mmap_allocator.h和mmap_allocator.cc中修改MemoryMapAllocation

   ```c++
   class MemoryMapAllocation : public Allocation {
     public:
       ...
       inline const int &fd() const { return fd_; }
       ...
   };
   
   void AllocateMemoryMap(
    std::string filename, int flags, size_t size, void **map_ptr_, int *fd_) {
        ...
        // 无论采用FD还是FS的传输策略, shm_open的步骤都是一样的:
        if (flags & MAPPED_SHAREDMEM) {
              fd = shm_open(filename.c_str(), file_flags, (mode_t)0600);
              PADDLE_ENFORCE_NE(
                  fd,
                  -1,
                  platform::errors::Unavailable(
                      "File descriptor %s open failed, unable in read-write mode",
                      filename.c_str()));
              VLOG(6) << "shm_open: " << filename;
        } 
        
        ...
        // 基于fd传输的策略, 需要设置MAPPED_KEEPFD的标志位
        ...
        if (flags & MAPPED_FROMFD) {
            PADDLE_ENFORCE_NE(shm_unlink(filename);,
                      -1,
                      platform::errors::Unavailable(
                          "Could not unlink the shared memory file <", filename, ">"));
        }
    }

   // 序列化时fd为-1, 这时申请一块shmem
   // 反序列化时fd为通过传输获得的, 这时直接mmap就行
   std::shared_ptr<MemoryMapAllocation>
   AllocateMemoryMapAllocationAndUnlink(int flags,
                                        size_t size,
                                        int fd) {
       void *ptr = nullptr;
       if (-1 == fd) {
           std::string handle = memory::allocation::GetIPCName();
           AllocateMemoryMap(handle, flags, size, &ptr, &fd);
       } else {
           AllocateMemoryMap("", flags, size, &ptr, &fd);
       }
       // 构造1个shmem的wapper
       return std::make_shared<MemoryMapAllocation>(
         ptr, size, "", flags, fd);
   }
   ```

4. 在paddle/fuild/pybind/tensor.cc中增加_share_file_descriptor和_new_file_descriptor的绑定，用于序列化和反序列化，传递fd

   * shm_open获得句柄后，立即shm_unlink删除inode，多进程间传输句柄。
   * 实际上，这个内存段直到访问它的所有进程都退出时才会删除
   * 句柄全部关闭后，文件系统释放存储

   ```c++
   .def("_share_file_descriptor",
        [](phi::DenseTensor &self) {
            ...
            auto *mmap_allocation = dynamic_cast<
                memory::allocation::MemoryMapAllocation *>(
                holder.get());
   
            // 如果这个tensor已经被共享过, 就可以直接返回它的metadata, 否则, 要在shmem上新开辟一块地方
            if (mmap_allocation == nullptr) {
                ...
                int flags = memory::allocation::MAPPED_SHAREDMEM |
                    memory::allocation::MAPPED_EXCLUSIVE|
                    memory::allocation::MAPPED_FROMFD|
                    memory::allocation::MAPPED_KEEPFD;
                
                auto shared_holder =
                    memory::allocation::AllocateMemoryMapAllocationAndUnlink(
                    flags, data_size, -1);
   
                memory::Copy(platform::CPUPlace(), shared_holder->ptr(),
                             platform::CPUPlace(), data_ptr, data_size);
                self.ResetHolder(shared_holder);
                mmap_allocation = shared_holder.get();
            }
            ...
   
            return py::make_tuple(mmap_allocation->fd(),
                                  mmap_allocation->size(), type_idx,
                                  vectorize(self.dims()), self.lod());
        },
        .def("_new_shared_file_descriptor",
              [](py::tuple t) {
                ...
                phi::DenseTensor tensor;
   
                const int &fd = t[0].cast<int>();
                size_t size = t[1].cast<size_t>();
                int flags = memory::allocation::MAPPED_SHAREDMEM |
                            memory::allocation::MAPPED_NOCREATE|
                            memory::allocation::MAPPED_FROMFD;
   
                auto shared_holder =
                    memory::allocation::AllocateMemoryMapAllocationAndUnlink(
                        flags, size, fd);
   
                tensor.ResetHolderWithType(
                    shared_holder,
                    static_cast<paddle::experimental::DataType>(t[2].cast<int>()));
                tensor.Resize(phi::make_ddim(t[3].cast<std::vector<int>>()));
                tensor.set_lod(t[4].cast<framework::LoD>());
   
                return tensor;
              },
   ```

# 六、测试和验收的考量

测试考虑的case如下：

1. 测试api是否可以正确执行
2. 测试tensor是否可以被正确共享，数值计算是否正确
3. 测试是否发生文件残留
4. 输入Tensor的`dtype`为`float32`、`float64`、`int32`、`int64`等类型时的结果正确性；

# 七、可行性分析和排期规划

工期上可以满足在当前版本周期内开发完成。

# 八、影响面

为独立新增API，对其他模块没有影响

# 名词解释

# 附件及参考资料
