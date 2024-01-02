## PHI算子库kernel注册全流程——以bitwise_add算子为例

以`bitwise_add`这个算子为例。例如我们能在`.cc`和`.cu`文件里面看到：

```cpp
PD_REGISTER_KERNEL(bitwise_and,
                   CPU,
                   ALL_LAYOUT,
                   phi::BitwiseAndKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
```

### 面对一个个类型，如何让他们一个个注册？

显然，`PD_REGISTER_KERNEL`是注册算子的宏，我们进去看看（前方是一大堆套娃）：

```cpp
#define PD_REGISTER_KERNEL(kernel_name, backend, layout, meta_kernel_fn, ...) \
  _PD_REGISTER_KERNEL(::phi::RegType::INNER,                                  \
                      kernel_name,                                            \
                      backend,                                                \
                      ::phi::backend##Context,                                \
                      layout,                                                 \
                      meta_kernel_fn,                                         \
                      FUNCTION_KERNEL_INSTANTIATION,                          \
                      ARG_PARSE_FUNCTOR,                                      \
                      PHI_KERNEL,                                             \
                      PHI_VARIADIC_KERNEL,                                    \
                      __VA_ARGS__)

```

> 首先要明确，`#define`会在编译的预处理阶段展开，通俗来说就是复制粘贴，所以`.cc`和`.cu`最后短短的一个注册，其实会根据下面的套娃宏不断一层层展开。

可以看到，这里调用了`_PD_REGISTER_KERNEL`这个宏，我们继续展开：

```cpp
#define _PD_REGISTER_KERNEL(reg_type,                                      \
                            kernel_name,                                   \
                            backend,                                       \
                            context,                                       \
                            layout,                                        \
                            meta_kernel_fn,                                \
                            kernel_instantiation_macro,                    \
                            arg_parse_functor_macro,                       \
                            kernel_unfold_macro,                           \
                            variadic_kernel_unfold_marco,                  \
                            ...)                                           \
  PD_STATIC_ASSERT_GLOBAL_NAMESPACE(                                       \
      PD_REGISTER_tp_kernel_ns_check_##kernel_name##_##backend##_##layout, \
      "PD_REGISTER_KERNEL must be called in global namespace.");           \
  PD_EXPAND(_PD_REGISTER_2TA_KERNEL(reg_type,                              \
                                    kernel_name,                           \
                                    backend,                               \
                                    context,                               \
                                    layout,                                \
                                    meta_kernel_fn,                        \
                                    kernel_instantiation_macro,            \
                                    arg_parse_functor_macro,               \
                                    kernel_unfold_macro,                   \
                                    variadic_kernel_unfold_marco,          \
                                    __VA_ARGS__))
```

可以发现，这里调用了`PD_STATIC_ASSERT_GLOBAL_NAMESPACE`和`PD_EXPAND(x) x`

1. `PD_STATIC_ASSERT_GLOBAL_NAMESPACE`中，我们继续展开：

   ```cpp
   #define PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg) \
     _PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)
   ```

   可以看到是调用了`_PD_STATIC_ASSERT_GLOBAL_NAMESPACE`宏，我们继续展开：

   ```cpp
   #define _PD_STATIC_ASSERT_GLOBAL_NAMESPACE(uniq_name, msg)                    \
     struct __test_global_namespace_##uniq_name##__ {};                          \
     static_assert(std::is_same<::__test_global_namespace_##uniq_name##__,       \
                                __test_global_namespace_##uniq_name##__>::value, \
                   msg)
   ```

   这一小部分套娃结束，我们回忆一下这里的`uniq_name`是什么？就是把`kernel_name`(例子中的`bitwise_and`)、`backend`(例子中的`CPU`)、`layout`(例子中的`ALL_LAYOUT`)拼接一下，例子中就是`PD_REGISTER_tp_kernel_ns_check_bitwise_and_CPU_ALL_LAYOUT`，然后目的就是用`static_assert`判断一下当前注册的时候，是不是在全局namespace中注册，如果不是在全局注册，则报错。

2. `PD_EXPAND`中，我们继续展开：

   ```cpp
   #define PD_EXPAND(x) x
   ```

   看样子它直接返回了输入？具体为什么加这个，还不太清楚。它包了一层`_PD_REGISTER_2TA_KERNEL`

   然后是在`_PD_REGISTER_2TA_KERNEL`，我们继续展开（以linux下为例）：

   ```cpp
   #define _PD_REGISTER_2TA_KERNEL(reg_type,                                   \
                                   kernel_name,                                \
                                   backend,                                    \
                                   context,                                    \
                                   layout,                                     \
                                   meta_kernel_fn,                             \
                                   kernel_instantiation_macro,                 \
                                   arg_parse_functor_macro,                    \
                                   kernel_unfold_macro,                        \
                                   variadic_kernel_unfold_marco,               \
                                   ...)                                        \
     static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
         const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
     PD_EXPAND(PD_KERNEL_REGISTRAR_INIT(                                       \
         reg_type,                                                             \
         kernel_name,                                                          \
         backend,                                                              \
         context,                                                              \
         layout,                                                               \
         &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \
         meta_kernel_fn,                                                       \
         arg_parse_functor_macro,                                              \
         kernel_unfold_macro,                                                  \
         variadic_kernel_unfold_marco,                                         \
         __VA_ARGS__));                                                        \
     void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
         const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
   ```

   其中：

   >  Note: `2TA` means `2 template argument`

   在这个宏中，

   1. 声明了一个函数

      ```cpp
      static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
      ```

      在这个例子中，就是

      ```cpp
      static void __PD_KERNEL_args_def_FN_bitwise_add_CPU_ALL_LAYOUT( 
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
      ```

      这里暂时不展开，在稍后会提到这部分。

   2. 这个函数，在这里例子中，就是`__PD_KERNEL_args_def_FN_bitwise_and_CPU_ALL_LAYOUT`，而后，又是一个`PD_EXPAND`，套了一个`PD_KERNEL_REGISTRAR_INIT`宏，我们继续展开它：

      ```cpp
      #define PD_KERNEL_REGISTRAR_INIT(reg_type,                          \
                                       kernel_name,                       \
                                       backend,                           \
                                       context,                           \
                                       layout,                            \
                                       args_def_fn,                       \
                                       meta_kernel_fn,                    \
                                       arg_parse_functor_macro,           \
                                       kernel_unfold_macro,               \
                                       variadic_kernel_unfold_marco,      \
                                       ...)                               \
        PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT(PD_NARGS(__VA_ARGS__),        \
                                            reg_type,                     \
                                            kernel_name,                  \
                                            backend,                      \
                                            context,                      \
                                            layout,                       \
                                            args_def_fn,                  \
                                            meta_kernel_fn,               \
                                            arg_parse_functor_macro,      \
                                            kernel_unfold_macro,          \
                                            variadic_kernel_unfold_marco, \
                                            __VA_ARGS__))
      ```

      然后又是一个`PD_EXPAND`套了一层`_PD_KERNEL_REGISTRAR_INIT`，我们继续展开：

      ```cpp
      #define _PD_KERNEL_REGISTRAR_INIT(N,                       \
                                        reg_type,                \
                                        kernel_name,             \
                                        backend,                 \
                                        context,                 \
                                        layout,                  \
                                        args_def_fn,             \
                                        meta_kernel_fn,          \
                                        arg_parse_functor_macro,       \
                                        kernel_unfold_macro,               \
                                        variadic_kernel_unfold_marco,      \
                                        ...)                     \
        PD_EXPAND(PD_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N) ( \
          reg_type,                                              \
          kernel_name,                                           \
          backend,                                               \
          context,                                               \
          layout,                                                \
          PD_ID,                                                 \
          args_def_fn,                                           \
          meta_kernel_fn,                                        \
          arg_parse_functor_macro,                                     \
          kernel_unfold_macro,                                             \
          variadic_kernel_unfold_marco,                                    \
          __VA_ARGS__))
      ```

      这里`PD_EXPAND`套了一层`PD_CONCATENATE`，这个`PD_CONCATENATE`就是：

      ```cpp
      #define PD_CONCATENATE(arg1, arg2) PD_CONCATENATE1(arg1, arg2)
      #define PD_CONCATENATE1(arg1, arg2) PD_CONCATENATE2(arg1, arg2)
      #define PD_CONCATENATE2(arg1, arg2) arg1##arg2
      ```

      （暂时不明白为什么要套娃这么多层，为什么不直接`#define PD_CONCATENATE(arg1, arg2) arg1##arg2`呢？）

      这里concat的目的是把`_PD_KERNEL_REGISTRAR_INIT_`和`N`连接起来，这里的`N`就是`PD_NARGS(__VA_ARGS__)`，也就是在这里：

      ```cpp
      #define PD_NARGS(...) _PD_NARGS((__VA_ARGS__, _PD_RESQ_N()))
      #define _PD_NARGS(...) _PD_ARG_N(__VA_ARGS__)
      #define _PD_ARG_N_EXPAND(                                                     \
          _1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) \
        N
      #define _PD_ARG_N(args) _PD_ARG_N_EXPAND args
      #define _PD_RESQ_N() 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
      ```

      可以一步步看看宏展开了什么，第一步先把`_PD_RESQ_N()`展开，得到：

      ```cpp
      _PD_NARGS((__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
      ```

      然后把`_PD_NARGS`展开，得到：

      ```cpp
      _PD_ARG_N((__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
      ```

      然后把`_PD_ARG_N`展开，得到：

      ```
      _PD_ARG_N_EXPAND (__VA_ARGS__, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0)
      ```

      然后我们回忆一下，`__VA_ARGS__`其实就是我们注册算子的时候的那几个类型，在`bitwise_add`的例子中，就是：

      ```cpp
                         bool,
                         uint8_t,
                         int8_t,
                         int16_t,
                         int,
                         int64_t) {}
      ```

      所以把`_PD_ARG_N_EXPAND`展开，得到：

      ```cpp
      _PD_ARG_N_EXPAND(bool, uint8_t, int8_t, int16_t, int, int64_t, 15, 14, 13, 12, 11, 10, 9, 8, 7, [[6]], 5, 4, 3, 2, 1, 0)
      
      _PD_ARG_N_EXPAND(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, N, ...) N
      ```

      上下对照宏来看，会发现宏定义中`N`的位置在具体例子中是`6`，所以展开得到的是`6`，后面的`5,4,3,2,1,0`都进入了`...`部分，而`6`也恰好是我们注册的type数量。

      所以总的来看，`PD_EXPAND(PD_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N)`的`N`就是表示我们注册的类型数量，所以在

      ```cpp
        PD_EXPAND(PD_CONCATENATE(_PD_KERNEL_REGISTRAR_INIT_, N) ( \
          reg_type,                                              \
          kernel_name,                                           \
          backend,                                               \
          context,                                               \
          layout,                                                \
          PD_ID,                                                 \
          args_def_fn,                                           \
          meta_kernel_fn,                                        \
          arg_parse_functor_macro,                                     \
          kernel_unfold_macro,                                             \
          variadic_kernel_unfold_marco,                                    \
          __VA_ARGS__))
      ```

      中，对`bitwise_and`这个例子而言，就是变成了：

      ```cpp
        _PD_KERNEL_REGISTRAR_INIT_6 ( \
          reg_type,                                              \
          kernel_name,                                           \
          backend,                                               \
          context,                                               \
          layout,                                                \
          PD_ID,                                                 \
          args_def_fn,                                           \
          meta_kernel_fn,                                        \
          arg_parse_functor_macro,                                     \
          kernel_unfold_macro,                                             \
          variadic_kernel_unfold_marco,                                    \
          __VA_ARGS__))
      ```

      然后我们展开`_PD_KERNEL_REGISTRAR_INIT_6`：

      ```cpp
      #define _PD_KERNEL_REGISTRAR_INIT_6(reg_type,                         \
                                          kernel_name,                      \
                                          backend,                          \
                                          context,                          \
                                          layout,                           \
                                          registrar_id,                     \
                                          args_def_fn,                      \
                                          meta_kernel_fn,                   \
                                          arg_parse_functor_macro,          \
                                          kernel_unfold_macro,              \
                                          variadic_kernel_unfold_marco,     \
                                          cpp_dtype,                        \
                                          ...)                              \
        _PD_CREATE_REGISTRAR_OBJECT(reg_type,                               \
                                    kernel_name,                            \
                                    backend,                                \
                                    context,                                \
                                    layout,                                 \
                                    registrar_id,                           \
                                    args_def_fn,                            \
                                    meta_kernel_fn,                         \
                                    arg_parse_functor_macro,                \
                                    kernel_unfold_macro,                    \
                                    variadic_kernel_unfold_marco,           \
                                    cpp_dtype)                              \
        PD_EXPAND(_PD_KERNEL_REGISTRAR_INIT_5(reg_type,                     \
                                              kernel_name,                  \
                                              backend,                      \
                                              context,                      \
                                              layout,                       \
                                              PD_ID,                        \
                                              args_def_fn,                  \
                                              meta_kernel_fn,               \
                                              arg_parse_functor_macro,      \
                                              kernel_unfold_macro,          \
                                              variadic_kernel_unfold_marco, \
                                              __VA_ARGS__))
      ```

      可以看到，主要是调用了两个宏：`_PD_CREATE_REGISTRAR_OBJECT`和`_PD_KERNEL_REGISTRAR_INIT_5`，我们先来看第一个：

      1. `_PD_CREATE_REGISTRAR_OBJECT`的定义：

         ```cpp
         #define _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                  \
                                             kernel_name,                               \
                                             backend,                                   \
                                             context,                                   \
                                             layout,                                    \
                                             registrar_id,                              \
                                             args_def_fn,                               \
                                             meta_kernel_fn,                            \
                                             arg_parse_functor_macro,                   \
                                             kernel_unfold_macro,                       \
                                             variadic_kernel_unfold_marco,              \
                                             cpp_dtype)                                 \
           static const ::phi::KernelRegistrar PD_CONCATENATE(                          \
               __reg_phi_kernel_##kernel_name##_##backend##_##layout##_, registrar_id)( \
               reg_type,                                                                \
               #kernel_name,                                                            \
               #backend,                                                                \
               DATA_LAYOUT(layout),                                                     \
               ::phi::CppTypeToDataType<cpp_dtype>::Type(),                             \
               arg_parse_functor_macro(meta_kernel_fn, cpp_dtype, context),             \
               args_def_fn,                                                             \
               kernel_unfold_macro(meta_kernel_fn<cpp_dtype, context>),                 \
               variadic_kernel_unfold_marco(meta_kernel_fn<cpp_dtype, context>));
         ```

         这里的`cpp_dtype`就是把例子中的类型一个个拆解开来，当前在`_PD_KERNEL_REGISTRAR_INIT_6`中的`_PD_CREATE_REGISTRAR_OBJECT`传入的`cpp_dtype`就是`bool`。

         然后把`PD_CONCATENATE`展开，可以看到这个宏中主要是~~声明了一个函数~~初始化了一个类，也就是调用了`KernelRegistrar`的构造函数，在具体例子中，就是这样：

         ```cpp
           static const ::phi::KernelRegistrar __reg_phi_kernel_bitwise_add_CPU_ALL_LAYOUT_0(
               reg_type,                                                                
               "bitwise_add",                                                            
               "CPU",                                                                
               phi::DataLayout::ALL_LAYOUT,                                                     
               ::phi::CppTypeToDataType<bool>::Type(),// 这里做了cpp中type到paddle中DateType的映射，也就是得到了DataType::BOOL,本质其实是指代一个枚举"1"
               arg_parse_functor_macro(meta_kernel_fn, bool, context),             
               args_def_fn,                                                             
               kernel_unfold_macro(meta_kernel_fn<cpp_dtype, context>),                 
               variadic_kernel_unfold_marco(meta_kernel_fn<cpp_dtype, context>));
         ```

      2. `_PD_KERNEL_REGISTRAR_INIT_5`是`_PD_KERNEL_REGISTRAR_INIT_6`中的第二个部分，可以发现他们长得很像，就是最后一个数字不一样，并且可预料的，`_PD_KERNEL_REGISTRAR_INIT_5`中也会像`_PD_KERNEL_REGISTRAR_INIT_6`一样，由`_PD_CREATE_REGISTRAR_OBJECT`和`_PD_KERNEL_REGISTRAR_INIT_4`构成，这样不断递归下去。

         其实留心可以发现，例如在`_PD_KERNEL_REGISTRAR_INIT_6`的调用时：

         ```cpp
           _PD_KERNEL_REGISTRAR_INIT_6 ( \
             reg_type,                                              \
             kernel_name,                                           \
             backend,                                               \
             context,                                               \
             layout,                                                \
             PD_ID,                                                 \
             args_def_fn,                                           \
             meta_kernel_fn,                                        \
             arg_parse_functor_macro,                                     \
             kernel_unfold_macro,                                             \
             variadic_kernel_unfold_marco,                                    \
             __VA_ARGS__))
         ```

         传入了11个参数+1个可变参数宏`__VA_ARGS__`

         然后在`_PD_KERNEL_REGISTRAR_INIT_6`的定义时：

         ```cpp
         #define _PD_KERNEL_REGISTRAR_INIT_6(reg_type,                         \
                                             kernel_name,                      \
                                             backend,                          \
                                             context,                          \
                                             layout,                           \
                                             registrar_id,                     \
                                             args_def_fn,                      \
                                             meta_kernel_fn,                   \
                                             arg_parse_functor_macro,          \
                                             kernel_unfold_macro,              \
                                             variadic_kernel_unfold_marco,     \
                                             cpp_dtype,                        \
                                             ...)                              \
         ```

         是有12个参数，这就意味着，第二十个参数是从传入的`__VA_ARGS__`中解析出来的，而我们知道`__VA_ARGS__`里面存的是一个个注册的cpp type，所以，`_PD_KERNEL_REGISTRAR_INIT_6`->`_PD_KERNEL_REGISTRAR_INIT_1`这样不断递归的行为，就是把所有类型一个个拿出来，并且给他们做`_PD_CREATE_REGISTRAR_OBJECT`操作。

         然后我们可以直接直接看到`_PD_KERNEL_REGISTRAR_INIT_1`的宏定义：

         ```cpp
         #define _PD_KERNEL_REGISTRAR_INIT_1(reg_type,                                \
                                             kernel_name,                             \
                                             backend,                                 \
                                             context,                                 \
                                             layout,                                  \
                                             registrar_id,                            \
                                             args_def_fn,                             \
                                             meta_kernel_fn,                          \
                                             arg_parse_functor_macro,                 \
                                             kernel_unfold_macro,                     \
                                             variadic_kernel_unfold_marco,            \
                                             cpp_dtype)                               \
           _PD_CREATE_REGISTRAR_OBJECT(reg_type,                                      \
                                       kernel_name,                                   \
                                       backend,                                       \
                                       context,                                       \
                                       layout,                                        \
                                       registrar_id,                                  \
                                       args_def_fn,                                   \
                                       meta_kernel_fn,                                \
                                       arg_parse_functor_macro,                       \
                                       kernel_unfold_macro,                           \
                                       variadic_kernel_unfold_marco,                  \
                                       cpp_dtype)                                     \
           TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
             return 0;                                                                \
           }
         ```

         在我们的例子中，此时传入的`cpp_dtype`应该是`int64_t`了：

         ```cpp
         PD_REGISTER_KERNEL(bitwise_and,
                            CPU,
                            ALL_LAYOUT,
                            phi::BitwiseAndKernel,
                            bool,
                            uint8_t,
                            int8_t,
                            int16_t,
                            int,
                            int64_t) {}
         ```

         然后最后定义了一个函数：

         ```cpp
           TEST_API int TouchKernelSymbolFor_##kernel_name##_##backend##_##layout() { \
             return 0;                                                                \
           }
         ```

         在这个例子中，就是

         ```cpp
         TEST_API int TouchKernelSymbolFor_bitwise_add_CPU_ALL_LAYOUT() { 
         	return 0;                                                                
         }
         ```

         明白了这里是递归注册所有传入的类型，那么具体的注册过程，就要看`_PD_CREATE_REGISTRAR_OBJECT`了

      ---

### 对于某一个类型，如何进行“注册”？

> 从[飞桨高可复用算子库 PHI 设计文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)中，可以知道，"注册"就是把kernel相关信息插入到一个全局的哈希表中。

前面提到`_PD_CREATE_REGISTRAR_OBJECT`，是用来注册具体的某个类，在例子中，就是：

```cpp
  static const ::phi::KernelRegistrar __reg_phi_kernel_bitwise_add_CPU_ALL_LAYOUT_0(
      reg_type,                                                                
      "bitwise_add",                                                            
      "CPU",                                                                
      phi::DataLayout::ALL_LAYOUT,                                                     
      ::phi::CppTypeToDataType<bool>::Type(),// 这里做了cpp中type到paddle中DateType的映射，也就是得到了DataType::BOOL,本质其实是指代一个枚举"1"
      arg_parse_functor_macro(meta_kernel_fn, bool, context),             
      args_def_fn,                                                             
      kernel_unfold_macro(meta_kernel_fn<cpp_dtype, context>),                 
      variadic_kernel_unfold_marco(meta_kernel_fn<cpp_dtype, context>));
```

这是`KernelRegistrar`类的构造函数，所以我们进入`KernelRegistrar`类中看看：

可以发现，它有两个构造函数：

```cpp
  KernelRegistrar(RegType reg_type,
                  const char* kernel_name_cstr,
                  const char* backend_cstr,
                  DataLayout layout,
                  DataType dtype,  // 传入了dtype
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn,
                  void* variadic_kernel_fn) {
    ConstructKernel(reg_type,
                    kernel_name_cstr,
                    backend_cstr,
                    layout,
                    dtype,
                    args_parse_fn,
                    args_def_fn,
                    kernel_fn,
                    variadic_kernel_fn);
  }
```

和

```cpp
  KernelRegistrar(RegType reg_type,
                  const char* kernel_name_cstr,
                  const char* backend_cstr,
                  DataLayout layout,
                  KernelArgsParseFn args_parse_fn,
                  KernelArgsDefFn args_def_fn,
                  KernelFn kernel_fn,
                  void* variadic_kernel_fn) {
    for (size_t dtype = static_cast<size_t>(DataType::BOOL);
         dtype != static_cast<size_t>(DataType::NUM_DATA_TYPES);
         dtype++) {
      // NOTE(zhiqiu): why skip these types, because fluid kernel has no kernel
      // of these type.
      if (dtype == static_cast<size_t>(DataType::UINT32) ||
          dtype == static_cast<size_t>(DataType::UINT64) ||
          dtype == static_cast<size_t>(DataType::UINT16)) {
        continue;
      }
      // NOTE(zhoushunjie): Only the strings kernels can support pstring dtype
      constexpr char strings_kernels_prefix[] = "strings_";
      if (dtype == static_cast<size_t>(DataType::PSTRING) &&
          strncmp(kernel_name_cstr,
                  strings_kernels_prefix,
                  strlen(strings_kernels_prefix))) {
        continue;
      }
      ConstructKernel(reg_type,
                      kernel_name_cstr,
                      backend_cstr,
                      layout,
                      static_cast<DataType>(dtype),
                      args_parse_fn,
                      args_def_fn,
                      kernel_fn,
                      variadic_kernel_fn);
    }
  }
```

两者区别就在于，有没有传入`dtype`，在`bitwise_add`的注册过程中，传入了dtype，所以是走第一个构造函数。然后可以看到里面是调用了`ConstructKernel`，而且发现参数都是一样的，所以这里单纯包了一层，转发了一下参数，我们继续看`ConstructKernel`：

```cpp
  void ConstructKernel(RegType reg_type,
                       const char* kernel_name_cstr,
                       const char* backend_cstr,
                       DataLayout layout,
                       DataType dtype,
                       KernelArgsParseFn args_parse_fn,
                       KernelArgsDefFn args_def_fn,
                       KernelFn kernel_fn,
                       void* variadic_kernel_fn) {
    std::string kernel_name(kernel_name_cstr);
    KernelKey kernel_key(
        paddle::experimental::StringToBackend(backend_cstr), layout, dtype);
    Kernel kernel(kernel_fn, variadic_kernel_fn);
    if (kernel.GetKernelRegisteredType() == KernelRegisteredType::FUNCTION) {
      args_parse_fn(kernel_key, kernel.mutable_args_def());
    }
    args_def_fn(kernel_key, &kernel);
    if (reg_type == RegType::INNER) {
      KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
    } else {
      CustomKernelMap::Instance().RegisterCustomKernel(
          kernel_name, kernel_key, kernel);
    }
  }
```

我们一行行来看他的实现：

1. `std::string kernel_name(kernel_name_cstr);`

   这里就是转成`string`类，得到的`kernel_name`在例子中就是`"bitwise_add"`这个字符串（从最初注册时输入的`bitwise_add`，被`#kernel_name`变成`chat*`，然后在这里转成`string`类）

2. `KernelKey kernel_key(paddle::experimental::StringToBackend(backend_cstr), layout, dtype);`

   这里传入了三个参数，类型分别为`Backend`,`DataLayout`,`DataType`：

   + 第一个参数先把`backend_cstr`这个字符串转成`Backend`类，其实`Backend`类也还是个枚举类（从最初注册时输入的`CPU`，被`#backend`变成`char*`，然后在这里变成`Backend`类）

   + 第二个参数，在例子中就是`ALL_LAYOUT`（从最初注册时输入`ALL_LAYOUT`，被`DATA_LAYOUT(layout)`中的`DATA_LAYOUT`宏：`#define DATA_LAYOUT(arg__) phi::DataLayout::arg__`直接变成了`DataLayout`类，也就是现在这里的类型）

   + 第三个参数，在例子中，这里以`init_6`为例，也就是抽出第一个注册的`dtype`时，就是`DataType::BOOL`（从最初注册时输入的`bool`，被`::phi::CppTypeToDataType<cpp_dtype>::Type()`中的`CppTypeToDataType<bool>`模板类，从cpp的基础类型转化成了`DataType`类）

   然后利用这三个参数，构造了一个`KernelKey`的对象，我们看看`KernelKey`的构造函数：

   ```cpp
   class KernelKey {
    public:
     KernelKey() = default;
   
     KernelKey(Backend backend, DataLayout layout, DataType dtype)
         : backend_(backend), layout_(layout), dtype_(dtype) {}
   
     explicit KernelKey(const Place& place)
         : backend_(TransToPhiBackend(place)),
           layout_(DataLayout::ALL_LAYOUT),
           dtype_(DataType::ALL_DTYPE) {}
   
     explicit KernelKey(const int& dtype, const Place& place)
         : backend_(TransToPhiBackend(place)),
           layout_(DataLayout::ALL_LAYOUT),
           dtype_(phi::TransToPhiDataType(dtype)) {}
   
     explicit KernelKey(const Place& place,
                        const DataLayout& layout,
                        const DataType& dtype)
         : backend_(TransToPhiBackend(place)), layout_(layout), dtype_(dtype) {}
   ```

   可以看到有四个，在我们的例子中，这里会调用第一个构造函数，就是存一下`backend`,`layout`,`dtype`。其他三个构造函数功能其实一样，只是做了一下兼容性方面的处理。

3. `Kernel kernel(kernel_fn, variadic_kernel_fn);`

   + 第一个参数`kernel_fn`，我们回溯回去看看它是什么，可以在`_PD_CREATE_REGISTRAR_OBJECT`宏中，发现它就是`kernel_unfold_macro(meta_kernel_fn<cpp_dtype, context>)`，继续分析：

     + 这里的`cpp_dtpe`此时就是`bool`(以N=6时解析出第一个类型为例)，而`context`就是之前`::phi::backend##Context`根据`backend`得到的，例子中的`context`就是`::phi::CPUContext`

       而`meta_kernel_fn`则是我们在注册时候传入的`phi::BitwiseAndKernel`，这就是我们在`.cc`中实现的`kernel`：

       ```cpp
       #define DEFINE_BITWISE_KERNEL(op_type)                                 \
         template <typename T, typename Context>                              \
         void Bitwise##op_type##Kernel(const Context& dev_ctx,                \
                                       const DenseTensor& x,                  \
                                       const DenseTensor& y,                  \
                                       DenseTensor* out) {                    \
           funcs::Bitwise##op_type##Functor<T> func;                          \
           funcs::ElementwiseCompute<funcs::Bitwise##op_type##Functor<T>, T>( \
               dev_ctx, x, y, func, out);                                     \
         }
       
       DEFINE_BITWISE_KERNEL(And)
       ```

       可以看到，在kernel的template中，就能对上了，`T`就是此时注册的cpp类型，然后`context`就是根据`backend`得到的上下文信息`CPUContext`

     + 接下来是调用了`kernel_unfold_macro`这个宏，而这个`kernel_unfold_macro`就是`PHI_KERNEL`宏，它一直作为参数一层层传到这里才发生展开，我们看看它做了什么：

       ```
       #define PHI_KERNEL(...) \
         ::phi::KernelImpl<decltype(&__VA_ARGS__), &__VA_ARGS__>::Compute
       ```

       可以看到，`&__VA_ARGS__`此时就是`BitwiseAddKernel`的函数指针，在这个例子中，就是变成：

       ```
       ::phi::KernelImpl<decltype(&BitwiseAddKernel), &BitwiseAddKernel>::Compute
       ```

       这是一个`Kernel_Fn`类型：

       ```cpp
       using KernelFn = std::function<void(KernelContext* ctx)>;
       ```

       这里由于`::phi::KernelImpl<decltype(&BitwiseAddKernel), &BitwiseAddKernel>::Compute`传入的是"函数指针类别"和"具体的函数指针"，刚好和` KernelImpl<Return (*)(DevCtx, Args...), kernel_fn>`匹配。

       可以看到调用了`KernelImpl`中的`Compute`方法：

       ```cpp
       template <typename Fn, Fn fn>
       struct KernelImpl;
       
       template <typename Return,
                 typename DevCtx,
                 typename... Args,
                 Return (*kernel_fn)(DevCtx, Args...)>
       struct KernelImpl<Return (*)(DevCtx, Args...), kernel_fn> {
         static void Compute(KernelContext* ctx) {
           KernelCallHelper<DevCtx, Args..., TypeTag<int>>::
               template Compute<0, 0, 0, 0>(ctx);
         }
       ```

       可以看到它的静态方法`Compute`调用了`KernelCallHelper`，此时模板中的`Return`被自动推导得到`void`，`DevCtx`被自动推导为`Context`类，`Args`则是将后面的其他参数打包成了参数包。

       至此，`kernel_unfold_macro`的展开就是得到了这里的`Compute`方法。

       也就是`Kernel`对象构建：

       ```cpp
       Kernel kernel(kernel_fn, variadic_kernel_fn);
       ```

       这里的第一个参数`kernel_fn`

   + 第二个参数`variadic_kernel_fn`的流程和第一个参数`kernel_fn`基本一致。

     传入的是`PHI_VARIADIC_KERNEL(kernel_fn)`，可以看看`PHI_VARIADIC_KERNEL`这个宏：

     ```cpp
     #define PHI_VARIADIC_KERNEL(...)                                     \
       reinterpret_cast<void*>(&::phi::KernelImpl<decltype(&__VA_ARGS__), \
                                                  &__VA_ARGS__>::VariadicCompute)
     ```

     在这个例子中，就是展开变成

     `reinterpret_cast<void*>(&::phi::KernelImpl<decltype(&BitwiseAddKernel),&BitwiseAddKernel>::VariadicCompute)`

     和前面的`kernel_fn`相比，这里依然是用的同一个`KernelImpl`实例（传入给模板的参数和前面一样），调用了`VariadicCompute`静态方法：

     ```cpp
       static void VariadicCompute(const DeviceContext& dev_ctx, Args... args) {
         return kernel_fn(static_cast<DevCtx>(dev_ctx), std::forward<Args>(args)...);
       }
     ```

     得到的是这个函数指针，指向由`BitwiseAddKernel`相关信息特化的`VariadicCompute`函数。

     在编译时期，传入的`kernel_fn`，也就是`BitwiseAddKernel`，作为宏体中的`__VA_ARGS__`，顺利地将`::phi::KernelImpl`进行了实例化，所以此时`variadic_kernel_fn`这个参数应该不为空。

     

   然后构造`Kernel`对象：

   ```cpp
     explicit Kernel(KernelFn fn, void* variadic_fn)
         : fn_(fn), variadic_fn_(variadic_fn) {
       if (variadic_fn == nullptr) {
         kernel_registered_type_ = KernelRegisteredType::STRUCTURE;
       } else {
         kernel_registered_type_ = KernelRegisteredType::FUNCTION;
       }
     }
   ```

   可以发现，主要是存了一下传入的`fn`和`variadic_fn`，因为`variadic_fn`不为空，所以`kernel_registered_type_`赋值为`KernelRegisteredType::FUNCTION`（看到这里涉及`structure`和`function`，猜测这块可能是兼容老的op体系用的？老的fluid体系为结构体算子，新的phi体系算子为函数式算子）

4. ```cpp
   if (kernel.GetKernelRegisteredType() == KernelRegisteredType::FUNCTION) {
         args_parse_fn(kernel_key, kernel.mutable_args_def());
   }
   ```

   前面我们知道，在例子中，kernel的`GetKernelRegisteredType`为`KernelRegisteredType::FUNCTION`，所以这里是要走if的。

   `args_parse_fn`是来自于`arg_parse_functor_macro(meta_kernel_fn, cpp_dtype, context)`，在例子中，就是：

   ```cpp
   arg_parse_functor_macro(phi::BitwiseAddKernel, bool, CPUContext)
   ```

   而`arg_parse_functor_macro`就是一个宏，我们看看他的定义：

   ```cpp
   // The macro for passing KernelArgsParseFunctor's function
   #define ARG_PARSE_FUNCTOR(meta_kernel_fn, cpp_dtype, context) \
     ::phi::KernelArgsParseFunctor<                              \
         decltype(&meta_kernel_fn<cpp_dtype, context>)>::Parse
   ```

   可以发现这个例子中，展开后是这样的：

   ```cpp
   ::phi::KernelArgsParseFunctor<decltype(&phi::BitwiseAddKernel<bool, CPUContext>)>::Parse
   ```

   看看`KernelArgsParseFunctor`中的实现：

   ```cpp
   template <typename Return_, typename... Args_>
   struct KernelArgsParseFunctor<Return_ (*)(Args_...)> {
     using Args = std::tuple<Args_...>;
     enum : std::size_t { Arity = sizeof...(Args_) };
     using Indices = std::make_index_sequence<Arity>;
     template <std::size_t Index>
     using Arg = typename std::tuple_element<Index, Args>::type;
   
     static void Parse(const KernelKey& default_key, KernelArgsDef* args_def) {
       // TODO(chenweihang): The fluid Tensor's default layout is NCHW,
       // it is not same as kernel's layout, we should fix this error on
       // fluid Tensor
   
       auto args_type = ParseArgType(Indices{});
       SetKernelArgsDef(args_type, default_key, args_def);
     }
   
    private:
     template <std::size_t... INDEX>
     static std::vector<std::type_index> ParseArgType(
         std::index_sequence<INDEX...>) {
       return {std::type_index(typeid(Arg<INDEX>))...};
     }
   };
   ```

   在前面的if中，调用了`args_parse_fn(kernel_key, kernel.mutable_args_def());`，所以就是：

   ```cpp
   ::phi::KernelArgsParseFunctor<decltype(&phi::BitwiseAddKernel<bool, CPUContext>)>::Parse(kernel_key, kernel.mutable_args_def())
   ```

   传入的第一个参数是带有`backend,layout,dtype`的`KernelKey`信息，第二个参数是`kernel`对象的指向成员变量`args_def_`的`KernelArgsDef*`指针：

   ```cpp
   KernelArgsDef* mutable_args_def() { return &args_def_; }
   ```

   当然此时这个`kernel`的`args_def_`是空的，而接下来要做的`Parse`操作，就是从传入的`kernel_key`中提取参数信息，去填充`kernel`对象的`args_def`变量（而不是利用这个变量去做什么其他事情，他是作为输出传进来的）。我们看看这里的`Parse`具体是怎么做的，下面有比较多的`std`标准库元函数的使用：

   + ```cpp
     auto args_type = ParseArgType(Indices{});
     ```

     这里的`Indices`是`std::make_index_sequence<Arity>;`，而`Arity`是`enum : std::size_t { Arity = sizeof...(Args_) };`，可以知道，这里的`Arity`表示的是参数包大小，在例子中，就是表示`phi::BitwiseAddKernel`的参数量：

     ```
     #define DEFINE_BITWISE_KERNEL(op_type)                                 \
       template <typename T, typename Context>                              \
       void Bitwise##op_type##Kernel(const Context& dev_ctx,                \
                                     const DenseTensor& x,                  \
                                     const DenseTensor& y,                  \
                                     DenseTensor* out) {                    \
         funcs::Bitwise##op_type##Functor<T> func;                          \
         funcs::ElementwiseCompute<funcs::Bitwise##op_type##Functor<T>, T>( \
             dev_ctx, x, y, func, out);                                     \
       }
     
     DEFINE_BITWISE_KERNEL(And)
     ```

     可见，`Arity`就是`4`，所以`Indices{}`将会是一个从 0 到 3 的整数序列。我们继续看`ParseArgType`的实现：

     ```cpp
       template <std::size_t... INDEX>
       static std::vector<std::type_index> ParseArgType(
           std::index_sequence<INDEX...>) {
         return {std::type_index(typeid(Arg<INDEX>))...};
       }
     ```

     这里是通过传入的`Indices{}`，推导出了`INDEX`就是0到3的整数序列。

     然后`Arg`是：

     ```cpp
       template <std::size_t Index>
       using Arg = typename std::tuple_element<Index, Args>::type;
     ```

     其中的`Args`是一个tuple：

     ```cpp
     using Args = std::tuple<Args_...>;
     ```

     可以发现，`Arg<Index>`就是在`Args`这个tuple中取下标为`Index`的元素。

     而后

     `{std::type_index(typeid(Arg<INDEX>))...}`是一个折叠表达式，展开可以得到：

     ```cpp
     return {std::type_index(typeid(Arg<0>)), std::type_index(typeid(Arg<1>)), std::type_index(typeid(Arg<2>)), std::type_index(typeid(Arg<3>))}
     ```

     然后将`Arg`展开，在具体例子中，得到：

     ```cpp
     return {std::type_index(typeid(const CPUContext&)), std::type_index(typeid(const DenseTensor&)), std::type_index(typeid(const DenseTensor&)), std::type_index(typeid(DenseTensor*))}
     ```

     所以`ParseArgType`这个函数就是巧妙地利用了模板自动推导，在`::phi::KernelArgsParseFunctor<decltype(&phi::BitwiseAddKernel<bool, CPUContext>)>`实例化的时候拆解了kernel，自动推导得到了当前kernel的参数类型，保存在`args_type`变量中。然后我们看接下来如何把这些类型信息赋值给`args_def`。

   + ```cpp
     SetKernelArgsDef(args_type, default_key, args_def);
     ```

     可以看到，这里调用了`SetKernelArgsDef`函数。这个函数比较简单，篇幅较长，下面我们一段段来看：

     ```cpp
     void SetKernelArgsDef(const std::vector<std::type_index>& args_type,
                           const KernelKey& default_key,
                           KernelArgsDef* args_def) {
       auto default_tensor_layout = phi::DataLayout::NCHW;
       if (default_key.layout() != phi::DataLayout::ANY) {
         default_tensor_layout = default_key.layout();
       }
     ```

     首先，这里做了一下特殊的处理，应该是和前面的todo相关：

     ```cpp
       static void Parse(const KernelKey& default_key, KernelArgsDef* args_def) {
         // TODO(chenweihang): The fluid Tensor's default layout is NCHW,
         // it is not same as kernel's layout, we should fix this error on
         // fluid Tensor
     ```

     具体例子中，由于传入的`kernel_key`的layout是`ALL_LAYOUT`，也就是`ANY`，所以tensor的默认layout并不影响，所以不用走这个if语句。

     然后是for循环，遍历我们前面拿到的kernel的所有参数：

     ```cpp
       for (auto arg_type : args_type) {
         if (arg_type == std::type_index(typeid(const CPUContext&))
     #if defined(PADDLE_WITH_DNNL)
             || arg_type == std::type_index(typeid(const OneDNNContext&))
     #endif
     #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
             || arg_type == std::type_index(typeid(const GPUContext&))
     #elif defined(PADDLE_WITH_XPU) && !defined(PADDLE_WITH_XPU_KP)
             || arg_type == std::type_index(typeid(const XPUContext&))
     #elif defined(PADDLE_WITH_XPU) && defined(PADDLE_WITH_XPU_KP)
               || arg_type == std::type_index(typeid(const KPSContext&))
     #endif
     #if defined(PADDLE_WITH_CUSTOM_DEVICE)
             || arg_type == std::type_index(typeid(const CustomContext&))) {
     #else
         ) {
     #endif
           // do nothing, skip context arg now
         }
     ```

     可以发现，如果遇到参数类型是`Context`相关的，就跳过这个参数，在具体例子中，就会跳过`std::type_index(typeid(const CPUContext&))`这个参数，剩下还有：

     ```cpp
     {std::type_index(typeid(const DenseTensor&)), std::type_index(typeid(const DenseTensor&)), std::type_index(typeid(DenseTensor*))}
     ```

     这三个参数。

     而后，是检测输入相关的参数：

     ```cpp
     else if (arg_type == std::type_index(typeid(const DenseTensor&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type ==
                    std::type_index(typeid(const paddle::optional<DenseTensor>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type ==
                    std::type_index(typeid(
                        const paddle::optional<std::vector<const DenseTensor*>>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type ==
                    std::type_index(typeid(const paddle::optional<SelectedRows>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    const std::vector<const DenseTensor*>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type ==
                    std::type_index(typeid(const phi::ExtendedTensor&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    const std::vector<const ExtendedTensor*>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    const std::vector<const SelectedRows*>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type ==
                    std::type_index(typeid(const std::vector<const TensorBase*>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    const std::vector<const TensorArray*>&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(const SelectedRows&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(const StringTensor&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(const SparseCooTensor&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    paddle::optional<const SparseCooTensor&>))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(const SparseCsrTensor&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(
                                    paddle::optional<const SparseCsrTensor&>))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
         } else if (arg_type == std::type_index(typeid(const TensorArray&))) {
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
     ```

     从这里其实就可以知道，我们在算子注册的时候，输入变量的类型为什么必须严格带`const`，而且常常需要`DenseTensor`的指针了把。因为要在这里对上，才能顺利地把参数类型信息存到`args_def`中去。

     具体例子中，我们有输入相关的`std::type_index(typeid(const DenseTensor&)), std::type_index(typeid(const DenseTensor&))`这两个参数，都是走这个：

     ```cpp
           args_def->AppendInput(default_key.backend(),
                                 default_tensor_layout,
                                 default_key.dtype(),
                                 arg_type);
     ```

     所以`args_def`中的`input_defs_`现在存了`TensorArgDef(Backend::CPU, DataLayout::ALL_LAYOUT, DataType::BOOL, std::type_index(typeid(const DenseTensor&)))`，`TensorArgDef(Backend::CPU, DataLayout::ALL_LAYOUT, DataType::BOOL, std::type_index(typeid(const DenseTensor&)))`这两个相同元素

     接下来是一系列输出类型的判断：

     ```cpp
     else if (arg_type == std::type_index(typeid(DenseTensor*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(std::vector<DenseTensor*>))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(SelectedRows*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(TensorArray*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(SparseCooTensor*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(SparseCsrTensor*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(StringTensor*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
         } else if (arg_type == std::type_index(typeid(ExtendedTensor*))) {
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
     ```

     具体例子中，此时遍历到了最后一个参数`std::type_index(typeid(DenseTensor*))`

     显然会走这个：

     ```cpp
           args_def->AppendOutput(default_key.backend(),
                                  default_tensor_layout,
                                  default_key.dtype(),
                                  arg_type);
     ```

     所以给`args_def`的`output_defs_`中加入了`TensorArgDef(backend, layout, dtype, type_index)`

     具体例子中就是：

     ```cpp
     TensorArgDef(Backend::CPU, DataLayout::ALL_LAYOUT, DataType::BOOL, std::type_index(typeid(DenseTensor*)))
     ```

     这样就遍历完了四个参数，存入了`args_def`中，完成了`Parse`操作。

   5. ```
      args_def_fn(kernel_key, &kernel);
      ```

      我们继续回到`ConstructKernel`中，前面利用`kernel_key`中的`backend, layout, dtype`，结合具体kernel实现时的参数类型，已经完善了`kernel`对象的`args_def_`成员变量，存好了这些相关信息。

      接下来是`args_def_fn`操作，我们需要回溯到`_PD_REGISTER_2TA_KERNEL`这个宏定义中去查看，以linux下为例：

      ```cpp
      #define _PD_REGISTER_2TA_KERNEL(reg_type,                                   \
                                      kernel_name,                                \
                                      backend,                                    \
                                      context,                                    \
                                      layout,                                     \
                                      meta_kernel_fn,                             \
                                      kernel_instantiation_macro,                 \
                                      arg_parse_functor_macro,                    \
                                      kernel_unfold_macro,                        \
                                      variadic_kernel_unfold_marco,               \
                                      ...)                                        \
        static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( \
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);           \
        PD_EXPAND(PD_KERNEL_REGISTRAR_INIT(                                       \
            reg_type,                                                             \
            kernel_name,                                                          \
            backend,                                                              \
            context,                                                              \
            layout,                                                               \
            &__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout,        \ /* 此为之后的args_def_fn */
            meta_kernel_fn,                                                       \
            arg_parse_functor_macro,                                              \
            kernel_unfold_macro,                                                  \
            variadic_kernel_unfold_marco,                                         \
            __VA_ARGS__));                                                        \
        void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
      ```

      可以看到，`args_def_fn`是一个函数指针`&__PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout`，具体例子中，这里就是：

      ```cpp
      &__PD_KERNEL_args_def_FN_bitwise_add_CPU_ALL_LAYOUT
      ```

      这里有他的声明：

      ```cpp
      static void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout( 
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
      ```

      具体则是：

      ```cpp
      static void __PD_KERNEL_args_def_FN_bitwise_add_CPU_ALL_LAYOUT( 
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel);
      ```

      定义则在下面：

      ```cpp
        void __PD_KERNEL_args_def_FN_##kernel_name##_##backend##_##layout(        \
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel)
      ```

      这里的定义长得比较奇怪，第一眼：这是定义？？？括号呢？？？具体实现呢？？？

      然后仔细观察了一下，应该是注册的时候：

      ```cpp
      PD_REGISTER_KERNEL(bitwise_and,
                         CPU,
                         ALL_LAYOUT,
                         phi::BitwiseAndKernel,
                         bool,
                         uint8_t,
                         int8_t,
                         int16_t,
                         int,
                         int64_t) {}
      ```

      最后不是带了个大括号`{}`吗？宏展开后，这个大括号就跑到这个定义下来了，这也是为什么定义写在了最后面，而不是直接跟声明写在一起，就是为了把这个地方的实现暴露给开发者，可以直接在注册的时候调整`kernel`和`kernel_key`，很巧妙。所以这个函数在这个例子中，应该确实是什么都不干，不过在其他例子中，就有用到这块设计的，例如在`full_kernel.cc`中：

      ```cpp
      namespace phi {
      
      template <typename T, typename Context>
      void FullBatchSizeLikeKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<int>& shape UNUSED,
                                   const Scalar& val,
                                   DataType dtype,
                                   int x_batch_size_dim,
                                   int out_batch_size_dim,
                                   DenseTensor* out) {
        if (!x.lod().empty() && x_batch_size_dim == 0) {
          // set the correct batch size for the LoDTensor.
          auto odims = out->dims();
          odims[out_batch_size_dim] = static_cast<int>(x.lod().back().size()) - 1;
          FullKernel<T, Context>(dev_ctx, common::vectorize(odims), val, dtype, out);
        }
        FullLikeKernel<T, Context>(dev_ctx, x, val, dtype, out);
      }
      
      }  // namespace phi
      
      PD_REGISTER_KERNEL(full_batch_size_like,
                         CPU,
                         ALL_LAYOUT,
                         phi::FullBatchSizeLikeKernel,
                         float,
                         double,
                         int,
                         int64_t,
                         bool) {
        kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
      }
      #if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      PD_REGISTER_KERNEL(full_batch_size_like,
                         GPU,
                         ALL_LAYOUT,
                         phi::FullBatchSizeLikeKernel,
                         float,
                         double,
                         int,
                         int64_t,
                         bool,
                         phi::dtype::float16,
                         phi::dtype::bfloat16) {
        kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
      }
      #endif
      
      ```

      这里的模板参数`Context`在注册cpu下的kernel时，有这样的定义：

      ```cpp
      void __PD_KERNEL_args_def_FN_full_batch_size_like_CPU_ALL_LAYOUT(        
            const ::phi::KernelKey& kernel_key, ::phi::Kernel* kernel) {
          kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
      }
      ```

      这里的`kernel->InputAt(0)`其实就是kernel的`const DenseTensor& x`

      所以在`args_def_fn(kernel_key, &kernel);`的时候，在这之前，由于注册时指定了`CPU`，所以`kernel_key`的backend是cpu，所以之前存入的参数`x`的backend就是cpu。现在就是”逆天改命“，把`x`的backend变成`ALL_BACKEND`；第二个也同理，注册的backend为gpu，也是在这时候改成`ALL_BACKEND`。

      （思考：对一个输入tensor而言，他的backend（或者说，处在这个位置的参数的backend意味着什么？）这是否意味着，在调用算子进行计算的时候，框架自动对其backend进行检测？有的也对输出的dtype进行修改，可能这里会调整kernel允许输出的类别？譬如什么都不加，就是输入`T`输出`T`，加入类似于`kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);`这种就是输入`T`，输出`bool`）

      

   6. ```cpp
      if (reg_type == RegType::INNER) {
          KernelFactory::Instance().kernels()[kernel_name][kernel_key] = kernel;
      } else {
          CustomKernelMap::Instance().RegisterCustomKernel(
              kernel_name, kernel_key, kernel);
      }
      ```

      最后，就是把`kernel_key`和`kernel`存到`KernelFactory`里面，在具体例子中，`reg_type`为`RegType::INNER`，所以走if分支。

      然后可以看看`KernelFactory`这个工厂模式的设计：

      ```cpp
      class KernelFactory {
       public:
        static KernelFactory& Instance();
        KernelNameMap& kernels() { return kernels_; }
        bool HasCompatiblePhiKernel(const std::string& op_type) const;
        bool HasStructuredKernel(const std::string& op_type) const;
        KernelResult SelectKernelOrThrowError(const std::string& kernel_name,
                                              const KernelKey& kernel_key,
                                              bool use_strided_kernel = false) const;
        bool HasKernel(const std::string& kernel_name,
                       const KernelKey& kernel_key) const;
        const Kernel& SelectKernel(const std::string& kernel_name,
                                   const KernelKey& kernel_key) const;
        const Kernel& SelectKernelWithGPUDNN(const std::string& kernel_name,
                                             const KernelKey& kernel_key) const;
        KernelKeyMap SelectKernelMap(const std::string& kernel_name) const;
        const KernelArgsDef& GetFirstKernelArgsDef(
            const std::string& kernel_name) const;
        void AddToLowPrecisionKernelList(const std::string& name,
                                         const DataType& kernel_key_type);
        std::map<const std::string, OpCount> GetLowPrecisionKernelList();
        void ClearLowPrecisionKernelList() { low_precision_kernels_.clear(); }
       private:
        KernelFactory() = default;
        KernelNameMap kernels_;
        // Get the low precision kernel list of current module.
        std::map<const std::string, OpCount> low_precision_kernels_;
      };
      ```

      全局的静态实例`Instance`，掌管着一个`KernelNameMap`类的成员变量`kernels_`

      而`KernelNameMap`就是`using KernelNameMap = paddle::flat_hash_map<std::string, KernelKeyMap>;`这样一个哈希表，然后`KernelKeyMap`同样是`using KernelKeyMap = paddle::flat_hash_map<KernelKey, Kernel, KernelKey::Hash>;`这样一个哈希表。所以这部分其实就是**通过`kernel_name`找到一个哈希表，然后再通过`kernel_key`再找一层，这样就能找到想要的具体`kernel`对象了**。可以从[飞桨高可复用算子库 PHI 设计文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)中看到，整体的设计如下图，可以看到图中组织地非常清晰了。

      将当前的kernel相关信息插入到哈希表中，这样就完成了这个算子的注册。

   ![kernel-design.png](https://github.com/PaddlePaddle/docs/raw/develop/docs/design/phi/images/kernel-design.png)

   > - `KernelFactory`作为管理 Kernel 的全局单例数据结构，和 fluid 的 OpKernelMap 类似，两级 map，第一层根据 name 找到 Kernel 集合，第二层根据 KernelKey 找到具体的 Kernel
   > - `KernelKey`和原先的 OpKernelType 类似，但将 place 和 library_type 字段合二为一称之为 Backend，因为原先的 LibraryType 是一个有局限的枚举类，原本就和 place 是强相关的，拆分反而增加了理解成本
   > - `Kernel`相比原先的 OpKernel 持有了更多信息，除了执行时的 Function，还持有了具体参数的信息，即`KernelArgsDef`，对于 Tensor 类输入输出，保存了 Tensor 类型信息、Device，数据类型、数据布局，对于 Attribute 类输入输出，保存了类型信息

   

   

   

   



## 参考资料

[飞桨高可复用算子库 PHI 设计文档](https://github.com/PaddlePaddle/docs/blob/develop/docs/design/phi/design_cn.md)

[Kernel选择分发体系梳理与优化](https://github.com/PaddlePaddle/community/blob/master/pfcc/paddle-code-reading/kernel_selection/20221130_kernel_selection.md)















