## 实现 Paddle flags 工具库

| 版本 | 作者      | 时间      |
| ---- | --------- | -------- |
| V1.0 | huangjiyi | 2023.7.31 |
| V2.0 | huangjiyi | 2023.8.21 |

## 一、概要

### 1. 相关背景

目前 Paddle 已基本完成 PHI 算子库的独立编译 ([PR#53735](https://github.com/PaddlePaddle/Paddle/pull/53735))，在实现这个目标的过程中出现过一个问题：phi 中用到 gflags 第三方库的 Flag 定义宏在 phi 编译成动态链接库后无法在 windows 上暴露 Flag 符号，当时的做法是在 phi 下重写 Flag 定义宏 (底层仍然依赖 gflags 第三方库)，使其能够在 windows 上暴露 Flag 符号 ([PR#52991](https://github.com/PaddlePaddle/Paddle/pull/52991))

但是目前还存在 gflags 第三方库相关的另外一个问题：由于 Paddle C++ 库中包含了 gflags 库文件，外部用户同时使用 paddle C++ 库和 gflags 库时，会出现以下错误：

``` bash
ERROR: something wrong with flag 'flagfile' in file '/Paddle/third_party/gflags/src/gflags.cc'.  One possibility: file '/Paddle/third_party/gflags/src/gflags.cc' is being linked both statically and dynamically into this executable.
```

这个错误是因为在 gflags 的源文件 `gflags.cc` 中，会注册一些 Flag，比如 `flagfile`：

``` C++
DEFINE_string(flagfile,   "", "load flags from file");
```

因为 Paddle 库中的 gflags 库文件会注册 `flagfile`，然后外部用户如果再依赖 gflags，会重复注册 `flagfile` 导致报错，`gflags.cc` 中的报错相关代码：

``` C++
void FlagRegistry::RegisterFlag(CommandLineFlag* flag) {
  Lock();
  pair<FlagIterator, bool> ins =
    flags_.insert(pair<const char*, CommandLineFlag*>(flag->name(), flag));
  if (ins.second == false) {   // means the name was already in the map
    if (strcmp(ins.first->second->filename(), flag->filename()) != 0) {
      ReportError(DIE, "ERROR: flag '%s' was defined more than once "
                  "(in files '%s' and '%s').\n",
                  flag->name(),
                  ins.first->second->filename(),
                  flag->filename());
    } else {
      ReportError(DIE, "ERROR: something wrong with flag '%s' in file '%s'.  "
                  "One possibility: file '%s' is being linked both statically "
                  "and dynamically into this executable.\n",
                  flag->name(),
                  flag->filename(), flag->filename());
    }
  }
  // Also add to the flags_by_ptr_ map.
  flags_by_ptr_[flag->current_->value_buffer_] = flag;
  Unlock();
}
```

另外，对于 Paddle 目前的使用需求，gflags 中的的很多功能是冗余的。

针对上述问题，计划针对 Paddle 的功能需求实现一个精简的 flags 独立工具库来替换 gflags。

### 2. 功能目标

在 Paddle 下实现一套独立的 flags 工具库，包括：

- 多种类型（bool, int32, uint32, int64, uint64, double, string）的 Flag 定义和声明宏
- 命令行参数解析，即根据命令行参数对已定义的 Flag 的 value 进行更新
- 根据环境变量的值对对应的 Flag 进行赋值
- 其他 Paddle 用到的 Flag 相关操作

初期暂时将 Paddle 目前依赖的第三方库的 gflags 保留，实现能够通过编译选项以及宏控制，选择使用哪个版本 flags 工具库（需要两个版本的接口一致），待后续 Paddle 下独立的 flags 工具库完善后，再考虑移除 gflags 第三方库。

### 3. 意义

完善 Paddle 下的 flags 工具，提高框架开发者开发体验以及用户使用体验

## 二、飞桨现状

Paddle 目前在 `paddle/phi/core/flags.h` 中对 gflags 中的 Flag 注册宏 `DEFINE_<type>` 和声明宏 `DEFINE_<type>` 进行了重写，重写的代码和 gflags 的实现基本一致，只是修改了一些接口名字和命名空间，同时添加了支持 Windows 下的 Flag 符号暴露，但 Paddle 目前的 Flag 注册宏和声明宏底层依然依赖的是 gflags 的代码

### Paddle 中现有的 gflags 用法

以下是 Paddle 中存在的一个 gflags 用法及其使用场景：

1. 目前 Paddle 中使用最多的接口是 Flag 注册和声明宏：`(PHI_)?(DEFINE|DECLARE)_<type>`，其中有 `PHI_` 前缀的宏是 Paddle 的重写版本，底层实现与 `(DEFINE|DECLARE)_<type>` 基本一致：
   - `(PHI_)?DEFINE_<type>(name,val, txt)`：定义全局标志变量 `FLAGS_name`，并且将 flag 的一些信息进行注册，约 200+ 处用法
   - `(PHI_)?DECLARE_<type>(name)` 用于声明 FLAG 全局变量，`extern` 用法，用于需要访问 `FLAGS_name` 的场景，约 300+ 处用法
2. `gflags::ParseCommandLineFlags(int* argc, char*** argv, bool remove_flags)`：命令行标志解析，约 20+ 处用法
   - 在 Paddle 中，一部分 `ParseCommandLineFlags` 在测试文件中使用，用于在命令行运行测试程序时设置一些可选参数；
   - 还有一部分在命令行输入 argv 的基础上，手动添加一些 flag，比如添加 `--tryfromenv` 设置环境变量 flag，再调用 `ParseCommandLineFlags` 进行解析，因为 gflags 没有直接根据环境变量设置 flags 的接口，所以才通过这种方式实现
3. `bool GetCommandLineOption(const char* name, std::string* OUTPUT)`：查找一个 flag，如果存在则将 `FLAG_##name` 存放在 `OUTPUT`，在 Paddle 中只用到了查找功能来判断一个 flag 是否被定义，1 处用法
4. `std::string SetCommandLineOption(const char* name, const char* value)`：用于将 `FLAG_##name` 的值设置为 `value`，2 处用法
5. `void AllowCommandLineReparsing()`：Paddle 中有一处用法放在 `ParseCommandLineFlags` 之前调用，函数名叫允许命令行重新解析，但在 `gflags.cc` 实现代码中，这个设置只是允许 `ParseCommandLineFlags` 传入一些未定义的 flag 而不报错，1 处用法

## 三、业内方案调研

### gflags

ref: https://github.com/gflags/gflags

![image-20230728163636633](https://github.com/huangjiyi/community/assets/43315610/95e0fff1-5c38-4d21-b5ec-0bb6052086e5)

上图中只列出了一些关键数据结构及其关键成员变量和方法

- `DEFINE_<type>(name, val, txt)`：除 string 类型外，`DEFINE_<type>` 底层均调用 `DEFINE_VARIABLE`

- `DECLARE_<type>(name)`：统一调用 `DECLARE_VARIABLE`，实现就是简单的 `extern` 用法

- `DEFINE_VARIABLE`：关键 Flag 定义宏

  ``` C++
  #define DEFINE_VARIABLE(type, shorttype, name, value, help)             \
    namespace fL##shorttype {                                             \
      static const type FLAGS_nono##name = value;                         \
      /* We always want to export defined variables, dll or no */         \
      GFLAGS_DLL_DEFINE_FLAG type FLAGS_##name = FLAGS_nono##name;        \
      static type FLAGS_no##name = FLAGS_nono##name;                      \
      static GFLAGS_NAMESPACE::FlagRegisterer o_##name(                   \
        #name, MAYBE_STRIPPED_HELP(help), __FILE__,                       \
        &FLAGS_##name, &FLAGS_no##name);                                  \
    }                                                                     \
    using fL##shorttype::FLAGS_##name
  ```

  `DEFINE_VARIABLE` 中定义了 3 个变量：

  - `FLAGS_##name`：全局变量，表示 Flag 当前值
  - `FLAGS_no##name`：静态全局变量，表示 Flag 默认值
  - `FLAGS_nono##name`：静态常量，只用来给 `FLAGS_##name` 和 `FLAGS_no##name` 赋值

  > 这里 gflags 的解释是：当 `value` 是一个编译时常量时，`FLAGS_nono##name` 能够在编译时确定，这样能够确保 `FLAGS_##name` 进行静态初始化（程序启动前），而不是动态初始化（程序启动后，但在 `main` 函数之前）；另外变量名称有含有 `no` 是为了避免同时定义 `name` 和 `no##name` 标志，因为 gflags 支持在命令行使用 `--no<name>` 设置 `FLAGS_name`  为 `false`
  >
  > PS：我觉得这里有点问题：只要 `value` 是编译时常量，使用 `value` 赋值同样能够确保 `FLAGS_##name` 在静态初始化阶段进行初始化，而且只有 `FLAGS_##name` 和 `FLAGS_no##name` 就可以避免同时定义 `name` 和 `no##name` 标志了，所以完全不需要一个额外的 `FLAGS_nono##name`
  >
  > ref: https://en.cppreference.com/w/cpp/language/initialization
  >
  > 另外 `DEFINE_string` 进行了额外的实现，gflags 的解释是 std::string 不是 POD (Plain Old Data) 类型，只能进行动态初始化而不能进行静态初始化，为了尽量避免在这种情况下出现崩溃，gflags 先用 char buffer 来存放字符串，使其能够进行静态初始化，后续再使用 placement-new 构建 std::string
  >
  > PS：这里有点疑惑：都是在程序启动之前进行初始化，为什么动态初始化可能会出问题，难不成可能在动态初始化之前就需要访问 Flag 吗？
  >
  > 感觉 gflags 关于初始化的这部分有些过度设计了，或者是因为这部分代码看记录是十几年前写的，那时候还没出 C++11.

- `FlagRegisterer`：`DEFINE_VARIABLE` 最后会构造一个 `FlagRegisterer` 对象，`FlagRegisterer` 的构造函数的具体实现是在 Flag 注册表中注册输入的 Flag

- `FlagValue`：存放标志数据指针和类型，以及一些相关操作，比较重要的是 `ParseFrom`，将字符串 value 转化为对应 type 的 value

- `CommandLineFlag`：存放一个命令行标志的所有信息，包括 name, description, default_value 和  current_value，其中 value 用 `FlagValue` 表示

- `FlagRegistry`：Flag 注册表，用于管理所有通过 `DEFINE_<type>` 定义的 Flag

  关键成员变量：

  - `flags`：key 为 name，value 为 flag 的查找表
  - `flags_by_ptr_`：key 为数据指针（即 `&FLAGS_##name`），value 为 flag 的查找表
  - `global_registry_`：注册表全局单例

  关键成员函数：

  - `void RegisterFlag(CommandLineFlag* flag)`：注册 flag
  - `CommandLineFlag* FindFlagLocked(const char* name)`：通过 name 查找 flag
  - `CommandLineFlag* FindFlagViaPtrLocked(const void* flag_ptr)`：通过数据指针查找 flag
  - `bool FlagRegistry::SetFlagLocked(CommandLineFlag* flag,const char* value)`：设置输入 flag 的 value

  静态函数：`static FlagRegistry* GlobalRegistry()`：获取注册表全局单例

- `ParseCommandLineFlags(int* argc, char*** argv, ...)`：命令行标志解析函数，具体功能是对命令行运行程序时输入的标志进行解析并更新 Flag 的值，解析的逻辑主要通过 `CommandLineFlagParser` 类实现

- `CommandLineFlagParser`：命令行标志解析实现类，关键就是实现的几个函数：

  - `ParseNewCommandLineFlags(int* argc, char*** argv, ...)`：命令行标志解析实现，具体就是从命令行输入中提取标志的 name 和 value，再调用 `FlagRegistry` 设置 value
  - `ProcessFlagfileLocked(const string& flagval, ...)`：如果命令行中存在 `--flagfile <file_path>` 或者再调用 `ParseCommandLineFlags` 之前设置了 `FLAGS_flagfile` 的值，那么就可以从提供的 flagfile 中读取一系列 flag
  - `ProcessFromenvLocked(const string& flagval, ...)`：同 `flagfile`，如果设置了 `--fromenv`，`--tryfromenv` 或者 `FLAGS_fromenv`，`FLAGS_tryfromenv`（value 为以 `,` 分割的环境变量），那么就可以将环境变量的值赋给对应的 Flag
  - `ProcessSingleOptionLocked(CommandLineFlag* flag, const char* value, ...)`：解析完参数后调用该函数进行设置，具体实现是调用 `GlobalRegistry()->SetFlagLocked(flag, value)` 更新 flag，但是如果 flag name 为 `flagfile`, `fromenv`, `tryfromenv` 时，会调用 `ProcessFlagfileLocked` 或者 `ProcessFromenvLocked`
  - `ProcessOptionsFromStringLocked(const string& content, ...)`：`ProcessFlagfileLocked` 的下层实现，输入 `content` 是文件的内容，具体实现是一行行读取并解析 Flag

### Pytorch

ref: https://github.com/pytorch/pytorch

Pytorch 可以选择是否使用基于 `gflags` 库实现的 `Flags` 工具，具体实现方式是设置了一个编译选项以及对应的宏，默认不适用 gflags：

``` cmake
option(USE_GFLAGS "Use GFLAGS" OFF)
set(C10_USE_GFLAGS ${USE_GFLAGS})
```

实现文件：

- `c10/util/Flags.h`：定义 flags 接口
- `c10/util/flags_use_gflags.cpp`：使用 gflags 第三方库实现接口（简单的封装）
- `c10/util/flags_use_no_gflags.cpp`：不使用 gflags 的实现版本

具体实现：

![image-20230725115154789](https://github.com/huangjiyi/community/assets/43315610/18cfe5bc-d50d-404a-85be-e4783b818ea2)

- `C10_DEFINE_<type>`：用于定义特定类型的标志，统一调用 `C10_DEFINE_typed_var` 宏

- `C10_DEFINE_typed_var`：最关键的一个宏，用于定义和注册 Flag

  ``` C++
  #define C10_DEFINE_typed_var(type, name, default_value, help_str)       \
    C10_EXPORT type FLAGS_##name = default_value;                         \
    namespace c10 {                                                       \
    namespace {                                                           \
    class C10FlagParser_##name : public C10FlagParser {                   \
     public:                                                              \
      explicit C10FlagParser_##name(const std::string& content) {         \
        success_ = C10FlagParser::Parse<type>(content, &FLAGS_##name);    \
      }                                                                   \
    };                                                                    \
    }                                                                     \
    RegistererC10FlagsRegistry g_C10FlagsRegistry_##name(                 \
        #name,                                                            \
        C10FlagsRegistry(),                                               \
        RegistererC10FlagsRegistry::DefaultCreator<C10FlagParser_##name>, \
        "(" #type ", default " #default_value ") " help_str);             \
    }
  ```

  - 首先定义全局变量 `FLAGS_##name`，用于存放 Flag 的值，也用于 Flag 的访问

  - 然后定义了一个 `C10FlagParser_##name` 类，其构造函数会调用 `C10FlagParser::Parse`，这个函数的功能是将输入的 `content` 字符串解析成对应 type 的值，然后赋值给 `FLAGS_##name`

  - 最后构造了一个 `RegistererC10FlagsRegistry` 类型的注册器对象 `g_C10FlagsRegistry_##name`，这个注册器对象的构造过程就是在注册表 `C10FlagsRegistry()` 中注册一个 `(key, creater)` 项，其中 `key` 为 `#name`，`creater` 是 ` RegistererC10FlagsRegistry::DefaultCreator<C10FlagParser_##name>` 函数，`creater` 具体就是构造一个 `C10FlagParser_##name` 对象，相当于给 `FLAGS_##nam` 赋值

  - `C10FlagsRegistry()`：用于获取 Flag 注册表单例，通过通用注册表 `c10::Registry` 构造得到，该注册表中每一项是一个 `(key, creater)` 对，其中 `key` 类型为 `std::string`，`creater` 类型为返回值为 `std::unique_ptr<C10FlagParser>`，输入为 `const string&` 的函数

    ``` C++
    C10_EXPORT ::c10::Registry<std::string, std::unique_ptr<C10FlagParser>, const string&>*
        C10FlagsRegistry() {
        static ::c10::Registry<std::string, std::unique_ptr<C10FlagParser>, const string&>*
            registry = new ::c10::
                Registry<std::string, std::unique_ptr<C10FlagParser>, const string&>();
        return registry;
      }
    ```

  - `RegistererC10FlagsRegistry`：Flag 注册器类型，由通用注册器类型 `c10::Registerer` 具体化得到，其中模板参数与 `C10FlagsRegistry` 具体化 `c10::Registry` 的模板参数对应，该注册器的功能就是你构造一个注册器对象，就会在指定的注册表中注册一个 Flag，代码见 `c10/util/Registry.h` 中的 `class Registerer`

    ``` C++
    typedef ::c10::Registerer<std::string, std::unique_ptr<C10FlagParser>, const std::string&> RegistererC10FlagsRegistry;
    ```

  - 综上，一个 Flag 的定义过程就是：定义 Flag 全局变量 (`FLAGS_##name`)，定义 Flag 赋值函数 (`C10FlagParser_##name` 的构造函数)，通过构造一个注册器对象在 Flag 注册表中注册 `key` 为 `#name`，`creater` 为 Flag 赋值函数的 `(key, creater)` 项，如果需要重新设置 `Flag_##name` 的值可以调用 key `#name` 对应的 `creater`

- `C10_DECLARE_<type>`：用于声明指定 Flag，统一调用 `C10_DECLARE_typed_var` 实现，底层就是一个 `extern` 用法：

  ``` C++
  #define C10_DECLARE_typed_var(type, name) C10_API extern type FLAGS_##name
  ```

- `ParseCommandLineFlags(int* pargc, char*** pargv)`：解析命令行参数，代码主要就是解析命令行参数的一些逻辑，这部分可以看 `c10/util/flags_use_no_gflags.cpp` 中的代码，在每个命令行参数被解析完后，会在通过 `C10FlagsRegistry()->Create(key, value)` 给注册表中对应的 Flag 赋值。

### 对比分析

**gflags** 

- 优点：提供的功能很多，同时各方面都考虑的很完善

- 缺点：很多功能 Paddle 不太需要，并且一些代码实现有些过度设计的感觉，整体代码比较复杂

**pytorch**

- 优点：整体实现比较简洁，方便理解，同时设计比较巧妙：pytorch 没有设计 Flag 数据结构，只针对每个 Flag 设计了对应的赋值函数，然后在注册表中只存放 name, help_string, 赋值函数

- 缺点：只实现了最主要的功能，并且没有设计 Flag 数据结构，Flag 注册表也是 c10 通用注册表的一个具体化示例，不方便扩展

## 四、设计思路与实现方案

### 1. 设计思路

针对[Paddle 中现有的 gflags 用法](#paddle-中现有的-gflags-用法)，对这些用法如何实现进行的分析如下：

- `DEFINE_<type>` 和 `DECLARE_<type>`

  如果只需要这两种用法的话，可以实现的非常简单：在 `DEFINE_<type>` 宏中定义一个全局变量 `FLAGS_##name`，在 `DECLARE_<type>` 宏中用 `extern` 声明这个全局变量

  这样的话只有当我们同时知道一个 Flag 的 name 和 type 的时候，才能用 `DECLARE_<type>` 访问 flag 的 value，但是在一些用法中只知道 Flag 的 name 而不知道 type (比如在命令行参数解析中)，这种情况下无法使用 `DECLARE_<type>` 访问

  因此**需要设计一个 Flag 注册表**，能够通过 name 查找到一个 flag 的 value (void* 类型数据指针) 和 type，这样在 `DEFINE_<type>` 宏定义完全局变量 `FLAGS_##name` 后，还需将 `(name, &FLAGS_##name, type)` 注册到注册表中。

  另外一个 Flag 的信息不仅包括 name, value, type，还有 description_string，file 等信息，因此**需要设计一个 Flag 数据结构**，包含一个 Flag 的完整信息，然后在注册表中可以通过 name 查找到一个完整的 Flag

- `ParseCommandLineFlags`：这部分主要是写一些标志解析逻辑，标志解析后需要调用注册表设置 value

  需要支持的功能：

  - 普通命令行标志的解析，一般格式为 `--name=value` 或 `--name value`，需要确定支持哪些格式，主要参考 gflags
  - 特殊标志：`--fromenv` 和 `--tryfromenv`，根据环境变量的值设置 Flag，Paddle 中有用到
  - 考虑是否支持其他的 gflags 特殊标志，比如 `--flagfile`，从一个文件中解析 Flag，Paddle 代码中没用到，不确定外部是否会用到
  - 报错机制：对于不满足目标格式的 Flag 或者解析得到未定义的 Flag 的报错机制，Paddle 中用到的 `AllowCommandLineReparsing()` 与这个机制相关

- `GetCommandLineOption`， `SetCommandLineOption`，`AllowCommandLineReparsing`：在 Flag 注册表中设计对应功能的接口即可

### 2. 实现方案

![image](https://github.com/huangjiyi/community/assets/43315610/87832764-c4f2-4620-b2c2-94f7d3c5b9a4)

下面从底层数据结构开始介绍

#### `Flag`: Flag 数据结构

``` C++
enum class FlagType : uint8_t {
  BOOL = 0,
  INT32 = 1,
  UINT32 = 2,
  INT64 = 3,
  UINT64 = 4,
  DOUBLE = 5,
  STRING = 6,
  UNDEFINED = 7,
};

class Flag {
 public:
  Flag(std::string name,
       std::string description,
       std::string file,
       FlagType type,
       const void* default_value,
       void* value)
      : name_(name),
        description_(description),
        file_(file),
        type_(type),
        default_value_(default_value),
        value_(value) {}
  ~Flag() = default;

  // Summary: --name_: type_, description_ (default: default_value_)
  std::string Summary() const;

  void SetValueFromString(const std::string& value);

 private:
  friend class FlagRegistry;

  const std::string name_;	   // flag name
  const std::string description_;  // description message
  const std::string file_;	   // file name where the flag is defined
  const FlagType type_;		   // flag value type
  const void* default_value_;	   // flag default value ptr
  void* value_;			   // flag current value ptr
};
```

- `FlagType` 表示 Flag 数据类型
- `Flag` 包含一个 Flag 的全部信息，主要参考了 gflags，相当于 gflags 中的 `CommandLineFlag` + `FlagValue`
- `SetValueFromString`：将输入的 `value` 字符串转化为目标 `type_` 的数值赋给 `value_`，在这个函数中需要检查 `value` 是否满足目标 `type_` 的格式
- `Summary`：对一个 Flag 的信息进行总结，用于打印帮助信息

#### `FlagRegistry`: Flag 注册表

``` C++
class FlagRegistry {
 public:
  static FlagRegistry* Instance() {
    static FlagRegistry* global_registry_ = new FlagRegistry();
    return global_registry_;
  }

  void RegisterFlag(Flag* flag);

  bool SetFlagValue(const std::string& name, const std::string& value);

  bool HasFlag(const std::string& name) const;

  void PrintAllFlagHelp(std::ostream& os) const;

 private:
  FlagRegistry() = default;

  std::map<std::string, Flag*> flags_;

  struct FlagCompare {
    bool operator()(const Flag* flag1, const Flag* flag2) const {
      return flag1->name_ < flag2->name_;
    }
  };

  std::map<std::string, std::set<Flag*, FlagCompare>> flags_by_file_;

  std::mutex mutex_;
};
```

- `FlagRegistry` 为 Flag 注册表类，用于管理所有定义的 Flag
- 只有一个全局单例，外部只能通过 `FlagRegistry::Instance()`  获取
- 主要数据：
  - `std::map<std::string, Flag*> flags_`：name 到 Flag 指针的查找表
  - `std::map<std::string, std::set<Flag*, FlagCompare>> flags_by_file_`：根据定义所在文件区分不同的 Flag，`key` 是文件名，`value` 是定义在该文件中的 Flag 指针集合（根据 flag name 排序），主要用于在打印所以 flag 是按定义文件进行输出。
  - `std::mutex mutex_`：互斥锁，在修改 `flags_` 前 lock
- 主要方法包括：
  - `RegisterFlag`：注册 Flag
  - `SetFlagValue`：将 `value` string 表示的值赋给 `flags_[name]->value_`，
  - `HasFlag`：查找 Flag 是否存在
  - `PrintAllFlagHelp`：打印所有 Flag 的帮助信息

#### `FlagRegisterer`: Flag 注册器

``` C++
class FlagRegisterer {
public:
  template <typename T>
  FlagRegisterer(std::string name,
                 std::string description,
                 std::string file,
                 const T* default_value,
                 T* value);
};

template <typename T>
struct FlagTypeTraits {
  static constexpr FlagType Type = FlagType::UNDEFINED;
};

#define DEFINE_FLAG_TYPE_TRAITS(type, flag_type) \
  template <>                                    \
  struct FlagTypeTraits<type> {                  \
    static constexpr FlagType Type = flag_type;  \
  }

DEFINE_FLAG_TYPE_TRAITS(bool, FlagType::BOOL);
DEFINE_FLAG_TYPE_TRAITS(int32_t, FlagType::INT32);
DEFINE_FLAG_TYPE_TRAITS(uint32_t, FlagType::UINT32);
DEFINE_FLAG_TYPE_TRAITS(int64_t, FlagType::INT64);
DEFINE_FLAG_TYPE_TRAITS(uint64_t, FlagType::UINT64);
DEFINE_FLAG_TYPE_TRAITS(double, FlagType::DOUBLE);
DEFINE_FLAG_TYPE_TRAITS(std::string, FlagType::STRING);

#undef DEFINE_FLAG_TYPE_TRAITS

template <typename T>
FlagRegisterer::FlagRegisterer(std::string name,
                               std::string help,
                               std::string file,
                               const T* default_value,
                               T* value) {
  FlagType type = FlagTypeTraits<T>::Type;
  Flag* flag = new Flag(name, help, file, type, default_value, value);
  FlagRegistry::Instance()->RegisterFlag(flag);
}
```

- `FlagRegisterer` 作为注册器，利用模板函数和结构体统一实现不同 type 的 flag 注册过程，在构造一个 `FlagRegisterer` 对象时，会根据构造输入在 Flag 注册表中进行注册。
- 其中设计了一个 `FlagTypeTraits` 利用模板实现内置数据类型到枚举类型 `FlagType` (`Flag` 数据结构中保存的类型) 的映射

#### `PD_DEFINE_<type>`: Flag 定义宏

``` C++
#define PD_DEFINE_VARIABLE(type, name, default_value, description)           \
  namespace paddle {                                                         \
  namespace flags {                                                          \
  static const type FLAGS_##name##_default = default_value;                  \
  type FLAGS_##name = default_value;                                         \
  /* Register FLAG */                                                        \
  static ::paddle::flags::FlagRegisterer flag_##name##_registerer(           \
      #name, description, __FILE__, &FLAGS_##name##_default, &FLAGS_##name); \
  }                                                                          \
  }                                                                          \
  using paddle::flags::FLAGS_##name

#define PD_DEFINE_bool(name, val, txt) \
  PD_DEFINE_VARIABLE(bool, name, val, txt)
#define PD_DEFINE_int32(name, val, txt) \
  PD_DEFINE_VARIABLE(int32_t, name, val, txt)
#define PD_DEFINE_uint32(name, val, txt) \
  PD_DEFINE_VARIABLE(uint32_t, name, val, txt)
#define PD_DEFINE_int64(name, val, txt) \
  PD_DEFINE_VARIABLE(int64_t, name, val, txt)
#define PD_DEFINE_uint64(name, val, txt) \
  PD_DEFINE_VARIABLE(uint64_t, name, val, txt)
#define PD_DEFINE_double(name, val, txt) \
  PD_DEFINE_VARIABLE(double, name, val, txt)
#define PD_DEFINE_string(name, val, txt) \
  PD_DEFINE_VARIABLE(string, name, val, txt)
```

- `PD_DEFINE_VARIABLE`：统一实现不同 type 的 Flag 定义和注册过程
- 全局变量 `FLAGS_##name` 放在了特殊的 `phi::flag##type` 命名空间中，然后通过 using 用法暴露出来

#### `PD_DECLARE_<type>`: Flag 声明宏

``` C++
#define PD_DECLARE_VARIABLE(type, name) \
  namespace paddle {                    \
  namespace flags {                     \
  extern type FLAGS_##name;             \
  }                                     \
  }                                     \
  using paddle::flags::FLAGS_##name

#define PD_DECLARE_bool(name) PD_DECLARE_VARIABLE(bool, name)
#define PD_DECLARE_int32(name) PD_DECLARE_VARIABLE(int32_t, name)
#define PD_DECLARE_uint32(name) PD_DECLARE_VARIABLE(uint32_t, name)
#define PD_DECLARE_int64(name) PD_DECLARE_VARIABLE(int64_t, name)
#define PD_DECLARE_uint64(name) PD_DECLARE_VARIABLE(uint64_t, name)
#define PD_DECLARE_double(name) PD_DECLARE_VARIABLE(double, name)
#define PD_DECLARE_string(name) PD_DECLARE_VARIABLE(string, name)
```

- `PD_DECLARE_VARIABLE`：统一实现不同 type Flag 的声明，具体实现就是简单的 `extern` 用法

#### `ParseCommandLineFlags`

实现命令行参数解析，`*pargc` 为参数数量，`*pargv` 为参数字符串数组，相邻的字符串在完整的命令中用空格分隔，其中第一个是运行的程序，大致解析逻辑如下：

``` C++
void SetFlagsFromEnv(const std::vector<std::string>& envs) {
    for (const std::string &env_var_name : envs) {
        // 获取环境变量 env 的值, 计划实现一个函数 GetValueFromEnv
        std::string value = GetValueFromEnv(env_var_name);
        FlagRegistry::Instance()->SetFlagValue(env_var_name, value);
    }
}

void ParseCommandLineFlags(int* pargc, char*** pargv) {
    // 1. 对 pargc, pargc 进行预处理，移除第一个程序名称
    size_t argv_num = *pargc - 1;
    std::vector<std::string> argvs(*pargv + 1, *pargv + *pargc);
    
    FlagRegistry* const flag_registry = FlagRegistry::Instance();
    // 2. 遍历每一个 argv, 解析得到每个 flag 的 name 和 value
  	for (size_t i = 0; i < argv_num; i++) {
        const std::string& argv = argvs[i];
        
        // 检查 argv 格式
        // ...
        
        // 处理特殊标志 --help
        if (argv == "--help" or argv == "-help") {
            // 打印帮助信息
            // ...
            exit(1);
        }
        
	    string name, value;
        // 解析 name 和 value
        // ...
        
        // 处理特殊标志 --fromenv 和 --tryfromenv
        if (name == "fromenv" || name == "tryfromenv") {
            std::vector<std::string> envs;
            // 解析需要设置的环境变量
            // ...
            SetFlagsFromEnv(envs);
            continue;
        }
        
        flag_registry->SetFlagValue(name, value);
  	}
}
```

- 在参数格式检查中，命令行参数的格式应该满足：`--help`, `--name=value`, `--name value`，其中双横线 `--` 可以换成单横线 `-`，`value` 可以放在 `""` 中，放在 `""` 中的 `value` 可以包含空格，否则 `value` 不能包含空格

- 这里说明一下不打算支持的 gflags 中的参数格式：

  - `--name` 或 `--noname` 用于 bool flag 赋值 true 或 false
  - 单独的 `--` 表示终止解析命令行参数

- 对于特殊标志：

  计划支持：

  - `--help`：

    与 `gflags` 保持一致，根据定义所在文件的顺序打印所有 Flag 的帮助信息，包括 name, default_value, description string，考虑到 Paddle 中定义了 300+ Flag，直接在命令行中打印出来不方便查询，因此另外设计了一个可以将打印的帮助信息输出到文件中的接口

  - `--fromen=value` 和 `--tryfromenv=value`：`value` 为用 `,` 分隔的环境变量名 `env1,env2,...`，实现的效果是将环境变量 `name` 的值赋给 `FLAGS_##name`，其中 `--tryfromenv` 对于没有定义的环境变量会忽略不会报错，`--fromenv` 则会报错

  计划不支持的 `gflags` 特殊标志：

  - 其他过滤规则打印 Flag 信息的 `--helpxxx` 标志
  - `--undefok=flagname,flagname,...`：允许列出的 Flag 没有定义而不会报错
  - `--flagfile=filepath`：从指定文件中读取 Flag，flagfile 中每一行一个 Flag

#### 报错机制

在代码中还需要设计一套报错机制，计划利用 Paddle 中的报错机制实现，报错主要包括以下几种情况：

- 针对 `ParseCommandLineFlags` 不符合目标格式参数的报错
- 针对要设置的 Flag 并没有定义（注册）的报错，这类报错可以设置一个开关函数
- 针对 `SetFlagsFromEnv` 中 `env_var_name` 在环境中不存在的报错
- 针对在 Flag 注册表中注册相同 name 的 Flag 的报错
- 针对 `value` 字符串不满足目标 type 格式的报错

在其中几种批量处理的情况中，可以先收集每一项的错误信息再统一报错

#### gflags 依赖可选

早期实现的版本会保留目前依赖 gflags 的版本，具体参考 [Pytorch](#Pytorch) 利用编译选项和宏来控制，如果新实现的版本与旧版本接口不同，会通过再封装一层来统一新旧版本的接口。

由于新实现的 Flag 注册定义宏为 `PD_(DEFINE|DECLARE)_<type>`，为了实现能够切换新旧版本，旧版本的 `(PHI_)?(DEFINE|DECLARE)_<type>` 需要全部替换为 `PD_(DEFINE|DECLARE)_<type>`，包括接口的定义和用法

### 3. 主要影响的模块接口变化

- 需要将所有的 `(PHI_)?(DEFINE|DECLARE)_<type>` 替换为 `PD_(DEFINE|DECLARE)_<type>`
- 其余的 gflags 用法（较少）与新实现的接口不同也需要替换
- cmake 中依赖关系的变化

## 五、测试与验收的考量

### 自测方案

- 因为 Paddle 里有非常多 flags 的用法，这些用法都是对于 flags 工具的测试，所以看最终 CI 有没有问题基本就行了，不需要额外写单测
- 需要分别测试新旧版本 flags 工具的 CI 通过情况

## 六、影响面

### 对用户的影响

无影响

### 对二次开发用户的影响

主要是一些接口变化的影响

### 对框架架构的影响

无影响

### 对性能的影响

无影响

## 七、排期规划

1. 8 月 15 日前完善设计文档，期间对于已经确定的部分进行开发
2. 8 月 31 日前基本完成开发，根据 Review 意见进行修改
3. 9 月 15 日前完成主要 PR 合入，后续根据反馈的问题进行修复
