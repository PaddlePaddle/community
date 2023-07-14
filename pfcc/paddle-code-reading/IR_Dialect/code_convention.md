# 【代码约定】IR代码相关约定

## 命名方式
- 函数名的每个单词首字母大写(即“驼峰变量名”或“帕斯卡变量名”)，没有下划线；但取值函数和设值函数是例外。

- 取值和设值函数可采用小写字母加下划线来命名。需要强调的是，取值函数的函数名一定是以名词为前缀的，而设值函数一定以```set_```作为前缀。（比如：```attribute(...)```和```set_attribute(...)```)。

- 对于首字母缩写的单词, 更倾向于将它们视作一个单词进行首字母大写（比如：写作```CreateDcePass()```而非```CreateDCEPass()```）。

- 在语意清晰的前提下，函数名应尽量简短。 （比如：Operation里面的成员函数名```info()```， 而非 ```op_info()```）

## 使用指南
- IR中很多组件采用了Pimpl设计模式，将具体实现对象封装在了Impl数据结构中。采用这种设计模式的数据结构包括 OpResult、Value、OpOperand、OpInfo、Type、Attribute等等，它们自身都只包含一个impl或者storage指针，可以快速浅拷贝。真正的实现对应OpResultImpl、ValueImpl、OpOperandImpl、OpInfoImpl、TypeStorage、AttributeStorage等数据结构。其中，后缀名为Impl表明该数据结构不对用户暴漏，头文件不会导出。后缀名为Storage表明该头文件会导出。对于采用Pimpl设计模式的这类数据结构，建议用户以值传递的方式进行使用。

- IR中Region、Block的概念类似于C++标准库的容器的概念。 其中，Region是Block的容器，Block是Operation的容器。 二者的实现方式类似于std::unique_ptr, 当调用push_back()、insert()等接口给容器添加添加元素时，容器会接管对应元素的生命周期。Region在析构时，会对包含的每一个Block成员，调用delete接口。Block在析构时，会对包含的每一个Operation成员，调用Operation::Destroy()接口。因此，用户需要注意：将同一个对象多次添加到容器会导致段错误。

- IR中Operation有两种创建方式：```Opertion::Create(...)```和```Builder::Build(...)```。其中，Create接口创建的对象需要通过```Operation::Destroy()```接口来回收，用户可以显式调用Destroy接口，也可以选择将其添加到某个Block中，由Block在自身析构的时候隐式调用Destroy()接口。 Build接口创建的对象会被隐式添加到Builder所关联的Block的插入点。

## 开发建议
考虑到Paddle框架是一个非常庞大的项目，而IR库又位于Paddle的最低层，由多方共用，因此，我们对IR库的开发者提出以下建议：

- 性能优先，在保证性能的基础上，兼顾代码规范性和便捷性。

- 尽量用前置声明替代头文件包含，必要时应进行头文件拆分，方便加快编译速度。
    - 以定义Attribtue为例，Attribue成员变量包含了一个AttribtueStorage指针。项目的很多地方都会使用Attribute类型，但不会直接使用AttribtueStorage类型。因此，我们在Attribute头文件中，以前置声明的形式避免了对AttribtueStorage头文件的包含。

- 真实行数超过5行的函数不建议内联。这儿强调真实行数，是因为IR中应用了大量的类似IR_ENFORCE这种宏定义，它的真实行数并不是表面看上去的只有一行。