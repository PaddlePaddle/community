# sourcetree git 使用教程

|   |  |
| --- | --- |
|提交作者 | gouzi | 
|提交时间 | 2022-11-15 | 
|版本号 | V1.0 | 
|依赖飞桨版本 | 不依赖于paddle的任何版本 | 
|文件名 | sourcetree.md | 

# 软件安装

1. sourcetree软件安装, 参考[官方文档](https://www.sourcetreeapp.com/)
2. 安装代码格式化工具 ( 需提前安装python3 )
```bash
python3 -m pip install pre-commit==2.17.0
```

# 配置Github账户

首先, 需要有一个 Github 账户.

然后, 我们把 Github 账户添加到 sourcetree 中

1. 打开设置
2. 点击账户

![](https://ai-studio-static-online.cdn.bcebos.com/4f12f9422e0f4211a302cc5988c2c9ebdf4dc5e5e1f34c40b3bdee69d350bec1)

3. 添加账户

![](https://ai-studio-static-online.cdn.bcebos.com/474348cec3d04dbfbb27277bb10734665fd3eee189924680bd327bef51a7b4bf)

# fork项目

由于我们没有主仓库的 push 权限, 所以我们 fork 到自己的仓库进行修改

![](https://ai-studio-static-online.cdn.bcebos.com/efd3408c233649708818376da4d058badd5b4743367143c1baeb0b48204c009c)

这个时候, 你就拥有了一个对应的下载地址, 比如你的昵称是Example, 你的 PaddlePaddle/paddle 仓库链接就是 `https://github.com/Example/paddle.git`

# 克隆项目
## 方法一:

这里使用常规的 URL 方式进行克隆项目

这里的 URL 填写的就是刚刚的地址

对应的 git 命令: `git clone`

1. clone 克隆你的仓库地址

![](https://ai-studio-static-online.cdn.bcebos.com/2cc21d2ffac845629362d15e09b8dd98cce06bdbcbe749c79f6d14105fb2ec59)

![](https://ai-studio-static-online.cdn.bcebos.com/08bbe1bd4605409c8ef3e31192bbae5a2d71c639b45c47bdb1949e7c6242f9fc)

![](https://ai-studio-static-online.cdn.bcebos.com/9fd4687675944ca0a60f03bbdd461b43a19e90d1cd534108baced74a25fd1947)

## 方法二：

使用 sourcetree 的远端功能进行克隆

对应的 git 命令: `git clone`

1. 点击远端
2. 点击克隆

![](https://ai-studio-static-online.cdn.bcebos.com/031458bb91ff48b79ab64c6a1aea23059ea8ff1bb3b44fed9e4ec349af110a4d)

![](https://ai-studio-static-online.cdn.bcebos.com/9fd4687675944ca0a60f03bbdd461b43a19e90d1cd534108baced74a25fd1947)

# 创建自己的分支

在自己的分支上进行修改, 方便进行多pr的提交

对应的 git 命令: `git checkout -b 新分支名称`

1. 检出分支

![](https://ai-studio-static-online.cdn.bcebos.com/5f4ac90466f2436abbb062c1a06596d1ac37372312094f4c819eb465cb8293a8)

![](https://ai-studio-static-online.cdn.bcebos.com/47bedd6d25fc4d25b2425fa85bcabb117c122cbfc0f7442ab78f6967436e9faa)

![](https://ai-studio-static-online.cdn.bcebos.com/228c75f766c5420987d05851a62ad53db60d6a78b45e4483be6bbc1063b282cc)

# 初始化 pre-commit 代码格式化工具

1. 点击终端

![](https://ai-studio-static-online.cdn.bcebos.com/2f90768682b04b03b0d191568cc5ed8d8698a1c18e9f437c828def208b824b79)

2. 终端运行```pre-commit install```

提示```pre-commit installed at .git/hooks/pre-commit```代表成功

# 修改内容

这里就修改你想修改的就可以

# 提交到自己的仓库

选择想要提交的文件, 填写描述, 点击右下角的提交

对应的 git 命令: `git add * `, `git commit -m "这是一次提交的描述"`, `git push`

1. 选择想要提交的文件并添加注释

![](https://ai-studio-static-online.cdn.bcebos.com/9cea080c8700476eac7e4775dbf50a8911db98ba09bb429684192d976dd8e695)

# 在网页端提交 Pull Request（PR）

回到github网页, 在页面中选择 Pull Request

1. 新建PR

![](https://ai-studio-static-online.cdn.bcebos.com/f93a40705abf4dc28c4f24326fe4c06561b4d5fc5d2547dbbd67958ce6a03143)

2. 选择提交的分支

![](https://ai-studio-static-online.cdn.bcebos.com/6b3b156b1a7c418ca4db3f56f7f756fc597efd7e480040d4ad7a932376d21768)

3. 添加描述

![](https://ai-studio-static-online.cdn.bcebos.com/22062d08d90440cc97f11494349152004a5ccdbbc34040b89ffe2cb7c1454a63)

4. 结束

看到这行就代表提交PR成功啦, 感谢你对开源项目的贡献!

![](https://ai-studio-static-online.cdn.bcebos.com/d746e1b2c56c4eaf96798e04cf46d84e1faeebd950e24559a03d6c385549fe59)

# 补充说明

注：更多内容请参考[贡献指南](https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/dev_guides/Overview_cn.html)