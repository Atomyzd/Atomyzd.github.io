---
layout: post
title: Solidity笔记一
categories: Blog
description: Solidity智能合约第一天笔记
keywords: Solidity,笔记
---

### Solidity源码与智能合约笔记
- Solidity源码->智能合约的步骤：
   1. 首先使用编译器将Solidity编写的源代码编译为字节码，编译过程中同时会产生智能合约的二进制接口规范（Appliciation Binary Interface，简称为ABI）
   2. 通过交易的方式将字节码部署到以太坊网络，部署成功会产生一个新的智能合约账户
   3. 使用JavaScript编写的DApp通常以web3.js+ABI调用智能合约中的函数来实现数据的读取与修改
- Solcjs
   1. 是Solidity源码库构建的目标之一，是Solidity的命令行编译器
   
   2. 使用npm可以便捷地安装Solidity编译器solcjs
   
   3. npm install -g solc (注：npm和node安装过程参考 [帖子](https://www.cnblogs.com/xbzhu/p/8886961.html) 
- Solidity源码与智能合约工作流程图
![Solidity Workflow](/images/posts/Solidity/Solidity-work.png)
- 一个简单的智能合约（存数取数合约）

```python
pragma solidity >0.4.22;

contract SimpleStorage{
    uint myData;
    function setData(uint newData) public{
        myData = newData;
    }
    function getData() public view returns(uint){
        return myData;        
    }
}
```


