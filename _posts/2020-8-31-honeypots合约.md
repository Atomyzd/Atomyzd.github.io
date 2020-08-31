---
layout: post
title: Honeypots合约总结
categories: Solidity
description: Solidity安全漏洞分析
keywords: Solidity，安全漏洞
---

## Solidity 蜜罐合约

### 定义

  蜜罐是一种智能合约，它可以假装将它的资金泄露给任意用户（受害者），前提是该用户向它发送了额外的资金。但是，用户提供的资金将会被捕获，蜜罐创建者（攻击者）将会获得这些资金。

### 蜜罐的三个阶段

1.	攻击者部署一个看起来有漏洞的合约，并以资金作为诱饵。

2.	受害者通过转最少数目的资金来利用合约，但是失败了。

3.	攻击者将诱饵以及受害者失去的资金一起撤回。

  实际上，攻击者与常规以太坊用户拥有相同的能力，他只需要资金来部署智能合约并且设置一个诱饵。


### 蜜罐的分类与技术

1.	以太坊虚拟机级别

2.	Solidity编译器级别

3.	Etherscan区块链浏览器级别

   具体情况如下图：
   
   ![level](/images/posts/Solidity/honneypots/level.png)

### 蜜罐分类的具体介绍

- **以太坊虚拟机级别的蜜罐**

   - **Balance Disorder**

   ![bdl](/images/posts/Solidity/honneypots/Balance-Disorder.png)
   
  从表面上看，只要调用者调用该函数时转入的资金大于或等于智能合约账本里的资金，转入的资金和智能合约账本里的资金都会被转到任意的地址。
  
  但是，调用者只要一调用该函数，账本里的资金就增加了转入的资金的量，因此条件判断根本不成立，该条件成立的唯一情况是，智能合约账户里的资金为0。

- **Solidity编译器级别的蜜罐**

   - **Inheritance Disorder**

   ![idl](/images/posts/Solidity/honneypots/Inheritance-Disorder.png)
   
  KingOfTheHill继承了Ownerable,并且，takeAll函数使用了onlyOwner修饰符，规定了owner变量才能够获取合约账户中的资金。

  调用者会发现，使用回调函数可以更改owner变量的值，只要在调用合约时转入的资金大于jackpot里的数值，当调用者调用回调函数时，交易成功，
  
  但是，当调用者调用takeAll函数时，交易失败，这是因为Solidity编译器将第2行的owner和第9行的owner当作两个不同的变量看待，更改第9行的owner不会影响到第2行里owner的地址。

   - **Skip Empty String Literal**
  
   ![se](/images/posts/Solidity/honneypots/Skip-Empty.png)
  
  在这个合约中，存在的一个显著的缺陷是，investor可以通过divest函数撤回任意数量的投资。通过第14行，investor希望执行loggedTransfer函数转资金给自己。
  
  但是，在 Solidity 0.4.12 之前，存在一个bug，如果空字符串””用作函数调用的参数，则编译器会跳过它。由于编译器的bug,最后会执行this.loggedTransfer(amount, msg.sender,owner);这条语句，因此，资金实际上是转给了合约所有者。
  
   - **Type Deduction Overflow**
  
    ![td](/images/posts/Solidity/honneypots/type-deduction.png)
  
  这个合约吸引调用者的点在于，当调用者调用Test函数并向合约转大于0.1ether的资金时，通过执行第7行到第13行代码，调用者会得到4*msg.value-2的转账。 
  
  但是，调用者没有注意到，使用var会造成类型推导溢出的问题。msg.value的单位是wei，0.1ether是100000000000000000wei,当使用var时，编译器推导出i的类型是uint8,当i=255时，i++则会造成整数溢出，i=0,此时，multi=0,执行第10行代码跳出循环。此时，amountToTransfer=510,即调用合约转进0.1ether，最终只得到510wei的资金，损失巨大。
  
   - **Uninitialised Struct**

    ![US](/images/posts/Solidity/honneypots/Unintialised-Struct.png)
  
  从表面上看，这个合约生成一个随机数，让调用者调用guessNumber函数猜这个随机数，调用者需要先转一定的资金下赌注，再将猜的数用参数传给这个函数，如果猜中了可以得到智能合约账户中的余额。但是，存储在区块链上的每个数据都是公开可用的，因此任何用户都可以轻松获得随机数的值（例如通过Etherscan查看合约的地址）。用户会天真的认为，是合约的所有者犯了一个低级错误。
  
  当用户调用guessBumber函数并转入赌注后，合约看起来会使用一个guessHistory结构体数据跟踪当前用户的状态，但是，在第11行，合约并没有对结构体进行初始化，这样就会造成未初始化存储指针的问题。struct在局部变量中默认是存放在storage中的，因此可以利用 Unintialised Storage Pointers的问题，此时，guessHistory会被当成一个指针，guessHistory.player指向randomNumber，guessHistory.number指向lastplayed。第12行会用调用者的地址重写randomNumber，从而使14行的条件不成立。
  
  此外，合约创建者也清楚有些调用者会看出调用者的地址会覆盖随机数，因此，将随机数限制在10以内（第10行）。

- **Etherscan区块链浏览器级别的蜜罐**

   - **Hidden State Update**
   
   ![hs](/images/posts/Solidity/honneypots/hidden-state.png)
   
  除了正常的事务外，Etherscan还显示所谓的内部消息（internal messages），这是来自其他合约而不是用户帐户的事务。但是出于可用性的考虑，Etherscan不显示包含空事务值的内部消息。上图中的合约是蜜罐技术的一个例子，表示为隐藏状态更新。在这个例子中余额被转移给那些能够猜出用于计算存储散列正确值的人。天真的用户将假定passHasBeenSet设置为false，并尝试调用不受保护的SetPass函数，该函数允许用已知值重写哈希，前提是至少有一个以太币转到合约。
  
  当分析Etherscan上的内部消息时，用户将找不到调用PassHasBeenSet函数的任何证据，因此假设passHasBeenSet被设置为false。但是蜜罐创建者可能会误导Etherscan执行的筛选，以便从另一个合约调用函数PassHasBeenSet并使用空事务值。因此，只查看EtherScan上显示的内部消息，不知情的用户就会认为变量设置为false，并自信地将以太币传输到setpass函数。
  
   - **Hidden Transfer**
  
   ![ht](/images/posts/Solidity/honneypots/hidden-transfer.png)
  
  Etherscan通过一个web界面来展示经过验证的智能合约源代码，在显示界面中，较长的代码行只能显示到一定的宽度，其余的代码被隐藏，只能够通过水平滚动看到隐藏的代码。
  
  上图的合约利用了这一特征，通过在第4行代码后引入一连串的空格有效的隐藏了后面的代码。用户只能看到展示的第4和第5行，认为转到智能合约账户的资金总和大于0.5ether即可转走智能合约账户余额的钱。而隐藏的代码限定了调用者必须是合约拥有者，否则会报错。判定条件是块号必须大于5040270，即这段代码只在主网上生效。当用户在测试网上调用合约时，由于测试网的块没有主网那么多，用户可以转走合约账户余额，从而让用户相信这个合约不是蜜罐。
  
   - **Straw Man Contract**
  
   ![smc](/images/posts/Solidity/honneypots/Straw-man.png)
  
  在上图中提供了一个蜜罐技术示例，将其表示为稻草人合约。乍一看合约第14行的现金提取功能似乎容易受到重入攻击（reentrancy attack）。为了能够使用重入攻击，用户必须首先调用Deposit函数并转移最小数量的以太币。最后用户调用CashOutT函数，该函数执行对TransferLog中存储的合约地址的调用。如图所示名为log的合约应该充当记录器logger。然而蜜罐创建者并没有用包含所示记录器合约字节码的地址初始化合约。相反它是用另一个地址初始化的，该地址指向一个实现相同接口的合约。如果函数AddMessage是用字符串“CashOut”调用的且调用者不是蜜罐创建者，则抛出一个异常。
  
  因此，用户执行的重入攻击总是失败的。另一种选择是在转移余额之前使用委托代理（delegatecall）。委托代理允许被调用方合约修改调用方合约的堆栈。因此攻击者只需将堆栈中包含的用户地址与自己的地址交换，当从委托代理返回时，余额将转移到攻击者而不是用户的地址。





   





