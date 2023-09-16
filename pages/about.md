--- 
layout: page
title: About Me
description: 知行合一
keywords: Atomyzd
comments: true
menu: 关于
permalink: /about/
---

江苏海事职业技术学院在职教师。

爱好：电子产品，装机，深度学习，动漫等

记录知识与生活。

## 教育背景

|  时间   | 学校专业 |  学位/身份 |
|  ----  | ----  |  ----  |
| 2014-2018  | 南通大学软件工程 |  学士  |
| 2018-2020  | 南通大学计算机技术 |  硕士  |
| 2019-2020  | 澳大利亚莫纳什大学 |  访问学生  |

## 工作经历

|  时间   | 单位 |  角色 |  职责  |
|  ----  | ----  |  ----  |  ----  |
| 2021-2022  | 南通理工学院 |  计算机专任教师  |  C语言和数据结构教学  |
| 2022-至今  | 江苏海事职业技术学院 |  人工智能专任教师  |  数据分析、机器学习和深度学习教学  |

## 技能证书

+ CET 6
+ CET 4
+ 中级软件设计师证书

### 奖项荣誉

+ 2020年 南通大学优秀硕士毕业论文
+ 2019年 南通大学研究生国家奖学金
+ 2019年 南通大学优秀研究生
+ 2019年 江苏省研究生数学建模科研创新实践大赛二等奖
+ 2017年 第十届中国大学生计算机设计大赛一等奖
+ 2017年 江苏省计算机设计大赛一等奖
+ 2017年 国家励志奖学金

## 常用链接

<ul>
{% for link in site.data.links %}
<li><a href="{{ link.url }}" target="_blank">{{ link.name}}</a></li>
{% endfor %}
</ul>


## Interest and Direction

{% for skill in site.data.skills %}
### {{ skill.name }}
<div class="btn-inline">
{% for keyword in skill.keywords %}
<button class="btn btn-outline" type="button">{{ keyword }}</button>
{% endfor %}
</div>
{% endfor %}

