---
layout: page
title: About Me
description: 知行合一
keywords: Atomyzd
comments: true
menu: 关于
permalink: /about/
---

江苏海事职业技术学院在职教师

爱好：电子产品，装机，深度学习，动漫等

记录知识与生活

## 教育背景

|  时间   | 学校专业 |  学位/身份 |
|  ----  | ----  |  ----  |
| 2014-2018  | 南通大学软件工程 |  学士  |
| 2018-2020  | 南通大学计算机技术 |  硕士  |
| 2019-2020  | 澳大利亚莫纳什大学 |  访问学生  |

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

