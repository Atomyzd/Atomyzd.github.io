--- 
layout: page
title: About Me
description: 软件工程专业研究生
keywords: Atomyzd
comments: true
menu: 关于
permalink: /about/
---

软件工程专业博士在读。

希望每天都能有所收获。

记录知识与生活。

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
