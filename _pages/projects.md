---
title: "Data Science Projects"
layout: categories
permalink: /projects/
author_profile: true
header:
  image: "/images/data.jpg"
---

{% for category in site.categories %}
  <h3>{{ category[0] }}</h3>
  <ul>
    {% for post in category[1] %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
{% endfor %}