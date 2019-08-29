---
layout: archive
permalink: /data-science-projects/
title: "Data Science Projects by Tags"
author_profile: true
header:
	image:  "/image/data.jpg"
---

{% include base_path %}
{% include group-by-array collection=site.posts-tag.html %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
	