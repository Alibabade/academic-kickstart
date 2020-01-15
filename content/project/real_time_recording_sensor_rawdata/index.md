---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Real_time_recording_sensor_rawdata"
subtitle: ""
summary: ""
authors: [admin]
tags: []
categories: [Software release, Python]
date: 2020-01-15T12:48:30Z
lastmod: 2020-01-15T12:48:30Z
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---
This is a small prototype that I had been working on when I was a visiting scholar (more like an intern) in Italy. The main purpose is reprogram a sensor receiver software (in python) into an executable file (i.e., EXE) in Windows OS.

**Background:** Two independent programs for data receiving via Bluetooth and visualization via browser respectively. In addition, two programs depend on various softwares and libraries (i.e., google chrome browser, tkinter, matplotlib etc.), which makes it difficult to install.  

**Goal:** combine two programs into one executable file without any other software and library dependence.  

**Method:** 1.  combine two programs into one program file with multiprocessing; 2. Visualize the sensor data and complete functions (i.e., extract data during usr specific time, export data into csv file, sample rate etc.) on a window created via tkinter and matplotlib; 3. convert python code into an single executable file via pyinstaller.  
