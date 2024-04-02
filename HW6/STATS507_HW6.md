---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.12.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name: Li Shuyan
# UMID: 63161545


> Github: https://github.com/lishuyanumich/Stats507


# Q1


1. To create a file named "Stats507" on my desktop, in the terminal I enter my Desktop first.

`cd Desktop`

Then,

`git clone git@github.com:lishuyanumich/Stats507.git`.

Next, I extract your code from PS2, Question 3 into a stand alone script or notebook and paste it to the file "Stats507" on my desktop.

Then, I enter "Stats507":

`cd Stats507`

Next, I add it to the repo:

`git add .`

`git commit -m "initial commit"`

`git push`



2. I create a README.

`echo "# Stats507" >> README.md`

`git init`

Then I use jupyter notebook open this README.md and briefly documents the purpose of the repo created in the warmup. And in the README, briefly document the script you included for the previous part - state what it dose and for what purpose. The README should include a link to this file.

`git add README.md`

`git commit -m "initial commit"`

`git push`


3. I commit the changes from the previous step and push them to the remote. A direct link to the commit from the remote’s history is https://github.com/lishuyanumich/Stats507/commit/4a1b44407f4056f9229d1e7b503062e998656a27.


4. Create a branch named “ps4”. 

`git checkout -b ps4`

Checkout that branch and edit the file from step 3 to include “Gender” as you did for PS4 Q1. Commit these changes to the branch and create an upstream branch on GitHub to track this branch. 

`git add PS2_Q3.ipynb`

`git commit -m "Add gender"`

`git add README.md`

`git commit -m "Add gender"`

`git push --set-upstream origin ps4`


5. Merge the “ps4” branch into the “main” branch. 

`git merge ps4`

Include a direct link to the commit from the remote’s history: https://github.com/lishuyanumich/Stats507/compare/ps4?expand=1.


# Q2 GitHub Collaboration
In this you question will extract your notes on a Panda’s topic from PS4, Question 0 to a script of their own. Then, you will collaborate to aggregate these into a single document for the course.

1. In your Stats507 repo make a folder called “pandas_notes”.

2. Extract your PS4, Question 0 topic tutorial and copy it into a script called “pd_topic_XYZ.py” replacing XYZ with your UM unique name. Include your name and UM email on a title “slide” (markdown cell) if you don’t have one already. Include a link in your writeup to this file.

`git checkout -b newbranch1`

`git add pandas_notes/pd_topic_lishuyan.py`

`git commit -m "new folder"`

`git checkout master`

`git merge newbranch1`

`git push -u origin master`

`git branch -D newbranch1`

https://github.com/lishuyanumich/Stats507/blob/master/pandas_notes/pd_topic_lishuyan.py
