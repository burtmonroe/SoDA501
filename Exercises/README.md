# Exercises

## Exercise #1 - 2019 - Squinting at Google Trends and Google Ngrams
Due Monday, Jan 14 (which is defined operationally as before Tuesday, 7:00 am)

### Part A:

Consider the following search on Google Trends: <https://trends.google.com/trends/explore?date=all&geo=US&q=islam> (relative use of the search term "islam" in the US for all time available.

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/1671_RC04/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"islam","geo":"US","time":"2004-01-01 2019-01-07"}],"category":0,"property":""}, {"exploreQuery":"date=all&geo=US&q=islam","guestPath":"https://trends.google.com:443/trends/embed/"}); </script>

You can see what appears to be a seasonal pattern. I want your team to discuss what you think the cause of that is and try to think of comparison search terms that would follow a related pattern if that were the cause. It doesn't have to be identical ... you might think of something that should move in the opposite direction or be the same but shifted 3 months. But of course just finding seasonality isn't hard. 

How about <https://trends.google.com/trends/explore?date=all&geo=US&q=islam,oranges> :

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/1671_RC04/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"islam","geo":"US","time":"2004-01-01 2019-01-07"},{"keyword":"oranges","geo":"US","time":"2004-01-01 2019-01-07"}],"category":0,"property":""}, {"exploreQuery":"date=all&geo=US&q=islam,oranges","guestPath":"https://trends.google.com:443/trends/embed/"}); </script> 


Or <https://trends.google.com/trends/explore?date=all&geo=US&q=islam,basketball> :

<script type="text/javascript" src="https://ssl.gstatic.com/trends_nrtr/1671_RC04/embed_loader.js"></script> <script type="text/javascript"> trends.embed.renderExploreWidget("TIMESERIES", {"comparisonItem":[{"keyword":"islam","geo":"US","time":"2004-01-01 2019-01-07"},{"keyword":"basketball","geo":"US","time":"2004-01-01 2019-01-07"}],"category":0,"property":""}, {"exploreQuery":"date=all&geo=US&q=islam,basketball","guestPath":"https://trends.google.com:443/trends/embed/"}); </script>


In any case ... does it look like you were right? If not, keep trying.

Write a paragraph giving your team's best explanation for the pattern, and a small set of comparison terms that you think best support your case. 

### Part B:

This figure from Google Books Ngrams Viewer implies that texting peaked in the 17th century:

<https://books.google.com/ngrams/graph?content=fyi%2Cftw%2Cwtf&year_start=1650&year_end=2000&corpus=15&smoothing=10&share=&direct_url=t1%3B%2Cfyi%3B%2Cc0%3B.t1%3B%2Cftw%3B%2Cc0%3B.t1%3B%2Cwtf%3B%2Cc0>

What the hell is going on here? Use any evidence you want, or just conjecture. Write another paragraph with your team's best explanation. 




## Exercise #2 - 2019 - Michel et al. Replication
Due Monday, Jan 14 (which is defined operationally as before Tuesday, 7:00 am)

Complete the replication and discussion of Michel, et al., in Bit by Bit, Chapter 2, Exercise 6, parts (a) through (g).

Technical note: So that we don't have anyone blowing up their hard drives / bandwidth etc., please note that when Salganik says "Please read all parts of the question before you begin coding" and "Get the raw data from the Google Books NGram Viewer," he is telling you to get the data *you need to answer this question.* For part (a) you need counts of single words (unigrams or 1-grams) like "1875" or "1932." You only need to download the data from a single link on the page he provides. The file at that link is 283M compressed and 2.4G uncompressed. If you've downloaded more than that, you've downloaded more than you need. Moreover, the data you actually need is a tiny fraction of that 2.4G, so you don't need to keep all of it, or load all of it into memory at any one time. (For part b, you also need the tiny "total_counts" file.)

You'll do this in assigned teams of three.

Submit your team's answer, and the code, as an R Notebook. (Create an R Markdown file, .Rmd, with output: html_notebook, which will create a notebook file, .nb.html . Submit - or provide in a github repository - both files. If your code works directly on the two data files as downloaded, assume they are in the same directory as your code and don't submit them. If your code works directly on a small subset cleaned outside of R, submit those altered files. There are two excellent sources / tutorials / resources on R Markdown and R Notebooks: <https://rmarkdown.rstudio.com> and [R Markdown: The Definitive Guide](https://bookdown.org/yihui/rmarkdown) )

----



# Archive

### Data Wrangling Exercise (Exercise 3 - 2018)

1. [Tidyverse (R) Solution](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/TidyverseSolution)
2. [data.table (R) Solution](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/data.tableSolution)
3. pandas (Python) Solution
4. Trifacta Wrangler Solution
