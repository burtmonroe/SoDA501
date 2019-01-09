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

----

##Archive

### Data Wrangling Exercise (Exercise 3 - 2018)

1. [Tidyverse (R) Solution](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/TidyverseSolution)
2. [data.table (R) Solution](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/data.tableSolution)
3. pandas (Python) Solution
4. Trifacta Wrangler Solution
