Solutions to the data-wrangling exercise described in [Exercise3.pdf](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/Exercise3.pdf). 

The input to the exercise is the raw data on 2016 Centre County, PA, precinct level votes found in ["CentreCountyPrecinctResults2016GeneralElection.txt"](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/CentreCountyPrecinctResults2016GeneralElection.txt), which were retrieved from [http://centrecountypa.gov/Index.aspx?NID=802](http://centrecountypa.gov/Index.aspx?NID=802).  (Update - 2020: This link is no longer accurate, and the original file does not appear to be available there any more. You can, however, find it on the Internet Archive's Wayback Machine, for example, here: [http://web.archive.org/web/20181106161514/centrecountypa.gov/index.aspx?NID=802](http://web.archive.org/web/20181106161514/centrecountypa.gov/index.aspx?NID=802). The file is no longer available due to the website being "updated" to provide the data in even less accessible form (pdf).)


The exercise asks you to extract the data on votes cast by precinct in statewide elections, and process them into a new table with precinct level data on total votes, Democratic share of two-party vote, and ballot rolloff from presidential votes to votes in other statewide races.


The exercise asks you to extract the data on votes cast by precinct in statewide elections, and process them into a new table with precinct level data on total votes, Democratic share of two-party vote, and ballot rolloff from presidential votes to votes in other statewide races.


There are three alternative solutions provided:

1. [R - tidyverse (dplyr)](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/TidyverseSolution/)
2. [R - data.table](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/datatableSolution/Exercise3-datatableSolution.html)
3. Python - pandas
