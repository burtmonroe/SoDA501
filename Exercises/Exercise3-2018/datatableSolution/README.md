## Solution with data.table

The notebook in this folder provides a solution to the data-wrangling exercise described in [Exercise3.pdf](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/Exercise3.pdf) using R library `data.table`.

 For alternative solutions see [https://burtmonroe.github.io/SoDA501/Exercise3-2018](https://burtmonroe.github.io/SoDA501/Exercise3-2018).
 
The `data.table` format does not seem to display consistently -- or I'm missing something -- within the usual `nb.html` format [here](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/Exercise3-datatableSolution.nb.html). The `html` version [here](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/Exercise3-datatableSolution.html) may be prettier. The R Notebook with executable code -- the Rmd file -- should be downloadable from the pulldown at upper right of the `nb.html` file or can be downloaded separately here: [Exercise3-datatableSolution.Rmd](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/Exercise3-datatableSolution.Rmd).  

#### General info on the exercise:

The input to the exercise is the raw data on 2016 Centre County, PA, precinct level votes found in ["CentreCountyPrecinctResults2016GeneralElection.txt"](https://burtmonroe.github.io/SoDA501/Exercises/Exercise3-2018/CentreCountyPrecinctResults2016GeneralElection.txt), which were retrieved from [http://centrecountypa.gov/Index.aspx?NID=802](http://centrecountypa.gov/Index.aspx?NID=802). (Update - 2020: This link is no longer accurate, and the original file does not appear to be available there any more. You can, however, find it on the Internet Archive's Wayback Machine, for example, here: [http://web.archive.org/web/20181106161514/centrecountypa.gov/index.aspx?NID=802](http://web.archive.org/web/20181106161514/centrecountypa.gov/index.aspx?NID=802). The file is no longer available due to the website being "updated" to provide the data in even less accessible form (pdf).)

The exercise asks you to extract the data on votes cast by precinct in statewide elections, and process them into a new table with precinct level data on total votes, Democratic share of two-party vote, and ballot rolloff from presidential votes to votes in other statewide races.

