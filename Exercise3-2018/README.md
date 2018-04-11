This document addresses one solution to the data-wrangling exercise described in [Exercise3.pdf](https://burtmonroe.github.io/SoDA501/Exercise3-2018/Exercise3.pdf). For the accompanying R Notebook with executable code, download the Rmd file from the pulldown "Code" menu on the upper right.  

The input to the exercise is the raw data on 2016 Centre County, PA, precinct level votes found in ["CentreCountyPrecinctResults2016GeneralElection.txt"](https://burtmonroe.github.io/SoDA501/Exercise3-2018/CentreCountyPrecinctResults2016GeneralElection.txt), which were retrieved from [http://centrecountypa.gov/Index.aspx?NID=802](http://centrecountypa.gov/Index.aspx?NID=802).

The exercise asks you to extract the data on votes cast by precinct in statewide elections, and process them into a new table with precinct level data on total votes, Democratic share of two-party vote, and ballot rolloff from presidential votes to votes in other statewide races.

This solution uses the R `tidyverse`. For alternative solutions see [https://burtmonroe.github.io/SoDA501/Exercise3-2018](https://burtmonroe.github.io/SoDA501/Exercise3-2018).
