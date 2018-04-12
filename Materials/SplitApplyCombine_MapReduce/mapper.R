#!/usr/bin/env Rscript
library(stringr)
library(magrittr)

f <- file("stdin")
open(f)
while(length(line <- readLines(f,n=1)) > 0) {
  splitline <- line %>%
    str_to_lower %>%
    str_replace_all("[[:punct:]]", "") %>%
    str_replace_all("\\s\\s+","\\s")   %>%
    str_split("\\s",simplify=FALSE)     %>%
    unlist
  for (word in splitline) {
    cat(word,"\t",1,"\n")
  }
}
