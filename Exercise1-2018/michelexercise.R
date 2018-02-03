#setwd("/Users/burtmonroe/Dropbox/SoDA501/SoDA501-Spring2018/Exercise1")
library(tidyr)
library(haven)
library(readr)
library(stringr)
library(dplyr)

micheldata <- read_tsv("micheldata.txt", col_names=c("ngram","year","term_count","volume_count"), col_types="ciii")

totcounts <- read_csv("totcounts.csv", col_names=c("year","match_count","page_count", "vol_count"))

merged.data <- inner_join(micheldata,totcounts)

merged.data <- mutate(merged.data,frac=term_count/match_count)

merged.data <- mutate(merged.data,f.sc = 10000*frac)

attach(merged.data)

pdf(file="Michel3A.pdf", width=8, height=6)
plot(c(1875,2000),c(0,1.5), type="n", main="Figure 3a (main)",ylab="Frequency (1e-4)", xlab="Year")
lines(year[ngram==1883],f.sc[ngram==1883],col="blue", lwd=2)
lines(year[ngram==1910],f.sc[ngram==1910],col="green", lwd=2)
lines(year[ngram==1950],f.sc[ngram==1950],col="red", lwd=2)
legend(1975,1.4,legend = c("1883","1910","1950"), col=c("blue","green","red"),cex=.8, lwd=2, y.intersp=1)
dev.off()

pdf(file="Michel3ATotals.pdf", width=8, height=6)
plot(c(1875,2000),c(0,max(term_count[ngram==1950])), type="n", main="Figure 3a, absolute counts",ylab="Count", xlab="Year")
lines(year[ngram==1883],term_count[ngram==1883],col="blue", lwd=2)
lines(year[ngram==1910],term_count[ngram==1910],col="green", lwd=2)
lines(year[ngram==1950],term_count[ngram==1950],col="red", lwd=2)
legend(1975,1.4,legend = c("1883","1910","1950"), col=c("blue","green","red"),cex=.8, lwd=2, y.intersp=.5)
dev.off()


halflife <- rep(0,101)
halflife2 <- rep(0,101)
peak <- rep(0,101)

for (i in 1:101) {
  y <- 1874+i
  postyeardata <- f.sc[ngram == as.character(y) & year >= y]
  halflife[i] <- sum(postyeardata > 0.5*max(postyeardata))
  halflife2[i] <- max(sort.list(postyeardata[postyeardata > 0.5*max(postyeardata)], dec=T))
  peak[i] <- max(postyeardata)
}
 
cor(halflife,halflife2)

pdf(file="Michel3AinsetHalflives.pdf", width=8, height=6)
plot(1875:1975,halflife, pch=19, col=rgb(0,0,0,.5), main="Halflife (Figure 3a inset)", ylab = "Halflife (Years)", xlab = "Year - Ngram")
dev.off() 

pdf(file="MichelPeakHalflife.pdf", width=8, height=8)
plot(c(5,25), c(0.5,2.5), type="n", main="Halflife vs. Peak for Year-ngrams", xlab="Halflife (Years)", ylab = "Peak (Frequency)")
points(halflife, peak, pch=19, col=rgb(0,0,0,.5))
text(halflife, peak,1875:1975, pos=4, cex=.8, col=rgb(0,0,0,.7))
dev.off()
