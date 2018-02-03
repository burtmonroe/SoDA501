#time wget http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-1gram-20120701-1.gz
#
#time gunzip *.gz
#
#time wget http://storage.googleapis.com/books/ngrams/books/googlebooks-eng-all-totalcounts-20120701.txt

echo Finding year ngrams ...

time LANG=C grep -P "^18[789]\d\s|^19\d\d\s" google*-1 > micheldata.txt

sort micheldata.txt -o micheldata.txt

echo Cleaning total_counts file ...

time tr '\t' '\n' < googlebooks-eng-all-totalcounts-20120701.txt > totcounts.txt

time tail -n +2 totcounts.txt > totcounts.csv

echo Running R script to create graphs

module use /gpfs/group/blm24/default/sw/modules

module load R/3.4.1-gcc-5.3.1

time Rscript michelexercise.R
