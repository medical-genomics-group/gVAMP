
args <- commandArgs(TRUE)
file_name <- args[1]
start <- as.numeric(args[2])
end <- as.numeric(args[3])
Mt <- as.numeric(args[4])

library(data.table)
library("dplyr")  

file_loc <-"/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/output"
betas_table<-fread(paste(file_loc, "/", file_name, ".csv", sep="", collapse=NULL))

#aggregating the data and calculating the estimates
vals_gmrm <-betas_table %>% group_by(V2) %>% summarize(suma = sum(as.numeric(V3)))
vals_gmrm$suma <- vals_gmrm$suma / (end - start + 1)
vals <-numeric(Mt)
vals[vals_gmrm$V2] = vals_gmrm$suma

#saving the estimates
file_out <- paste(file_loc, "/", file_name, "_gibbs_est.csv", sep="", collapse=NULL) 
write.table(vals, file = file_out, sep = " ", col.names = FALSE, row.names = FALSE)
print("done saving the estimates")

# file_name <- "ukb_ht_noNA"
# start = 5
# end = 10