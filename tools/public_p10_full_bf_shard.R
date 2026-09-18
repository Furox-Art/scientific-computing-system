suppressPackageStartupMessages(library(BayesFactor))
args <- commandArgs(trailingOnly=TRUE)
inp <- args[1]; outp <- args[2]
d <- read.csv(inp,check.names=FALSE)
xcols <- grep("^x[0-9]+$",names(d),value=TRUE)
ycols <- grep("^y[0-9]+$",names(d),value=TRUE)
vt <- numeric(nrow(d)); nvt <- numeric(nrow(d)); bg <- numeric(nrow(d))
for(i in seq_len(nrow(d))){
  x <- as.numeric(d[i,xcols]); y <- as.numeric(d[i,ycols])
  vt[i] <- as.vector(ttestBF(x=x,mu=0.5,rscale="medium",nullInterval=c(0.5,Inf))[1])
  nvt[i] <- as.vector(ttestBF(x=y,mu=0.5,rscale="medium",nullInterval=c(0.5,Inf))[1])
  bg[i] <- as.vector(ttestBF(x=x,y=y,mu=0,rscale="medium",nullInterval=c(0.5,Inf))[1])
}
o <- data.frame(train_i=d$train_i,test_i=d$test_i,VT_BF=vt,NVT_BF=nvt,Between_BF=bg)
write.csv(o,outp,row.names=FALSE)
