suppressPackageStartupMessages(library(BayesFactor))
args <- commandArgs(trailingOnly=TRUE)
outdir <- args[1]

calc_one <- function(path) {
  d <- read.csv(path, check.names=FALSE)
  xcols <- grep("^x[0-9]+$", names(d), value=TRUE)
  out <- numeric(nrow(d))
  sec <- numeric(nrow(d))
  for (i in seq_len(nrow(d))) {
    x <- as.numeric(d[i,xcols])
    t0 <- proc.time()[3]
    bf <- ttestBF(x=x, mu=0.5, rscale="medium", nullInterval=c(0.5,Inf))
    sec[i] <- proc.time()[3]-t0
    out[i] <- as.vector(bf[1])
  }
  data.frame(train_i=d$train_i,test_i=d$test_i,source_bf=d$source_bf,recomputed_bf=out,seconds=sec)
}

calc_two <- function(path) {
  d <- read.csv(path, check.names=FALSE)
  xcols <- grep("^x[0-9]+$", names(d), value=TRUE)
  ycols <- grep("^y[0-9]+$", names(d), value=TRUE)
  out <- numeric(nrow(d))
  sec <- numeric(nrow(d))
  for (i in seq_len(nrow(d))) {
    x <- as.numeric(d[i,xcols]); y <- as.numeric(d[i,ycols])
    t0 <- proc.time()[3]
    bf <- ttestBF(x=x, y=y, mu=0, rscale="medium", nullInterval=c(0.5,Inf))
    sec[i] <- proc.time()[3]-t0
    out[i] <- as.vector(bf[1])
  }
  data.frame(train_i=d$train_i,test_i=d$test_i,source_bf=d$source_bf,recomputed_bf=out,seconds=sec)
}

summarize <- function(df) {
  ad <- abs(df$recomputed_bf-df$source_bf)
  rd <- ad/pmax(abs(df$source_bf), .Machine$double.eps)
  list(
    n=nrow(df),
    max_abs_diff=max(ad),
    median_abs_diff=median(ad),
    max_rel_diff=max(rd),
    median_rel_diff=median(rd),
    pearson=cor(df$source_bf,df$recomputed_bf),
    exact_equal_count=sum(df$source_bf==df$recomputed_bf),
    mean_seconds_per_cell=mean(df$seconds),
    total_seconds=sum(df$seconds)
  )
}

vt <- calc_one(file.path(outdir,"vt_touch2vis_qa.csv"))
nvt <- calc_one(file.path(outdir,"nvt_touch2vis_qa.csv"))
bg <- calc_two(file.path(outdir,"between_touch2vis_qa.csv"))
write.csv(vt,file.path(outdir,"vt_touch2vis_qa_recomputed.csv"),row.names=FALSE)
write.csv(nvt,file.path(outdir,"nvt_touch2vis_qa_recomputed.csv"),row.names=FALSE)
write.csv(bg,file.path(outdir,"between_touch2vis_qa_recomputed.csv"),row.names=FALSE)

j <- list(
  status="SOURCE_BAYESFACTOR_QA_GRID_RECOMPUTED",
  BayesFactor_version=as.character(packageVersion("BayesFactor")),
  R_version=R.version.string,
  source_semantics=list(
    one_sample='ttestBF(x=x, mu=0.5, rscale="medium", nullInterval=c(0.5,Inf)); as.vector(bf[1])',
    between='ttestBF(x=x, y=y, mu=0, rscale="medium", nullInterval=c(0.5,Inf)); as.vector(bf[1])'
  ),
  VT=summarize(vt),
  NVT=summarize(nvt),
  between_groups=summarize(bg)
)
jsonlite::write_json(j,file.path(outdir,"bayesfactor_qa_result.json"),auto_unbox=TRUE,pretty=TRUE,digits=17)
print(j)
