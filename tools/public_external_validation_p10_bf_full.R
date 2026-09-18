suppressPackageStartupMessages(library(BayesFactor))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
infile <- args[1]
outfile <- args[2]
summaryfile <- args[3]
d <- read.csv(infile, check.names=FALSE)
vtcols <- grep("^vt[0-9]+$", names(d), value=TRUE)
nvtcols <- grep("^nvt[0-9]+$", names(d), value=TRUE)

calc_ids <- function(ids) {
  out <- matrix(NA_real_, nrow=length(ids), ncol=3)
  for (j in seq_along(ids)) {
    i <- ids[j]
    x <- as.numeric(d[i,vtcols]); y <- as.numeric(d[i,nvtcols])
    out[j,1] <- as.vector(ttestBF(x=x, mu=0.5, rscale="medium", nullInterval=c(0.5,Inf))[1])
    out[j,2] <- as.vector(ttestBF(x=y, mu=0.5, rscale="medium", nullInterval=c(0.5,Inf))[1])
    out[j,3] <- as.vector(ttestBF(x=x, y=y, mu=0, rscale="medium", nullInterval=c(0.5,Inf))[1])
  }
  out
}

ncores <- min(2L, parallel::detectCores())
ids <- seq_len(nrow(d))
parts <- split(ids, cut(ids, breaks=ncores, labels=FALSE))
t0 <- proc.time()[3]
pieces <- parallel::mclapply(parts, calc_ids, mc.cores=ncores)
res <- do.call(rbind,pieces)
ord <- unlist(parts)
res2 <- matrix(NA_real_, nrow=nrow(d), ncol=3)
res2[ord,] <- res
elapsed <- proc.time()[3]-t0

z <- data.frame(
  train_i=d$train_i,test_i=d$test_i,
  source_vt=d$source_vt,recomputed_vt=res2[,1],
  source_nvt=d$source_nvt,recomputed_nvt=res2[,2],
  source_between=d$source_between,recomputed_between=res2[,3]
)
con <- gzfile(outfile,"wt")
write.csv(z,con,row.names=FALSE)
close(con)

summ <- function(source,recomp) {
  ad <- abs(recomp-source); rd <- ad/pmax(abs(source),.Machine$double.eps)
  imax <- which.max(recomp)
  list(
    n=length(source),
    source_BF_gt_6=sum(source>6), recomputed_BF_gt_6=sum(recomp>6),
    BF_gt_6_classification_mismatches=sum((source>6)!=(recomp>6)),
    source_BF_lt_1_over_6=sum(source<(1/6)), recomputed_BF_lt_1_over_6=sum(recomp<(1/6)),
    BF_lt_1_over_6_classification_mismatches=sum((source<(1/6))!=(recomp<(1/6))),
    max_abs_diff=max(ad), max_rel_diff=max(rd), median_abs_diff=median(ad), median_rel_diff=median(rd),
    pearson=cor(source,recomp),
    recomputed_max_BF=max(recomp),
    recomputed_max_train_i=z$train_i[imax],
    recomputed_max_test_i=z$test_i[imax]
  )
}
j <- list(
  status="P10_FULL_PRIMARY_BF_SHARD_RECOMPUTED",
  BayesFactor_version=as.character(packageVersion("BayesFactor")),
  R_version=R.version.string,
  rows=nrow(d), elapsed_seconds=elapsed, ncores=ncores,
  VT=summ(z$source_vt,z$recomputed_vt),
  NVT=summ(z$source_nvt,z$recomputed_nvt),
  between_groups=summ(z$source_between,z$recomputed_between)
)
write_json(j,summaryfile,auto_unbox=TRUE,pretty=TRUE,digits=17)
print(j)
