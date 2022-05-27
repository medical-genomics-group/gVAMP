## simple R script to simulate example genotype data
## MRR 14.07.21
## This requires the software plink: https://www.cog-genomics.org/plink2

set.seed(171014)
require(MASS)

## set sample size, N
N = 10000

## total number of covariates, M
M = 20000

## simulate marker data
X <- matrix(rbinom(N*M,2,0.4),N,M)

## total variance explained by marker effects
h2 = 0.5

## variance of the marker effects
## only 5000 causal markers
vg = h2/5000

## sample marker effects
sigma_b <- diag(2)*vg
b <- mvrnorm(5000,c(0,0),sigma_b)

## generate genetic values
beta <- matrix(rep(0,M*2),M,2)
index1 <- sample(1:M,5000)
beta[index1,1] <- b[,1]
beta[index1,2] <- b[,2]
g <- scale(X) %*% beta

## generate residuals
sigma_e <- matrix(c(1-var(g[,1]), 0,
                      0, 1-var(g[,2])),2,2)
e <- mvrnorm(N,c(0,0),sigma_e)

## output phenotype
y = g + e

## output genetic data
X[X == 2] <- "AA"
X[X == 1] <- "AG"
X[X == 0] <- "GG"

## output to plink .ped/.map format
ped <- data.frame("FID" = 1:N,
                  "IID" = 1:N,
                  "PID" = rep(0,N),
                  "MID" = rep(0,N),
                  "Sex" = rep(1,N),
                  "phen" = rep(0,N))
ped <- cbind(ped,X)
write.table(ped,"test.ped", row.names=FALSE, col.names=FALSE, quote=FALSE)

map <- data.frame("chr" = rep(1,M),
                  "rs" = paste("rs",1:M, sep=''),
                  "dist" = rep(0,M),
                  "bp" = 1:M)
write.table(map,"test.map", row.names=FALSE, col.names=FALSE, quote=FALSE)

## convert from .ped/.map to plink binary format
system("plink --file test --make-bed --out test")

## remove .ped/.map files
system("rm *.ped")
system("rm *.map")
system("rm *.log")

## output phenotype files
phen <- data.frame("FID" = 1:N,
                   "IID" = 1:N,
                   "phen1" = y[,1],
                   "phen2" = y[,2])
write.table(phen[,c(1,2,3)],"test1.phen", row.names=FALSE, col.names=FALSE, quote=FALSE)
write.table(phen[,c(1,2,4)],"test2.phen", row.names=FALSE, col.names=FALSE, quote=FALSE)

