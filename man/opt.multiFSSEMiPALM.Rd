% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/solver.R
\name{opt.multiFSSEMiPALM}
\alias{opt.multiFSSEMiPALM}
\title{opt.multiFSSEMiPALM}
\usage{
opt.multiFSSEMiPALM(
  Xs,
  Ys,
  Bs,
  Fs,
  Sk,
  sigma2,
  nlambda = 20,
  nrho = 20,
  p,
  q,
  wt = TRUE
)
}
\arguments{
\item{Xs}{eQTL matrices}

\item{Ys}{Gene expression matrices}

\item{Bs}{initialized GRN-matrices}

\item{Fs}{initialized eQTL effect matrices}

\item{Sk}{eQTL index of genes}

\item{sigma2}{initialized noise variance}

\item{nlambda}{number of hyper-parameter of lasso term in CV}

\item{nrho}{number of hyper-parameter of fused-lasso term in CV}

\item{p}{number of genes}

\item{q}{number of eQTLs}

\item{wt}{use adaptive lasso or not. Default TRUE.}
}
\value{
list of model selection result
}
\description{
optimize multiFSSEMiPALM's parameters by minimize BIC,
when feature size is large (> 300), BIC methods will be much faster than Cross-validation
}
\examples{
seed = 1234
N = 100                                           # sample size
Ng = 5                                            # gene number
Nk = 5 * 3                                        # eQTL number
Ns = 1                                            # sparse ratio
sigma2 = 0.01                                     # sigma2
set.seed(seed)
library(fssemR)
data = randomFSSEMdata(n = N, p = Ng, k = Nk, sparse = Ns, df = 0.3, sigma2 = sigma2,
                       u = 5, type = "DG", nhub = 1, dag = TRUE)
## If we assume that different condition has different genetics perturbations (eQTLs)
## data$Data$X = list(data$Data$X, data$Data$X)
## gamma = cv.multiRegression(data$Data$X, data$Data$Y, data$Data$Sk, ngamma = 20, nfold = 5,
##                            N, Ng, Nk)
gamma = 0.6784248     ## optimal gamma computed by cv.multiRegression
fit   = multiRegression(data$Data$X, data$Data$Y, data$Data$Sk, gamma, N, Ng, Nk,
                      trans = FALSE)
Xs    = data$Data$X
Ys    = data$Data$Y
Sk    = data$Data$Sk

fitm <- opt.multiFSSEMiPALM(Xs = Xs, Ys = Ys, Bs = fit$Bs, Fs = fit$Fs, Sk = Sk,
                           sigma2 = fit$sigma2, nlambda = 10, nrho = 10,
                           p = Ng, q = Nk, wt = TRUE)

fitc0 <- fitm$fit

(TPR(fitc0$Bs[[1]], data$Vars$B[[1]]) + TPR(fitc0$Bs[[2]], data$Vars$B[[2]])) / 2
(FDR(fitc0$Bs[[1]], data$Vars$B[[1]]) + FDR(fitc0$Bs[[2]], data$Vars$B[[2]])) / 2
TPR(fitc0$Bs[[1]] - fitc0$Bs[[2]], data$Vars$B[[1]] - data$Vars$B[[2]])
FDR(fitc0$Bs[[1]] - fitc0$Bs[[2]], data$Vars$B[[1]] - data$Vars$B[[2]])
}
