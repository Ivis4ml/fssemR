% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{logLikFSSEM}
\alias{logLikFSSEM}
\title{logLikFSSEM}
\usage{
logLikFSSEM(Bs, Wl, Wf, lambda, rho, sigma2, Dets, n, p)
}
\arguments{
\item{Bs}{Network matrices}

\item{Wl}{Weights for lasso term}

\item{Wf}{Weights for fused term}

\item{lambda}{Hyperparameter of lasso term}

\item{rho}{Hyperparameter of fused lasso term}

\item{sigma2}{noise variance}

\item{Dets}{determinants of I-B matrices}

\item{n}{number of observations}

\item{p}{number of genes}
}
\value{
objective value of FSSEM with specified hyper-paramters
}
\description{
logLikFSSEM
}
