PacLPSolver <- function(A, tol){
 # Serial algorithm implementation of Zhu & Orecchia (2014) 
  # Args: 
  #       A: m by n matrix with non-negative entries
  #       x.init: initial solution estimate such that
  #               (x.init)_i \in [0, 1/||A,i||_\infty]
  #       tol: the epsilon
  
  m <- nrow(A)
  n <- ncol(A)
  stopifnot(A>=0)
 
  mu <- tol/(4*log(n*m/tol))
  L <- 4/mu
  tau <- 1/(3*n*L)
  alpha <- 1/(n*L)
  
  infnorm <- apply(A, 2, max) 
  x <- (1-tol/2)/infnorm 
  y <- x
  z <- rep(0,n)
  T <- ceiling(3*n*L*log(1/tol))
  print(T, file = TRUE)
 
  for(k in seq(T)){
    alpha <- alpha/(1-tau) 
    x <- tau*z + (1-tau)*y 
    i <- sample(n,1)
    
    xii <- Threshold(A = A, x = x, mu = mu, i = i)
    
   # z.new[i] <- Mirror(infnorm[i], zi = z[i], delta = n*alpha*xii)
    tmp <- Mirror(infnorm[i], zi = z[i], delta = n*alpha*xii)
    y <- x
    
   # y <- x + 1/(n*alpha*L)*(z.new-z)
   # z <- z.new
    y[i] <- x[i]+1/(n*alpha*L)*(tmp - z[i])
    z[i] <- tmp
    #print(as.numeric(crossprod(x-y,rep(1,n))),fill=TRUE)
  }
  return(y)
}

Mirror <- function(infnorm, zi, delta){
  res <- zi - delta/infnorm
  if(res < 0) 
    res <- 0
  if(res > 1/infnorm)
      res <- 1/infnorm
  return(res)
}

Threshold <- function(A, x, mu, i){
  v <- crossprod(exp((A%*%x - 1)/mu), A[,i]) - 1
  res <- ifelse(v>1, 1, v) 
  return(res) 
}