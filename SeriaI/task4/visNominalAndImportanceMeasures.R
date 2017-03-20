library("foreign"); library("ggplot2")

plot(dnorm,xlim=c(-5,5))
x=seq(-5,5,by=.01)
lines(x,dt(x,df=5),col=2)
impoSampling<- read.csv('importanceSampling.csv')
logWeights 	<- read.csv('logWeights.csv')
logWeightsNominal <- read.csv('logWeightsNominal.csv')
logWeightsImportance <- read.csv('logWeightsImportance.csv')
