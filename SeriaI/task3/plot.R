library("foreign"); library("ggplot2")

params 		<- read.csv('realP.csv')
impoSampling<- read.csv('importanceSampling.csv')
logWeights 	<- read.csv('logWeights.csv')

D <- exp(log(impoSampling) + matrix(logWeights,ncol=1))
D <- D %>% tbl_df

ggplot() + 
geom_point( data=D, aes( x=lam, y=mu), color='blue', size=.2, alpha=.3 ) + 
geom_point( data=params, aes( x=lam, y=mu), color='red', size=4 ) + 
xlab('Lambda') + 
ylab('Mu')

