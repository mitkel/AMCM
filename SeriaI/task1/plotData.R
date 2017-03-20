library(ggplot2); library(dplyr); library(tidyr)

params <- read.csv('realParameters.csv') %>% tbl_df
systematicGibbs <- read.csv('systematicGibbs.csv') %>% tbl_df
randomGibbs <- read.csv('randomGibbs.csv') %>% tbl_df

D <- bind_rows(
	systematicGibbs %>% mutate( update = 'systematic' ),
	randomGibbs %>% mutate( update = 'random' )
)

estimates<- D %>% group_by( update ) %>% summarise( lam = mean(lam), mu= mean(mu) )

nicePlot <- ggplot(data=D, aes( x=lam, y=mu)) + 
			# geom_point( alpha=.01 ) + 
			geom_point( data=params, aes( x=lam, y=mu), color='red', size=4 ) + 
			geom_point( data=estimates, aes( x=lam, y=mu), color='blue', size=4 ) + 
			geom_density_2d() +
			facet_grid(update~.) + 
			xlab('Lambda') + 
			ylab('Mu')
