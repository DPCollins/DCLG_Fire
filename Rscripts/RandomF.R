library(miscTools)
library(randomForest)
library(MASS)
library(ggplot2)
library(party)
library(reshape2)
library(leaps)

graphics.off()
source("multiplot.R")



#RFT = read.csv("RTestRF.csv")  
RFT <- read.csv("FiresRTest.csv")
FireT <- RFT

FireT <- FireT[FireT$Fire_Incidents2012<800,]
RFT <- RFT[RFT$Fire_Incidents2012<800,]

head(RFT) # head of the dataframe

COLS <- names(RFT) # Feature name lookup (as we change names to X1 etc below)

f <- c(1:ncol(RFT))
f2 <- paste('X', f, sep='')
colnames(RFT) <- f2 

d_fra <- 0.7
#d_fra <- 0.6

shuf <- sample(nrow(RFT))

X_Train <- RFT[shuf[1:round(d_fra*length(shuf))],1:ncol(RFT)-1]
Y_Train <- RFT[shuf[1:round(d_fra*length(shuf))], ncol(RFT)]

X_Test <- RFT[shuf[round(d_fra*length(shuf)):length(shuf)],1:ncol(RFT)-1]
Y_Test <- RFT[shuf[round(d_fra*length(shuf)):length(shuf)],ncol(RFT)]

#X_Train <- RFT[1:round(d_fra*nrow(RFT)),1:ncol(RFT)-1]
#Y_Train <- RFT[1:round(d_fra*nrow(RFT)), ncol(RFT)]

#X_Test <- RFT[round(d_fra*nrow(RFT)):nrow(RFT),1:ncol(RFT)-1]
#Y_Test <- RFT[round(d_fra*nrow(RFT)):nrow(RFT),ncol(RFT)]


### CForest ##############################################################

CFor <- function() {
# 	a <- dLoad()

	#set.seed(47)
	CF <- cforest(Y_Train ~ ., X_Train, controls =cforest_unbiased(ntree=100))

	varImpCF <- system.time(varimp(CF, conditional=TRUE))
	v2 <- as.data.frame(varImpCF)
	v2['lab'] = labels(varImpCF)

	v2[order(v),]

	ggplot(aes(x=lab, y=v, label=lab), data=v2) + geom_point() + geom_text(data=subset(v2, v>0), vjust=2, size=4) + scale_x_discrete(breaks=v2[v2['v']>0])

	r2CF <- rSquared(Y_Test, Y_Test - predict(CF,newdata=X_Test))
	
	col <- subset(v2, v>0)

	}

##########################################################################



#CF <-cforest(Species ~ ., data=iris, controls=cforest_control(mtry=2, mincriterion=0))
#tr <- party:::prettytree(CF@ensemble[[0]], names(CF@data@get("input")))
#plot(new("BinaryTree", tree=tr, data=CF@data, responses=CF@responses))

##################################################

# Useful plotting of dataframe

# d <- melt(diamonds[,-c(2:4)])
# ggplot(d,aes(x = value)) + 
#     facet_wrap(~variable,scales = "free_x") + 
#     geom_histogram()

##################################################


RandF <- function() {
	
	graphics.off()
	
# 	a <-dLoad()
# 	Y_Train <- a[[2]]
# 	X_Train <- a[[1]]
# 	
# 	Y_Test <- a[[4]]
# 	X_Test <- a[[3]]

	RF <- randomForest(Y_Train ~ ., X_Train, ntree=1000, proximity=TRUE, importance=TRUE)

	RF_impTI <- importance(RF,type=1)
	RF_impT2 <- importance(RF, type=2)

	h2 <- order(RF_impTI, decreasing=TRUE)
 	


	quartz()
	plot(RF, log="y")

	quartz()
	varImpPlot(RF)

	r2 <- rSquared(Y_Test, Y_Test - predict(RF, X_Test))


	mse <- mean((Y_Test - predict(RF, X_Test))^2)
	MeanMSE <- sum((Y_Test-mean(Y_Train))^2)/length(Y_Test)

	print(paste("R^2 is: ",r2))
	print(paste("MSE is: ", mse))
	print(paste("MSE based on Mean: ", MeanMSE)) 


	p <- ggplot(aes(x=actual, y=pred),
	  data=data.frame(actual=Y_Test, pred=predict(RF, X_Test)))
	q <- p + geom_point() +
		geom_abline(color="red") +
		ggtitle(paste("RandomForest Regression in R r^2=", r2, sep=""))

	quartz()
	print(q)
	
	cc <- COLS[h2[1:10]]
	
	print(cc)
	
	return(cc)	

	}

###################################################################

TopFeat <- function(reps) { 

	
	m <- matrix(,reps,10)

	for (i in 1:reps) {
		print(i)
		gg <- RandF()
		for (j in 1:10) {
			m[i,j] <- gg[j]
			}
		}
	return(as.data.frame(m))
	}	

# plotting varimp from TopFeat (frequency in position 1, 2, 3 etc

freq=table(col(as.matrix(Test$V3)), as.matrix(Test$V3))
f <-melt(freq)
ggplot(f, aes(Var2, value))+geom_bar(aes(fill=Var1), position="dodge", stat="identity")





# ggplot(FireT, aes(x = Fire_Incidents2012))+geom_histogram()

###################################################################

# line or scatter two columns and colour by column name (melt allows this)

#df <- data.frame(x=c(1:10), y1=c(10:1), y2=c(1:10))
#mdf <- melt(df, id.vars="x")
#qplot(x, value, group=variable, data=mdf, geom="line")


#######################################################################################
# Day of Week Study of all Fires

# Y <- read.csv('TukeyHSD.csv')
# Y$DOW <- as.factor(Y$DOW)
# summary(Y)
# tapply(Y$Fires, Y$DOW, mean)
# aj <- lm(Y$Fires ~ Y$DOW)
# summary(aj)
# anova(aj)
# 
# a1 <- aov(Y$Fires ~ Y$DOW)
# 
# posthoc <- TukeyHSD(x=a1, 'Y$DOW', conf.level=0.95)
# plot(posthoc)


#######################################################################################

##### Model selection R LEAPS AIC Step etc.

#x1 <- RFT[,1:length(RFT)-1]
x1 <- FireT[,1:5]
y1 <- FireT[,length(FireT)]

Leap = regsubsets(x = x1, y = y1, names = names(FireT)[1:length(FireT)-1], nbest=10)

l <- leaps(x=x1, y=y1, names=names(FireT)[1:5], method="adjr2")


### STEP ###########################################

null <- lm(Fire_Incidents2012~1, data = FireT)
full <- lm(Fire_Incidents2012~., data = FireT)
 
lstep <- step(null, scope=list(upper=full), data=FireT, direction="both", test="F")

mpd <- lm(scale(Fire_Incidents2012) ~ scale(Total.Crime.Rate.2011.12) + scale(Area_Sq_km) + scale(All.Household.Spaces2011) +scale(Median.House.Price) + scale(Annual_Mean_of_NO2) + scale(X..NotBorn_UK)  + scale(X.Other_qualifications) + scale(X._Domestic_Buildings_.2005.) + scale(Annual_Mean_of_Nox_.microg.m.3.), data=FireT)

### LASSO #########################################

# use crossvalidation to find the best lambda
library(glmnet)
# x = scale(FireT[,1:length(FireT)-1])
# y = matrix(scale(FireT$Fire_Incidents2012))

x = scale((FireT[,1:length(FireT)-1]))
y = scale(matrix((FireT$Fire_Incidents2012)))

x2 <- as.matrix(x)

DAT <- data.frame(x)
DAT['FI'] <- y
# DAT <- data.frame(scale(DAT))
#scatterplotMatrix(DAT[,c(73,74,75,78,84)])

LASS <- function(TS,num, alpha) {

# samp is test sample size in %
	
	st <- vector()
	av <- vector()
	rr <- vector()
	
	for (i in 1:num) {
	
		cat("\n")
		print('##############################')
		cat("Iteration: ", i, "\n")
		print('##############################')
		
		samp <- round(TS*nrow(DAT))
	
		vec <- c(1:nrow(DAT))
		vshuf <- sample(vec, replace=FALSE, prob=NULL)
	
	
		x3 <- as.matrix(DAT[vshuf[1:samp],1:ncol(DAT)-1])
		y3 <- as.matrix(DAT$FI[vshuf[1:samp]])

		xT <- DAT[vshuf[samp:nrow(DAT)], 1:ncol(DAT)-1]
		yT <- DAT$FI[vshuf[samp:nrow(DAT)]]


		# alpha=1 - LASSO alpha=0 RIDGE

		cv <- cv.glmnet(x3,y3, family="gaussian", alpha=alpha,nfolds=10)
		l <- cv$lambda.min
		#alpha=1

		# fit the model
		fits <- glmnet( x3, y3, alpha=alpha, family="gaussian", nlambda=100)
		# Find coefficients
		res <- predict(fits, s=l, type="coefficients")
		
		print(res)

		# Test by fitting held back Test sample

		res2 <- predict(fits, s=l, newx=as.matrix(xT), type="response")

		GG = data.frame(res2)
		GG['yT'] = yT

		gp <- ggplot(GG, aes(x=X1, y=yT)) + geom_point(size=2, colour = "red") + geom_line(aes(x=yT, y=yT), colour="black")

		GG['id'] <-1:nrow(GG)
		GG.melted <- melt(GG, id="id")

		gp2 <- ggplot(data = GG.melted, aes(x = id, y = value, color = variable)) + geom_point() + geom_line()

		multiplot(gp,gp2)
	
		MSE <- mean((yT-res2)^2)
		ASE <- sqrt(MSE)
		medSE <- sqrt(median((yT-res2)^2))

		MEAN <- mean((yT-mean(y3))^2)
		
		St <- sum((yT-mean(yT))^2)
		Sr <- sum((yT-res2)^2)
		
		r2 <- 1 - (Sr/St)
		
		Cols <- rownames(res)[res@i+1][2:length(res@i)]  ## these are non zeros values from LASSSO
	
		print(ASE)
		print(medSE)
		print(Cols)
		
		st[i] <- MSE
		av[i] <- MEAN
		rr[i] <- r2 
		
	}
return(list(st, av, rr, res))
}

####################################################################
Top <- rownames(x[[4]])[order(x[[4]], decreasing=TRUE)][1:5]
Bot <- rownames(x[[4]])[order(x[[4]], decreasing=FALSE)][1:5]

F = data.frame(x[[3]])
colnames(F) <- 'X1'

ggplot(F, aes(x=X1)) + 
						geom_histogram(aes(y=..density..), binwidth=0.02, fill='white', colour='black', alpha=0.7) + 
					    geom_density(fill="blue", alpha=0.4)


#####################################################################

g <- as.data.frame(x)
x3 <- g[colnames(g[Cols])]

### below is result from res
LAS <- lm(scale(Fire_Incidents2012) ~ scale(Total.Crime.Rate.2011.12) +  scale(Area_Sq_km) + scale(X.CouncilTaxBandAorB) + scale(Annual_Mean_of_NO2) + scale(X._Domestic_Buildings_.2005.)  + scale(X.of_16._.who_are_schoolchildren_and.full.time.students..Age.18.and.over) + scale(X.CouncilTaxBandAorB)  + scale(All.Dwellings2011), data=FireT)

LAS <- lm(scale(Fire_Incidents2012) ~ scale(Total.Crime.Rate.2011.12) +  scale(Area_Sq_km) + scale(X.CouncilTaxBandAorB) + scale(Annual_Mean_of_NO2)  + scale(X.of_16._.who_are_schoolchildren_and.full.time.students..Age.18.and.over) + scale(X.CouncilTaxBandAorB)  + scale(All.Dwellings2011) + scale(population_estimates_2011), data=FireT)

### below values are from CForest variable importance calculation earlier 

frmCFOREST <- c('Average_PTAL_Score2011_.pub.transport.accessibility.', 'Annual_Mean_of_NO2', 'Annual_mean_of_particulate_matter_.2011.', 'Total.Crime.Rate.2011.12', 'X.Employment_and_Support_Allowance_Claimants_.2012.', 'X.Flat_maisonette_apartment','X.Terraced', 'X.CouplehouseW_dependent_Children', 'X.One_person_household', 'X.PrivateRented')

#########################################################################

h <- c('Total.Crime.Rate.2011.12',  'Area_Sq_km', 'X.CouncilTaxBandAorB',  'Annual_Mean_of_NO2', 'X._Domestic_Buildings_.2005.', 'X.of_16._.who_are_schoolchildren_and.full.time.students..Age.18.and.over','All.Dwellings2011')



xx <- FireT[,h[1:length(h)-1]]

Leap2 = regsubsets(x = xx, y = y1, names = h, nbest=10)

##### Relative Importance

# Calculate Relative Importance for Each Predictor  (LASSO)
library(relaimpo)
calc.relimp(LAS,type=c("lmg","last","first","pratt"),
   rela=TRUE)

# Calculate Relative Importance for Each Predictor  (All variables)
library(relaimpo)
calc.relimp(full,type=c("lmg","last","first","pratt"),
   rela=TRUE)


# Bootstrap Measures of Relative Importance (1000 samples) 
boot <- boot.relimp(LAS, b = 1000, type = c("lmg", 
  "last", "first", "pratt"), rank = TRUE, 
  diff = TRUE, rela = TRUE)
booteval.relimp(boot) # print result
plot(booteval.relimp(boot,sort=TRUE)) # plot result


