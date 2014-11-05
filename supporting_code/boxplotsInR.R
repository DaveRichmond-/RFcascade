# install.packages("ggplot2",dep=TRUE)
# install.packages("MASS",dep=TRUE)
# install.packages("pracma",dep=TRUE)
# install.packages("gridExtra",dep=TRUE)

library(ggplot2)
library(MASS)
library(pracma)
library(gridExtra)

setwd('/Users/richmond/Analysis/SomiteTracker/RFs/real_data/on_registered_data/Cascade_w_Smoothing/MBS_AAM_Inference/2init/U=4_PW=4/allDataEveryLevel_fraction=0.33_depth=10/Labels')

diceTable = read.csv("diceScores.txt")

# boxplots

dir.create('Boxplot')
cd('Boxplot')

for (i in 0:21)
{
  dataByClass<-data.frame(diceTable[(diceTable$class == i) ,])
  nam <- paste("d", i, sep = "")
  assign(nam, ggplot(dataByClass, aes(factor(level),diceScore)) + geom_boxplot(aes(fill = factor(level))) + ylim(0,1))
}

png("boxplot.png",height=1200,width=1600)
myPlot <- grid.arrange(d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,ncol=6,nrow=4)
dev.off()


# parallel coord plots

cd('..')
dir.create('PCP')
cd('PCP')

# PCP
for (i in 0:21)
  {
  
  diceData<-data.frame(
    "1stLevel"=c(diceTable[(diceTable$class == i) & (diceTable$level == 0),"diceScore"]),
    "2ndLevel"=c(diceTable[(diceTable$class == i) & (diceTable$level == 1),"diceScore"]),
    "3rdLevel"=c(diceTable[(diceTable$class == i) & (diceTable$level == 2),"diceScore"])
  )
  
  myPlot <- parcoord(diceData, col=rainbow(length(diceData[,1])))
  
  print(myPlot)
  dev.copy(png,strcat(c("class",toString(i),"_PCplot.png")))
  dev.off()
  }
