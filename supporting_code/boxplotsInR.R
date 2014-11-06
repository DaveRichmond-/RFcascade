
args <- commandArgs(trailingOnly = TRUE)

pkgs <- c("ggplot2","MASS","pracma","gridExtra")

for(x in pkgs){
  if(!is.element(x, installed.packages()[,1]))
    {install.packages(x, dep=TRUE, repos="http://cran.fhcrc.org")
  } else {print(paste(x, " library already installed"))}
}

library(ggplot2)
library(MASS)
library(pracma)
library(gridExtra)

setwd(args[1])

diceTable = read.csv("diceScores.txt")

# boxplots and parallel coord plots

dir.create('Boxplot')
cd('Boxplot')

for (i in 0:21)
{
  dataByClass<-data.frame(diceTable[(diceTable$class == i) ,])
  nam <- paste("d", i, sep = "")
  assign(nam, ggplot(dataByClass, aes(factor(level),diceScore)) + geom_boxplot(aes(fill = factor(level))) + geom_line(aes(group=image), colour="gray") + ylim(0,1))
}

png("boxplot.png",height=1200,width=1600)
myPlot <- grid.arrange(d0,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,ncol=6,nrow=4)
dev.off()
