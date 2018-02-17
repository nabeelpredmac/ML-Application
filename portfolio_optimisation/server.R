library(quantmod)
library(fPortfolio)


if (!require(quantmod)) {
  stop("This app requires the quantmod package. To install it, run 'install.packages(\"quantmod\")'.\n")
}

shinyServer(function(input, output) {
  
  analyse_assets_plot <- function(input){
    par(mfrow=c(1,2))
    symbol_list <-c(input$stock_main1,input$stock_main2,input$stock_main3)
    datam<- NULL
    for(s in symbol_list){
      abc<-getSymbols(s ,auto.assign=FALSE)
      data<-data.frame(abc)
      colnames(data)<- c("Open","High","Low","Close","Volume","Adjusted")
      data$date<-as.Date(row.names(data))
      data$Symbol <- s
      row.names(data) <- c(1:nrow(data))
      datam<- rbind(datam,data)
      
    }
    
    datam$Symbol<- as.factor(datam$Symbol)
    closed<-datam[,c("Symbol","Close","date")] 
    indexes <- reshape(closed,timevar ="Symbol",direction= "wide" ,idvar ="date")
    
    rt <- returns(timeSeries(indexes[,c(-1)],indexes$date))
    # colnames(rt)<- c("Returns.Asset1","Returns.Asset2", "Returns.Asset3")
    
    
    ewSpec <- portfolioSpec()
    nAssets <- ncol(rt)
    setWeights(ewSpec) <- rep(1/nAssets, times = nAssets)
    ewPortfolio <- feasiblePortfolio(data = rt,spec = ewSpec,constraints = "LongOnly")
    
    
    minriskSpec <- portfolioSpec()
    targetReturn <- getTargetReturn(ewPortfolio@portfolio)["mean"]
    setTargetReturn(minriskSpec) <- targetReturn
    minriskPortfolio <- efficientPortfolio(data = rt,spec = minriskSpec,constraints = "LongOnly")
    #plots
    Frontier <- portfolioFrontier(rt, minriskSpec)
    
    weightsPlot(Frontier, mtext = FALSE)
     text <- "Mean-Variance Portfolio - Minimum Risk Constraints"
     mtext(text, side = 3, line = 4, font = 2, cex = 0.9)
    tailoredFrontierPlot(object = Frontier, mText = "MV Portfolio - LongOnly  Constraints",risk = "Cov")
    
  }

  output$plot_main <- renderPlot({ analyse_assets_plot(input) })
})
