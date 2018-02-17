
library(quantmod) 
symbols <- stockSymbols()
shinyUI(
  pageWithSidebar(
  headerPanel("Asset Analyser"),
  sidebarPanel(
    selectInput('stock_main1', 'Asset 1', choices = symbols,selected="CETC", multiple=FALSE, selectize=TRUE),
    selectInput('stock_main2', 'Asset 2', choices = symbols,selected="GTT", multiple=FALSE, selectize=TRUE),
    selectInput('stock_main3', 'Asset 3', choices = symbols,selected="UEIC", multiple=FALSE, selectize=TRUE),
    width=2 ),


  mainPanel(plotOutput(outputId ="plot_main"))
) )
