# app_simple.R - Financial Time Series Analysis (USD Only)
library(shiny)
library(plotly)
library(quantmod)
library(rugarch)
library(DT)

ui <- fluidPage(
  titlePanel("Financial Time Series Analysis (USD Only)"),
  
  sidebarLayout(
    sidebarPanel(
      width = 3,
      
      # Stock selection
      textInput(
        "selected_stocks",
        h4("Ticker Symbols (comma-separated)"),
        value = "AAPL,MSFT,SHEL.L",
        placeholder = "AAPL, MSFT, SHEL.L, 005930.KS"
      ),
      
      helpText("All prices will be shown in USD"),
      helpText("Supports US and international tickers"),
      
      # Date range
      dateRangeInput(
        "date_range",
        h4("Analysis Period"),
        start = "2020-01-01",
        end = Sys.Date()
      ),
      
      tags$hr(),
      
      # Refresh button
      actionButton("refresh_data", "Load/Refresh Data", 
                   icon = icon("sync"),
                   style = "width: 100%; margin-bottom: 10px;"),
      
      tags$hr(),
      
      # GARCH settings
      h4("GARCH Model"),
      
      selectInput(
        "garch_model",
        h4("Model Type"),
        choices = c(
          "sGARCH" = "sGARCH",
          "ARCH" = "ARCH", 
          "eGARCH" = "eGARCH",
          "gjrGARCH" = "gjrGARCH",
          "apARCH" = "apARCH",
          "iGARCH" = "iGARCH"
        ),
        selected = "sGARCH"
      ),
      
      numericInput(
        "arch_order",
        h4("ARCH Order (q)"),
        value = 1,
        min = 1,
        max = 3
      ),
      
      numericInput(
        "garch_order",
        h4("GARCH Order (p)"),
        value = 1,
        min = 1,
        max = 3
      ),
      
      selectInput(
        "distribution",
        h4("Distribution"),
        choices = c(
          "Normal" = "norm",
          "Student's t" = "std",
          "Skewed t" = "sstd",
          "GED" = "ged"
        ),
        selected = "std"
      ),
      
      br(),
      actionButton("run_garch", "Run GARCH Analysis (First Ticker)", 
                   style = "width: 100%; font-weight: bold;"),
      
      br(), br(),
      downloadButton("download_data", "Download Results", 
                     style = "width: 100%;")
    ),
    
    mainPanel(
      width = 9,
      tabsetPanel(
        tabPanel("Prices",
                 h4("Closing Prices (USD)"),
                 plotlyOutput("price_plot", height = "400px"),
                 DTOutput("price_table")),
        
        tabPanel("Returns",
                 h4("Daily Returns (USD)"),
                 plotlyOutput("returns_plot", height = "400px"),
                 DTOutput("returns_table")),
        
        tabPanel("Statistics",
                 h4("Descriptive Statistics"),
                 DTOutput("stats_table"),
                 plotlyOutput("hist_plot", height = "300px")),
        
        tabPanel("GARCH",
                 h4(textOutput("garch_title")),
                 DTOutput("garch_table"),
                 plotlyOutput("volatility_plot", height = "300px"),
                 plotOutput("diagnostics_plot", height = "300px")),
        
        tabPanel("Risk",
                 h4("Risk Metrics"),
                 DTOutput("risk_table"),
                 plotlyOutput("var_plot", height = "300px"))
      )
    )
  )
)

server <- function(input, output, session) {
  
  # Reactive values
  stock_data <- reactiveVal(NULL)
  garch_results <- reactiveVal(NULL)
  
  # Load stock data
  observeEvent(input$refresh_data, {
    req(input$selected_stocks)
    
    tickers <- trimws(unlist(strsplit(input$selected_stocks, ",")))
    tickers <- tickers[tickers != ""]
    
    if (length(tickers) == 0) {
      showNotification("Please enter tickers", type = "error")
      return()
    }
    
    # Limit
    if (length(tickers) > 5) {
      showNotification("Using first 5 tickers", type = "warning")
      tickers <- tickers[1:5]
    }
    
    data_list <- list()
    
    withProgress(message = "Loading data...", value = 0, {
      for (i in seq_along(tickers)) {
        ticker <- tickers[i]
        
        incProgress(1/length(tickers), detail = ticker)
        
        tryCatch({
          stock <- getSymbols(
            ticker,
            src = "yahoo",
            from = input$date_range[1],
            to = input$date_range[2],
            auto.assign = FALSE
          )
          
          # Get prices in USD (Yahoo already converts for most stocks)
          prices_usd <- Cl(stock)
          colnames(prices_usd) <- ticker
          
          returns_usd <- diff(log(prices_usd))[-1]
          colnames(returns_usd) <- ticker
          
          data_list[[ticker]] <- list(
            prices = prices_usd,
            returns = returns_usd
          )
          
        }, error = function(e) {
          showNotification(paste("Failed:", ticker), type = "warning")
        })
      }
    })
    
    if (length(data_list) > 0) {
      stock_data(data_list)
      showNotification(paste("Loaded", length(data_list), "ticker(s)"), type = "message")
    }
  })
  
  # GARCH title
  output$garch_title <- renderText({
    req(garch_results())
    paste("GARCH Results for", garch_results()$ticker)
  })
  
  # Price plot
  output$price_plot <- renderPlotly({
    req(stock_data())
    
    plot_data <- data.frame()
    
    for (ticker in names(stock_data())) {
      prices <- stock_data()[[ticker]]$prices
      temp_df <- data.frame(
        Date = index(prices),
        Price = as.numeric(prices),
        Ticker = ticker
      )
      plot_data <- rbind(plot_data, temp_df)
    }
    
    plot_ly(plot_data, x = ~Date, y = ~Price, color = ~Ticker,
            type = 'scatter', mode = 'lines') %>%
      layout(xaxis = list(title = "Date"),
             yaxis = list(title = "Price (USD)"),
             hovermode = 'x unified')
  })
  
  # Price table
  output$price_table <- renderDT({
    req(stock_data())
    
    summary_list <- list()
    
    for (ticker in names(stock_data())) {
      prices <- as.numeric(stock_data()[[ticker]]$prices)
      
      summary_list[[ticker]] <- data.frame(
        Ticker = ticker,
        Start_USD = round(prices[1], 2),
        End_USD = round(prices[length(prices)], 2),
        Min_USD = round(min(prices), 2),
        Max_USD = round(max(prices), 2),
        Return_Pct = round((prices[length(prices)] / prices[1] - 1) * 100, 1),
        Days = length(prices)
      )
    }
    
    datatable(do.call(rbind, summary_list), 
              options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Returns plot
  output$returns_plot <- renderPlotly({
    req(stock_data())
    
    plot_data <- data.frame()
    
    for (ticker in names(stock_data())) {
      returns <- stock_data()[[ticker]]$returns
      temp_df <- data.frame(
        Date = index(returns),
        Return = 100 * as.numeric(returns),
        Ticker = ticker
      )
      plot_data <- rbind(plot_data, temp_df)
    }
    
    plot_ly(plot_data, x = ~Date, y = ~Return, color = ~Ticker,
            type = 'scatter', mode = 'lines') %>%
      layout(xaxis = list(title = "Date"),
             yaxis = list(title = "Return (%)"),
             hovermode = 'x unified')
  })
  
  # Returns table
  output$returns_table <- renderDT({
    req(stock_data())
    
    summary_list <- list()
    
    for (ticker in names(stock_data())) {
      returns <- 100 * as.numeric(stock_data()[[ticker]]$returns)
      
      summary_list[[ticker]] <- data.frame(
        Ticker = ticker,
        Mean_Pct = round(mean(returns, na.rm = TRUE), 3),
        SD_Pct = round(sd(returns, na.rm = TRUE), 3),
        Min_Pct = round(min(returns, na.rm = TRUE), 3),
        Max_Pct = round(max(returns, na.rm = TRUE), 3),
        Days = length(returns)
      )
    }
    
    datatable(do.call(rbind, summary_list), 
              options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Statistics table
  output$stats_table <- renderDT({
    req(stock_data())
    
    stats_list <- list()
    
    for (ticker in names(stock_data())) {
      returns <- 100 * as.numeric(stock_data()[[ticker]]$returns)
      
      stats_list[[ticker]] <- data.frame(
        Ticker = ticker,
        Mean = round(mean(returns, na.rm = TRUE), 4),
        SD = round(sd(returns, na.rm = TRUE), 4),
        Skewness = round(e1071::skewness(returns, na.rm = TRUE), 4),
        Kurtosis = round(e1071::kurtosis(returns, na.rm = TRUE) - 3, 4),
        Days = length(returns)
      )
    }
    
    datatable(do.call(rbind, stats_list), 
              options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # Histogram
  output$hist_plot <- renderPlotly({
    req(stock_data())
    
    plot_data <- data.frame()
    
    for (ticker in names(stock_data())) {
      returns <- 100 * as.numeric(stock_data()[[ticker]]$returns)
      temp_df <- data.frame(
        Return = returns,
        Ticker = ticker
      )
      plot_data <- rbind(plot_data, temp_df)
    }
    
    plot_ly(plot_data, x = ~Return, color = ~Ticker, type = "histogram",
            opacity = 0.6, nbinsx = 30) %>%
      layout(xaxis = list(title = "Return (%)"),
             yaxis = list(title = "Frequency"),
             barmode = "overlay")
  })
  
  # GARCH analysis
  observeEvent(input$run_garch, {
    req(stock_data())
    
    tickers <- names(stock_data())
    if (length(tickers) == 0) return()
    
    ticker <- tickers[1]
    returns <- as.numeric(stock_data()[[ticker]]$returns)
    
    if (length(returns) < 50) {
      showNotification("Need at least 50 observations", type = "warning")
      return()
    }
    
    withProgress(message = "Fitting GARCH...", value = 0.5, {
      
      tryCatch({
        # Adjust for ARCH model
        garch_order_val <- input$garch_order
        if (input$garch_model == "ARCH") {
          garch_order_val <- 0
        }
        
        spec <- ugarchspec(
          variance.model = list(
            model = input$garch_model,
            garchOrder = c(input$arch_order, garch_order_val)
          ),
          mean.model = list(armaOrder = c(0, 0)),
          distribution.model = input$distribution
        )
        
        fit <- ugarchfit(spec, data = returns, solver = "hybrid")
        
        results <- list(
          fit = fit,
          sigma = sigma(fit),
          residuals = residuals(fit, standardize = TRUE),
          ticker = ticker
        )
        
        garch_results(results)
        showNotification("GARCH model fitted", type = "message")
        
      }, error = function(e) {
        showNotification(paste("GARCH failed:", e$message), type = "error")
      })
    })
  })
  
  # GARCH table
  output$garch_table <- renderDT({
    req(garch_results())
    
    fit <- garch_results()$fit
    
    coefs <- coef(fit)
    result_df <- data.frame(
      Parameter = names(coefs),
      Estimate = round(coefs, 6),
      Std.Error = round(sqrt(diag(vcov(fit))), 6)
    )
    
    if (!is.null(infocriteria(fit))) {
      info_criteria <- infocriteria(fit)
      result_df <- rbind(result_df,
                         data.frame(Parameter = "AIC", Estimate = round(info_criteria[1], 4), Std.Error = NA),
                         data.frame(Parameter = "BIC", Estimate = round(info_criteria[2], 4), Std.Error = NA)
      )
    }
    
    datatable(result_df, options = list(pageLength = 15, scrollX = TRUE))
  })
  
  # Volatility plot
  output$volatility_plot <- renderPlotly({
    req(garch_results())
    
    results <- garch_results()
    ticker_data <- stock_data()[[results$ticker]]
    dates <- index(ticker_data$returns)
    
    n_obs <- length(results$sigma)
    if (length(dates) > n_obs) {
      dates <- tail(dates, n_obs)
    }
    
    plot_data <- data.frame(
      Date = dates,
      Volatility = 100 * results$sigma
    )
    
    plot_ly(plot_data, x = ~Date, y = ~Volatility,
            type = 'scatter', mode = 'lines',
            line = list(color = 'blue', width = 2)) %>%
      layout(xaxis = list(title = "Date"),
             yaxis = list(title = "Volatility (%)"),
             hovermode = 'x unified')
  })
  
  # Diagnostics plot
  output$diagnostics_plot <- renderPlot({
    req(garch_results())
    
    results <- garch_results()
    residuals <- as.numeric(results$residuals)
    
    par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
    
    acf(residuals^2, main = "ACF of Squared Residuals",
        lag.max = 20, ylab = "ACF", xlab = "Lag")
    grid()
    
    qqnorm(residuals, main = "QQ Plot of Residuals")
    qqline(residuals, col = "red")
    grid()
    
    par(mfrow = c(1, 1))
  })
  
  # Risk table
  output$risk_table <- renderDT({
    req(stock_data())
    
    risk_list <- list()
    
    for (ticker in names(stock_data())) {
      returns <- 100 * as.numeric(stock_data()[[ticker]]$returns)
      
      risk_list[[ticker]] <- data.frame(
        Ticker = ticker,
        Annual_Vol_Pct = round(sd(returns, na.rm = TRUE) * sqrt(252), 3),
        Sharpe = round(mean(returns, na.rm = TRUE) / sd(returns, na.rm = TRUE) * sqrt(252), 3),
        VaR_5_Pct = round(quantile(returns, 0.05, na.rm = TRUE), 3),
        CVaR_5_Pct = round(mean(returns[returns <= quantile(returns, 0.05, na.rm = TRUE)], na.rm = TRUE), 3),
        Days = length(returns)
      )
    }
    
    datatable(do.call(rbind, risk_list), 
              options = list(pageLength = 10, scrollX = TRUE))
  })
  
  # VaR plot
  output$var_plot <- renderPlotly({
    req(stock_data())
    
    plot_data <- data.frame()
    
    for (ticker in names(stock_data())) {
      returns <- 100 * as.numeric(stock_data()[[ticker]]$returns)
      var_5 <- quantile(returns, 0.05, na.rm = TRUE)
      
      temp_df <- data.frame(
        Return = returns,
        Ticker = ticker,
        VaR = var_5
      )
      plot_data <- rbind(plot_data, temp_df)
    }
    
    p <- plot_ly()
    
    for (ticker in unique(plot_data$Ticker)) {
      ticker_data <- plot_data[plot_data$Ticker == ticker, ]
      var_line <- unique(ticker_data$VaR)
      
      p <- p %>% add_trace(
        x = ticker_data$Return,
        type = "histogram",
        name = ticker,
        opacity = 0.5,
        nbinsx = 30
      ) %>%
        add_trace(
          x = c(var_line, var_line),
          y = c(0, 50),
          type = "scatter",
          mode = "lines",
          name = paste(ticker, "VaR"),
          line = list(dash = "dash", width = 2),
          showlegend = FALSE
        )
    }
    
    p %>% layout(
      xaxis = list(title = "Return (%)"),
      yaxis = list(title = "Frequency"),
      barmode = "overlay"
    )
  })
  
  # Download handler
  output$download_data <- downloadHandler(
    filename = function() {
      paste("financial_data_", Sys.Date(), ".csv", sep = "")
    },
    content = function(file) {
      req(stock_data())
      
      all_data <- data.frame()
      
      for (ticker in names(stock_data())) {
        data <- stock_data()[[ticker]]
        prices <- data$prices
        returns <- data$returns
        
        temp_df <- data.frame(
          Date = index(prices),
          Ticker = ticker,
          Price_USD = as.numeric(prices),
          Return_Pct = NA
        )
        
        # Match returns to dates
        return_dates <- index(returns)
        price_dates <- index(prices)
        for (i in seq_along(return_dates)) {
          idx <- which(price_dates == return_dates[i])
          if (length(idx) > 0) {
            temp_df$Return_Pct[idx] <- 100 * as.numeric(returns)[i]
          }
        }
        
        all_data <- rbind(all_data, temp_df)
      }
      
      write.csv(all_data, file, row.names = FALSE)
    }
  )
  
  # Observer for ARCH model
  observe({
    if (input$garch_model == "ARCH") {
      updateNumericInput(session, "garch_order", value = 0, min = 0, max = 0)
    } else {
      updateNumericInput(session, "garch_order", value = 1, min = 1, max = 3)
    }
  })
}

shinyApp(ui = ui, server = server)

