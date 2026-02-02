######################################################
############## Modelling Time Volatility #############
# Comparison Between ARIMA And ARIMA-GARCH Modelling #
######################################################

#1, Install Packages
install.packages("rugarch")
install.packages("xts")
library(dplyr)
library(lubridate)
library(rugarch)
library(forecast)
library(xts)
library(tseries)
library(ggplot2)
library(patchwork) #patch plot in same layout for ggplot

#====================================================================#
#2. Load Data

oil <- read.csv("Brent Oil Futures Historical Data.csv")

str(oil)

#====================================================================#
#3. Pre-processing

#3(i) Change Data Type

oil['Date'] <- mdy(oil[,1])

#3(ii) Maintain Relevant Columns
oil <- oil[c('Date','Price')]

#3(iii) Transform Into Time Series Data Using XTS library
oil_xts <- xts(oil$Price, order.by = as.Date(oil$Date))

colnames(oil_xts) <- "Price"
head(oil_xts)
str(oil_xts)
# 1032 rows
# start: 3/1/22
# end: 31/12/25

#====================================================================#
#4. Splitting Dataset

# Determine the split point (e.g., 80% for training)
n <- nrow(oil_xts)
train_size <- round(n * 0.70)

# Split using indexing
train_data <- oil_xts[1:train_size, ]
test_data  <- oil_xts[(train_size + 1):n, ]

#====================================================================#
#5(i) Visualize Train-Test Data Set

windows(10,10)

# Create a plot for entire dataset
plot(oil_xts, col = "white", main = "Brentt Oil Price: Train and Test Data")

# Add the training data (Blue)
lines(train_data, col = "darkgreen", lwd = 2)

# Add the test data (Red)
lines(test_data, col = "darkred", lwd = 2)

# Add legend
addLegend("topright", on=1,
          legend.names = c("Train", "Test"),
          lty = 1, lwd = 3,
          col=c("darkgreen", "darkred"))

#5(ii) Decompose Train Data Set

# Convert the numeric values of train_data to a ts object
# Use frequency = 260 for business days
train_ts <- ts(as.numeric(train_data), frequency = 260)

# Perform the decomposition
decomp_mult <- decompose(train_ts, type = "multiplicative")

# Plot the components (Observed, Trend, Seasonal, and Random)
windows(10,10)
plot(decomp_mult,
     col = "blue")

# The trend is decreasing over time.
# There is clear seasonal pattern repeats roughly annually.
# The remaining irregular fluctuations after removing trend
# and seasonal effects, representing noise or unexplained variation.

#====================================================================#
#6 Stationarity Check

#6(i). Through ACF & PACF Plot
windows(10,10)
par(mfrow = c(2,1))

# ACF
acf(train_data,
    lag.max = 40,
    main = "Autocorrelation Function (ACF)")

# PACF
pacf(train_data,
     lag.max = 40,
     main = "Partial Autocorrelation Function (PACF)")
# Based on the ACF plot, values decrease very slowly,
# lingering long past the confidence interval, indicating a strong trend
# Thus, data is non-stationary

#--------------------------------------------------------------------------#

#6(ii). Through ADF Test

# If p-value is less than 0.05, it means times series is Stationary
# Based on the following hypothesis
# H0:Time Series is not stationary
# H10:Time Series is stationary
adf_result <- adf.test(train_data, alternative = "stationary")

print(adf_result)
# ADF p-value: 0.2497
# Since the p-value is > 0.05, thus fail to reject H0:Time Series is not stationary
# So the oil price is non-stationary

#--------------------------------------------------------------------------#

#6(iii). Differencing Non-stationarity 
train_diff <- na.omit(diff(train_data))

windows(10,10)
plot(train_diff)

#--------------------------------------------------------------------------#

#6(iv). Re-check again through ACF & PACF Plot
windows(10,10)
par(mfrow = c(2,1))

# ACF
acf(train_diff,
    main = "Autocorrelation Function (ACF)")
# MA(5)

# PACF
pacf(train_diff,
     main = "Partial Autocorrelation Function (PACF)")
# AR(5)

# After performed First-Order Differencing, the time series data seems stationary
# ACF bars drop quickly to within the confidence interval) after at Lag 1

#--------------------------------------------------------------------------#

#6(v). Re-check again through ADF Test

# If p-value is less than 0.05, it means times series is Stationary
# Based on the following hypothesis
# H0:Time Series is not stationary
# H10:Time Series is stationary
adf.test(train_diff)
# ADF p-value: 0.01
# Since the p-value is < 0.05, thus reject H0:Time Series is not stationary
# So the oil price is stationary

#====================================================================#
#7. Volatility Check

# Volatility clustering means large changes tend to be followed by large changes, and small changes by small changes

# Create Squared Differences based on output of previous differencing

sq_diff = train_diff**2

windows(10,10)
par(mfrow = c(2,1))
# Plot ACF : SQUARED differences
acf(sq_diff,
    main="Autocorrelation (ACF) - Squared Difference")

# Plot PACF : SQUARED differences
pacf(sq_diff,
     main="Partial Autocorrelation (PACF) - Squared Difference")

# Based on the ACF and PACF plot of Squared Difference,
# Most of the lags are outside the confidence interval.
# Thus, A GARCH (Generalized Autoregressive Conditional Heteroskedasticity) model is required.
# The significant autocorrelation in the Squared Differences indicate volatility clustering exist in the oil price data set.
# It shows that the size of price change today is correleted with the size of the price change yesterday.
# This volatility shows data is heteroskedastic (not constant) which is suitable to be used with the GARCH model.

#====================================================================#
#8. ARIMA Modelling

arima.manual.model <- Arima(train_data, order = c(5,1,5))

arima.auto.model <- auto.arima(train_data, 
                          seasonal = FALSE, 
                          stepwise = FALSE, 
                          approximation = FALSE)

# Comparing Arima model
summary(arima.manual.model)
# ARIMA(5,1,5)
# AIC = 3141.64
# BIC = 3192.03
# MAE = 1.497527
# RMSE = 2.095965

summary(arima.auto.model)
# ARIMA(0,1,5)
# AIC = 3153.73
# BIC = 3181.21
# MAE = 1.51698
# RMSE = 2.136068

arima.comparison <- data.frame(
  Model = c("Manual ARIMA(5,1,5)", "Auto ARIMA (0,1,5)"),
  AIC  = c(AIC(arima.manual.model), AIC(arima.auto.model)),
  BIC  = c(BIC(arima.manual.model), BIC(arima.auto.model)),
  MAE = c(1.497527, 1.51698),
  RMSE = c(2.095965, 2.136068)
)

arima.comparison <- t(arima.comparison)
colnames(arima.comparison) <- as.character(unlist(arima.comparison[1, ]))
arima.comparison <- arima.comparison[-1, ]

#--------------------------------------------------------------------------#
# Forecast

# h should be the number of observations in your test_data
h_period <- nrow(test_data)

# Generate forecast
arima_fc <- forecast(arima.manual.model, h = h_period)

# Extract predicted prices (the 'Point Forecast' column)
arima_pred <- as.numeric(arima_fc$mean)

#--------------------------------------------------------------------------#
# Calculate MAE, RMSE
# Compare the forecast object to your actual test values
arima_accuracy_metrics <- accuracy(arima_pred, as.numeric(test_data))

# Print the metrics
print(arima_accuracy_metrics)
# MAE = 6.659535
# RMSE = 7.961321

#--------------------------------------------------------------------------#
# Create Dataframe to Store Arima Modelling data

arima_df <- data.frame(
  date = index(test_data),
  actual = as.numeric(test_data),
  forecast = as.numeric(arima_fc$mean),
  lower_ci = as.numeric(arima_fc$lower[,2]), # 95% CI
  upper_ci = as.numeric(arima_fc$upper[,2]) # 95% CI
  )

#--------------------------------------------------------------------------#
# Plot the forecast
windows(10,10)
ggplot(arima_df, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = forecast, color = "ARIMA Forecast"), size = 1) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.1) +
  labs(title = "Brent Oil Price: ARIMA Forecast vs Actual",
       x = "Date",
       y = "Price",
       color = "Legend") +
  theme_minimal() +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months")

#====================================================================#
#8. ARIMA-GARCH Modelling

# Define the specification
spec <- ugarchspec(
  # 1. Variance Equation: GARCH(1,1)
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  
  # 2. Mean Equation: ARMA(0,5) (Assuming d=1 was handled by differencing)
  mean.model = list(armaOrder = c(5, 5), include.mean = F),
  # Not include mean because to avoid the mean being drift since the trend of the data is downward 
  
  # 3. Distribution: 'std' (Student's t) is better for oil prices than 'norm'
  distribution.model = "std" 
)

# Fit the model to your differenced training data
arima_garch.model <- ugarchfit(spec = spec, data = train_diff)

# View the results
print(arima_garch.model)

#--------------------------------------------------------------------------#
# Forecast

# Generate forecast
arima_garch_fc <- ugarchforecast(arima_garch.model, n.ahead = h_period)

# Extract the predicted means (price)
garch_returns_pred <- as.numeric(fitted(arima_garch_fc))

# Convert returns to price level (Re-integration)
# Last known price from training data
last_price <- as.numeric(tail(train_data, 1))
# Use cumsum (cumulative sum) to add returns to the last training price
garch_price_pred <- last_price + cumsum(garch_returns_pred)

# Extract the predicted volatility (variance)
garch_sigma <- as.numeric(sigma(arima_garch_fc))

# Compound the variance (Cumulative Sum of Variances)
# Variance is sigma squared. We sum them up to the forecast horizon.
cum_variance <- cumsum(garch_sigma^2)
cum_sigma_price <- sqrt(cum_variance)

# Calculate 95% intervals using the t-distribution quantile
shape_param <- coef(arima_garch.model)["shape"]
t_crit <- qdist("std", p = 0.975, shape = shape_param)

upper_ci <- garch_price_pred + (t_crit * cum_sigma_price)
lower_ci <- garch_price_pred - (t_crit * cum_sigma_price)

#--------------------------------------------------------------------------#
# Calculate MAE, RMSE

# Compare the forecast object to your actual test values
arima_garch_accuracy_metrics <- accuracy(garch_price_pred, as.numeric(test_data))

# Print the metrics
print(arima_garch_accuracy_metrics)
# MAE = 6.429104
# RMSE = 7.719397

#--------------------------------------------------------------------------#
# Create Dataframe to Store Arima Modelling data

arima_garch_df <- data.frame(
  date = index(test_data),
  actual = as.numeric(test_data),
  forecast = as.numeric(garch_price_pred),
  lower_ci = as.numeric(lower_ci), # 95% CI
  upper_ci = as.numeric(upper_ci) # 95% CI
)

#--------------------------------------------------------------------------#
# Plot the forecast

windows(10,10)
ggplot(arima_garch_df, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = forecast, color = "ARIMA Forecast"), size = 1) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.1) +
  labs(title = "Brent Oil Price: ARIMA-GARCH Forecast vs Actual",
       x = "Date",
       y = "Price",
       color = "Legend") +
  theme_minimal() +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months")

#====================================================================#
#9. Comparing ARIMA Model & ARIMA_GARCH Model

#9(i). MAE & RMSE

comparison_table <- data.frame(
  `ARIMA(5,1,5)` = arima_accuracy_metrics[,c(2,3)],
  `ARIMA(5,1,5)-GARCH(1,1)` = arima_garch_accuracy_metrics[,c(2,3)],
  check.names = FALSE)

#9(ii). Visualisation

# Define the Y-axis range and breaks
# We find the min/max across both models to ensure the plots align
all_values <- c(arima_df$actual, arima_df$lower_ci, arima_df$upper_ci, 
                arima_garch_df$lower_ci, arima_garch_df$upper_ci)

y_min <- floor(min(all_values, na.rm = TRUE) / 10) * 10
y_max <- ceiling(max(all_values, na.rm = TRUE) / 10) * 10
y_breaks <- seq(y_min, y_max, by = 10)

# ARIMA Model
arima_plot <- ggplot(arima_df, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = forecast, color = "Forecast"), size = 1) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.15) +
  labs(title = "Brent Oil Price: ARIMA (5,1,5) Forecast vs Actual",
       x = NULL,
       y = "Price") +
  scale_y_continuous(breaks = y_breaks, limits = c(y_min, y_max)) +
  scale_color_manual(values = c("Actual" = "black", "Forecast" = "red")) +
  theme_minimal() +
  theme(axis.text.x = element_blank())

# ARIMA-GARCH Model
arima_garch_plot <- ggplot(arima_garch_df, aes(x = date)) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = forecast, color = "Forecast"), size = 1) +
  geom_ribbon(aes(ymin = lower_ci, ymax = upper_ci), fill = "blue", alpha = 0.15) +
  labs(title = "Brent Oil Price: ARIMA(5,1,5)-GARCH(1,1) Forecast vs Actual",
       x = "Date",
       y = "Price") +
  scale_y_continuous(breaks = y_breaks, limits = c(y_min, y_max)) +
  scale_color_manual(values = c("Actual" = "black", "Forecast" = "red")) +
  theme_minimal() +
  scale_x_date(date_labels = "%b %Y", date_breaks = "3 months")

# Combine plot
windows(10,10)
arima_plot / arima_garch_plot + 
  plot_layout(guides = "collect") & theme(legend.position = "bottom")

