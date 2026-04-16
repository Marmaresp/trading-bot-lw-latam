from cryptobot import CryptoBot
import requests

bot = CryptoBot(
    symbol="SOL",
    timeframe="1h",
    exchange="binanceus",
    max_position_pct=0.10,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
)

symbols = [s["symbol"] for s in requests.get("https://api.binance.us/api/v3/exchangeInfo").json()["symbols"]]
print(symbols[:10])

bot.fetch_data(last_n=500)

bot.summary()
bot.plot_price()

# Features solo de momentum_select
bot.create_features(mode="momentum_select")

# Detectar régimen solo como referencia
bot.detect_regime()
bot.regime_report()

# Fijar estrategia compatible con esas features
bot.select_strategy("momentum")
print("selected_strategy:", bot.selected_strategy)

# Entrenar modelo
bot.train_models()
bot.optimize_model()
bot.feature_importance()
bot.plot_feature_importance()

# Generar señales
bot.get_signals(confidence_threshold=0.0)
bot.plot_signals()

# Exportar resultados
df_export = bot.data[["Open", "High", "Low", "Close"]].copy()

if bot.signals is not None:
    df_export["signals"] = bot.signals.reindex(df_export.index).values

df_export["selected_strategy"] = bot.selected_strategy
df_export["regime"] = bot.regime
df_export["model_name"] = bot.model_name


accuracy = None
if bot.model_metrics is not None:
    accuracy = (
        bot.model_metrics.get("accuracy")
        or bot.model_metrics.get("Accuracy")
        or bot.model_metrics.get("test_accuracy")
        or bot.model_metrics.get("oos_accuracy")
    )

df_export["accuracy"] = accuracy

df_export = df_export.reset_index()
df_export.to_csv("result_OHC.csv", index=False)

# Backtesting
bot.backtest()
bot.backtest_plot()
bot.plot_performance()