"""
CryptoBot — Clase principal del trading bot educativo.

Este módulo contiene la clase CryptoBot que integra:
- Conexión a exchanges via CCXT
- Feature engineering con indicadores técnicos
- Detección de régimen de mercado (GMM)
- Recomendación de estrategias según régimen
- Entrenamiento y comparación de modelos ML
- Backtesting con backtesting.py
- Paper trading en testnet
"""

import warnings
from typing import Optional

import pandas as pd

from .backtesting_ import BacktestMixin
from .config import DEFAULT_SYMBOL, DEFAULT_TIMEFRAME
from .constants import (
    DEFAULT_EXCHANGE,
    DEFAULT_MAX_POSITION_PCT,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
    VALID_TIMEFRAMES,
)
from .data import DataMixin
from .features import FeaturesMixin
from .models import ModelsMixin
from .persistence import PersistenceMixin
from .regime import RegimeMixin
from .scanner import ScannerMixin
from .signals import SignalsMixin
from .trading import TradingMixin
from .visualization import VisualizationMixin

warnings.filterwarnings("ignore")


class CryptoBot(
    DataMixin,
    FeaturesMixin,
    RegimeMixin,
    ModelsMixin,
    SignalsMixin,
    BacktestMixin,
    VisualizationMixin,
    TradingMixin,
    PersistenceMixin,
    ScannerMixin,
):
    """
    Trading bot educativo para criptomonedas.

    Integra data pipeline, feature engineering, detección de régimen,
    entrenamiento de modelos ML, backtesting y paper trading.

    Parameters
    ----------
    symbol : str
        Símbolo de la criptomoneda (e.g., "BTC", "ETH", "SOL").
        Se combina internamente con "/USDT" para CCXT.
    timeframe : str
        Temporalidad de las velas. Opciones: "15m", "30m", "1h", "4h", "1d" (default).
    exchange : str
        Exchange a usar via CCXT (default: "binanceus").
        Verificados desde Google Colab: "binanceus", "kraken",
        "cryptocom", "okx", "coinbase", "bitget", "kucoin",
        "gemini", "bitso", "poloniex".
        ⚠️ No funcionan desde Colab: "binance", "bybit" (geo-block US).
    max_position_pct : float
        Máximo porcentaje del balance por trade (default: 0.10 = 10%).
    stop_loss_pct : float
        Stop loss como porcentaje (default: 0.05 = 5%).
    take_profit_pct : float
        Take profit como porcentaje (default: 0.10 = 10%).

    Attributes
    ----------
    data : pd.DataFrame or None
        OHLCV data. Columnas: Open, High, Low, Close, Volume.
    features : pd.DataFrame or None
        Data con indicadores técnicos agregados.
    regime : str or None
        Régimen de mercado actual ("Bull", "Bear", "Sideways").
    regime_probabilities : dict or None
        Probabilidades de cada régimen.
    selected_strategy : str or None
        Estrategia seleccionada por el usuario.
    model : object or None
        Mejor modelo entrenado.
    model_metrics : dict or None
        Métricas del modelo entrenado.
    signals : pd.Series or None
        Señales generadas: 1 (BUY), -1 (SELL), 0 (HOLD).
    backtest_results : object or None
        Resultados del backtesting.
    trades : list
        Historial de trades ejecutados.

    Examples
    --------
    >>> from cryptobot_lewagon import CryptoBot
    >>> bot = CryptoBot(symbol="BTC")
    >>> bot.fetch_data(last_n=200)
    >>> bot.create_features()
    >>> bot.detect_regime()
    >>> bot.recommend_strategies()
    >>> bot.select_strategy("trend_following")
    >>> bot.train_models()
    >>> bot.get_signals()
    >>> bot.backtest()
    """

    def __init__(
        self,
        symbol: str = DEFAULT_SYMBOL,
        timeframe: str = DEFAULT_TIMEFRAME,
        exchange: str = DEFAULT_EXCHANGE,
        max_position_pct: float = DEFAULT_MAX_POSITION_PCT,
        stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct: float = DEFAULT_TAKE_PROFIT_PCT,
    ):
        # ── Validation ──────────────────────────────────
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Timeframe '{timeframe}' no válido. Opciones: {VALID_TIMEFRAMES}"
            )

        if not (0 < max_position_pct <= 1):
            raise ValueError(
                f"max_position_pct debe estar entre 0 y 1, recibido: {max_position_pct}"
            )
        if not (0 < stop_loss_pct < 1):
            raise ValueError(
                f"stop_loss_pct debe estar entre 0 y 1, recibido: {stop_loss_pct}"
            )
        if not (0 < take_profit_pct < 1):
            raise ValueError(
                f"take_profit_pct debe estar entre 0 y 1, recibido: {take_profit_pct}"
            )

        # ── Config ──────────────────────────────────────
        self.symbol = symbol.upper()
        self._pair = f"{self.symbol}/USDT"
        self.timeframe = timeframe
        self.exchange_id = exchange.lower()

        # ── Risk Management ─────────────────────────────
        self.max_position_pct = max_position_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # ── State (se llenan con los métodos) ───────────
        self.data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.regime: Optional[str] = None
        self.regime_probabilities: Optional[dict] = None
        self.regime_model = None
        self.selected_strategy: Optional[str] = None
        self.model = None
        self.model_name: Optional[str] = None
        self.model_metrics: Optional[dict] = None
        self.model_comparison: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.Series] = None
        self.backtest_results = None
        self._bt_object = None
        self.trades: list = []
        self.pair_data: Optional[pd.DataFrame] = None
        self.pair_symbol: Optional[str] = None
        self.fear_greed_enabled: bool = False

        # ── Temporal Split (train/test) ──────────────────
        self._test_start = None
        self._X_test = None
        self._y_test = None
        self._oos_metrics: Optional[dict] = None

        # ── Exchange Connection ─────────────────────────
        self._exchange = None
        self._testnet_connected = False

        # ── Initialize Exchange (public API only) ───────
        self._init_exchange()

    # ════════════════════════════════════════════════════════════
    #  PRIVATE HELPERS
    # ════════════════════════════════════════════════════════════

    def _init_exchange(self):
        """Inicializa conexión al exchange via CCXT (solo API pública)."""
        import ccxt

        exchange_class = getattr(ccxt, self.exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Exchange '{self.exchange_id}' no soportado por CCXT.")

        self._exchange = exchange_class({"enableRateLimit": True})
        print(f"✅ Conectado a {self.exchange_id.capitalize()} (API pública)")

    @staticmethod
    def supported_exchanges() -> list:
        """Retorna la lista de exchanges soportados por CCXT."""
        import ccxt

        return ccxt.exchanges

    @staticmethod
    def is_exchange_supported(exchange_id: str) -> bool:
        """Verifica si un exchange está soportado por CCXT."""
        import ccxt

        return exchange_id.lower() in ccxt.exchanges

    def _require_data(self):
        """Verifica que fetch_data() fue ejecutado."""
        if self.data is None:
            raise RuntimeError(
                "❌ No hay datos cargados. Ejecuta bot.fetch_data() primero."
            )

    def _require_features(self):
        """Verifica que create_features() fue ejecutado."""
        if self.features is None:
            raise RuntimeError(
                "❌ No hay features creados. Ejecuta bot.create_features() primero."
            )

    def _require_regime(self):
        """Verifica que detect_regime() fue ejecutado."""
        if self.regime is None:
            raise RuntimeError(
                "❌ No hay régimen detectado. Ejecuta bot.detect_regime() primero."
            )

    def _require_strategy(self):
        """Verifica que select_strategy() fue ejecutado."""
        if self.selected_strategy is None:
            raise RuntimeError(
                "❌ No hay estrategia seleccionada. Ejecuta bot.select_strategy() primero."
            )

    def _require_model(self):
        """Verifica que train_models() fue ejecutado."""
        if self.model is None:
            raise RuntimeError(
                "❌ No hay modelo entrenado. Ejecuta bot.train_models() primero."
            )

    def _require_signals(self):
        """Verifica que get_signals() fue ejecutado."""
        if self.signals is None:
            raise RuntimeError(
                "❌ No hay señales generadas. Ejecuta bot.get_signals() primero."
            )

    def _require_pair_data(self):
        """Verifica que fetch_data() fue ejecutado con pair_symbol."""
        if self.pair_data is None:
            raise RuntimeError(
                "❌ No hay datos del par secundario. "
                "Ejecuta bot.fetch_data(pair_symbol='ETH') primero."
            )

    def _require_testnet(self):
        """Verifica que connect_testnet() fue ejecutado."""
        if not self._testnet_connected:
            raise RuntimeError(
                "❌ No conectado al testnet. Ejecuta bot.connect_testnet() primero."
            )

    # ════════════════════════════════════════════════════════════
    #  DISPLAY
    # ════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        """Representación del bot para print/display."""
        status_parts = [
            f"CryptoBot('{self.symbol}', {self.timeframe}, {self.exchange_id})",
            f"  Data:     {'✅ ' + str(len(self.data)) + ' registros' if self.data is not None else '❌ No cargada'}",
            f"  Features: {'✅ ' + str(len(self.features.columns)) + ' columnas' if self.features is not None else '❌ No creados'}",
            f"  Régimen:  {self.regime if self.regime else '❌ No detectado'}",
            f"  Strategy: {self.selected_strategy if self.selected_strategy else '❌ No seleccionada'}",
            f"  Modelo:   {self.model_name if self.model_name else '❌ No entrenado'}",
            f"  FGI:      {'✅ Activado' if self.fear_greed_enabled else '❌ No activado'}",
            f"  Señales:  {'✅' if self.signals is not None else '❌ No generadas'}",
            f"  Testnet:  {'✅ Conectado' if self._testnet_connected else '❌ No conectado'}",
            f"  Trades:   {len(self.trades)} registrados",
        ]
        return "\n".join(status_parts)

