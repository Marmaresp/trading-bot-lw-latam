"""Mixin para feature engineering: indicadores técnicos."""

import ta
import pandas as pd

from .config import ATR_PERIOD, RSI_PERIOD, SMA_FAST_WINDOW, SMA_SLOW_WINDOW, VOLATILITY_WINDOW


class FeaturesMixin:
    """Métodos de feature engineering: create_features()."""

    def create_features(self, mode: str = "core") -> "FeaturesMixin":
        """
        Agrega indicadores técnicos al DataFrame.

        Dos modos disponibles:
        - "core": ~10 indicadores clave que se cubrieron en el curso.
          Ideal para exploración y comprensión.
        - "full": 86+ indicadores via ta.add_all_ta_features().
          Ideal para training de modelos donde feature importance
          determina qué indicadores son relevantes.

        Parameters
        ----------
        mode : str, default "core"
            "core" para indicadores esenciales, "full" para todos.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Core Features
        -------------
        - SMA_20, SMA_50 : Medias móviles simples
        - RSI_14 : Relative Strength Index
        - MACD, MACD_signal : Moving Average Convergence Divergence
        - BB_upper, BB_lower : Bollinger Bands
        - ATR_14 : Average True Range (volatilidad)
        - volume_change : Cambio porcentual del volumen
        - returns : Retorno porcentual diario
        - volatility_20 : Volatilidad rolling 20 períodos

        Full Features (mode="full")
        ---------------------------
        Generados por ta.add_all_ta_features() + returns y volatility_20.

        Momentum (18):
        - momentum_ao : Awesome Oscillator — diferencia de medias móviles de precios medios
        - momentum_kama : Kaufman Adaptive Moving Average — media móvil adaptativa al ruido 
        - momentum_ppo : Percentage Price Oscillator — diferencia % entre dos EMAs de precio 
        - momentum_ppo_hist : PPO Histogram — diferencia entre PPO y su señal 
        - momentum_ppo_signal : PPO Signal — EMA de la línea PPO 
        - momentum_pvo : Percentage Volume Oscillator — diferencia % entre dos EMAs de volumen 
        - momentum_pvo_hist : PVO Histogram — diferencia entre PVO y su señal
        - momentum_pvo_signal : PVO Signal — EMA de la línea PVO 
        - momentum_roc : Rate of Change — cambio % del precio respecto a n períodos atrás 
        - momentum_rsi : Relative Strength Index — sobrecompra/sobreventa (0-100) 
        - momentum_stoch : Stochastic Oscillator %K — posición del cierre en el rango alto-bajo
        - momentum_stoch_signal : Stochastic %D — media móvil de %K
        - momentum_stoch_rsi : Stochastic RSI — RSI aplicado al RSI
        - momentum_stoch_rsi_k : StochRSI %K — media suavizada del StochRSI
        - momentum_stoch_rsi_d : StochRSI %D — media suavizada de StochRSI %K
        - momentum_tsi : True Strength Index — doble suavizado de cambios de precio
        - momentum_uo : Ultimate Oscillator — impulso en tres marcos temporales
        - momentum_wr : Williams %R — cierre relativo al máximo del período

        Tendencia (32):
        - trend_adx : Average Directional Index — fuerza de la tendencia (sin dirección)
        - trend_adx_pos : ADX +DI — fuerza del movimiento alcista
        - trend_adx_neg : ADX -DI — fuerza del movimiento bajista
        - trend_aroon_up : Aroon Up — períodos desde el máximo más alto
        - trend_aroon_down : Aroon Down — períodos desde el mínimo más bajo
        - trend_aroon_ind : Aroon Indicator — diferencia Aroon Up - Down
        - trend_cci : Commodity Channel Index — desviación del precio típico de su media
        - trend_dpo : Detrended Price Oscillator — precio sin tendencia para ver ciclos
        - trend_ema_fast : EMA rápida (12 períodos)
        - trend_ema_slow : EMA lenta (26 períodos)
        - trend_sma_fast : SMA rápida (12 períodos)
        - trend_sma_slow : SMA lenta (26 períodos)
        - trend_macd : MACD Line — diferencia entre EMA rápida y lenta
        - trend_macd_signal : MACD Signal — EMA de 9 períodos de la línea MACD
        - trend_macd_diff : MACD Histogram — diferencia MACD - Signal
        - trend_ichimoku_conv : Ichimoku Tenkan-sen — punto medio del rango de 9 períodos
        - trend_ichimoku_base : Ichimoku Kijun-sen — punto medio del rango de 26 períodos
        - trend_ichimoku_a : Ichimoku Senkou Span A — borde superior de la nube
        - trend_ichimoku_b : Ichimoku Senkou Span B — borde inferior de la nube
        - trend_visual_ichimoku_a : Senkou Span A desplazada 26 períodos (visual)
        - trend_visual_ichimoku_b : Senkou Span B desplazada 26 períodos (visual)
        - trend_kst : Know Sure Thing — oscilador de 4 tasas de cambio suavizadas
        - trend_kst_sig : KST Signal — media móvil de KST
        - trend_kst_diff : KST Diff — diferencia KST - Signal
        - trend_mass_index : Mass Index — detecta reversiones por expansión de rango
        - trend_psar_up : Parabolic SAR Up — valor SAR en tendencia alcista (NaN si bajista)
        - trend_psar_down : Parabolic SAR Down — valor SAR en tendencia bajista (NaN si alcista)
        - trend_psar_up_indicator : PSAR Up Indicator — 1 cuando cambia a tendencia alcista
        - trend_psar_down_indicator : PSAR Down Indicator — 1 cuando cambia a tendencia bajista
        - trend_stc : Schaff Trend Cycle — MACD + estocástico para ciclos de tendencia
        - trend_trix : TRIX — tasa de cambio % de la triple EMA
        - trend_vortex_ind_pos : Vortex +VI — movimiento de tendencia positivo
        - trend_vortex_ind_neg : Vortex -VI — movimiento de tendencia negativo
        - trend_vortex_ind_diff : Vortex Diff — diferencia +VI - (-VI)

        Volatilidad (21):
        - volatility_atr : Average True Range — grado de volatilidad del precio
        - volatility_bbh : Bollinger Band High — banda superior (media + K×std)
        - volatility_bbl : Bollinger Band Low — banda inferior (media - K×std)
        - volatility_bbm : Bollinger Band Mid — media móvil central (20 períodos)
        - volatility_bbw : Bollinger Band Width — anchura relativa de las bandas
        - volatility_bbp : Bollinger Band %B — posición del precio dentro de las bandas (0-1)
        - volatility_bbhi : BB High Indicator — 1 si cierre > banda alta
        - volatility_bbli : BB Low Indicator — 1 si cierre < banda baja
        - volatility_kch : Keltner Channel High — banda superior del canal
        - volatility_kcl : Keltner Channel Low — banda inferior del canal
        - volatility_kcc : Keltner Channel Mid — línea central del canal
        - volatility_kcw : Keltner Channel Width — anchura relativa del canal
        - volatility_kcp : Keltner Channel %B — posición del precio en el canal (0-1)
        - volatility_kchi : KC High Indicator — 1 si cierre > banda alta
        - volatility_kcli : KC Low Indicator — 1 si cierre < banda baja
        - volatility_dch : Donchian Channel High — máximo más alto de n períodos
        - volatility_dcl : Donchian Channel Low — mínimo más bajo de n períodos
        - volatility_dcm : Donchian Channel Mid — punto medio del canal
        - volatility_dcw : Donchian Channel Width — anchura relativa del canal
        - volatility_dcp : Donchian Channel %B — posición del precio en el canal (0-1)
        - volatility_ui : Ulcer Index — profundidad y duración de caídas desde máximos

        Volumen (10):
        - volume_adi : Accumulation/Distribution — flujo de dinero basado en rango y volumen
        - volume_cmf : Chaikin Money Flow — flujo de dinero ponderado por volumen (n períodos)
        - volume_em : Ease of Movement — relación entre cambio de precio y volumen
        - volume_sma_em : SMA Ease of Movement — media móvil suavizada del EoM
        - volume_fi : Force Index — presión compradora/vendedora (precio × volumen)
        - volume_mfi : Money Flow Index — RSI ponderado por volumen (0-100)
        - volume_nvi : Negative Volume Index — cambios en días de volumen decreciente
        - volume_obv : On-Balance Volume — volumen acumulado según dirección del precio
        - volume_vpt : Volume Price Trend — volumen acumulado × cambio % del precio
        - volume_vwap : VWAP — precio promedio ponderado por volumen

        Otros (3):
        - others_cr : Cumulative Return — retorno total acumulado desde el inicio
        - others_dlr : Daily Log Return — retorno logarítmico diario
        - others_dr : Daily Return — retorno porcentual diario

        Manuales (2):
        - returns : Retorno porcentual del precio de cierre
        - volatility_20 : Volatilidad rolling 20 períodos (std de returns)

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado fetch_data() previamente.
        ValueError
            Si mode no es "core" o "full".
        """
        self._require_data()

        df = self.data.copy()
        original_rows = len(df)

        if mode == "core":
            # Medias móviles
            df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=SMA_FAST_WINDOW)
            df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=SMA_SLOW_WINDOW)

            # RSI
            df["RSI_14"] = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)

            # MACD
            df["MACD"] = ta.trend.macd(df["Close"])
            df["MACD_signal"] = ta.trend.macd_signal(df["Close"])

            # Bollinger Bands
            df["BB_upper"] = ta.volatility.bollinger_hband(df["Close"])
            df["BB_lower"] = ta.volatility.bollinger_lband(df["Close"])

            # ATR
            df["ATR_14"] = ta.volatility.average_true_range(
                df["High"], df["Low"], df["Close"], window=ATR_PERIOD
            )

            # Indicadores manuales
            df["volume_change"] = df["Volume"].pct_change()
            df["returns"] = df["Close"].pct_change()
            df["volatility_20"] = df["returns"].rolling(window=VOLATILITY_WINDOW).std()

            # Incluir FGI si está disponible
            if getattr(self, "fear_greed_enabled", False) and "fgi_value" in self.data.columns:
                df["fgi_value"] = self.data["fgi_value"]

            df.dropna(inplace=True)
            n_features = len(df.columns) - 5  # descontar OHLCV originales

        elif mode == "full":
            df = ta.add_all_ta_features(
                df, open="Open", high="High", low="Low", close="Close", volume="Volume"
            )
            # add_all_ta_features no genera estos
            df["returns"] = df["Close"].pct_change()
            df["volatility_20"] = df["returns"].rolling(window=VOLATILITY_WINDOW).std()

            # ffill para indicadores con NaN por diseño (e.g. PSAR up/down)
            df.ffill(inplace=True)

            # Incluir FGI si está disponible
            if getattr(self, "fear_greed_enabled", False) and "fgi_value" in self.data.columns:
                df["fgi_value"] = self.data["fgi_value"]

            df.dropna(inplace=True)
            n_features = len(df.columns) - 5  # descontar OHLCV originales
        
        elif mode == "momentum_select":
        # Solo momentum seleccionados (verdes)
        
            df["momentum_kama"] = ta.momentum.kama(df["Close"])
            df["momentum_ppo"] = ta.momentum.ppo(df["Close"])
            df["momentum_ppo_signal"] = ta.momentum.ppo_signal(df["Close"])
            
            df["momentum_pvo"] = ta.momentum.pvo(df["Volume"])
            df["momentum_pvo_signal"] = ta.momentum.pvo_signal(df["Volume"])
            
            df["momentum_roc"] = ta.momentum.roc(df["Close"])
            df["momentum_rsi"] = ta.momentum.rsi(df["Close"], window=RSI_PERIOD)

            
            df["returns"] = df["Close"].pct_change()
            df["volatility_20"] = df["returns"].rolling(window=VOLATILITY_WINDOW).std()

            # Incluir FGI si aplica
            if getattr(self, "fear_greed_enabled", False) and "fgi_value" in self.data.columns:
                df["fgi_value"] = self.data["fgi_value"]

            df.dropna(inplace=True)
            n_features = len(df.columns) - 5

        else:
            raise ValueError(f"mode debe ser 'core', 'full' o 'momentum_select' , recibido: '{mode}'")
           

        self.features = df

        print(f"🔧 Features creados ({mode}): {n_features} indicadores")
        print(f"   Registros: {len(df)} (de {original_rows} originales)")

        return self
