#modifique para automatizar la estrategia el 20/3/26
"""Mixin para market intelligence: régimen de mercado y estrategias."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ta as ta_lib
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    CHART_HEIGHT_SECONDARY,
    REGIME_CORRELATION_THRESHOLD,
    REGIME_GMM_N_INIT,
    REGIME_LONG_WINDOW,
    REGIME_MEDIUM_WINDOW,
    REGIME_NAN_THRESHOLD,
    REGIME_PCA_VARIANCE,
    REGIME_SHORT_WINDOW,
    REGIME_STRUCTURAL_SMA,
    REGIME_VARIANCE_THRESHOLD,
)
from .constants import COLOR_PALETTE, REGIME_LABELS, STRATEGY_REGISTRY


class RegimeMixin:
    """Métodos de market intelligence: detect_regime(), regime_report(), recommend_strategies(), select_strategy()."""

    def detect_regime(self, n_regimes: int = 3) -> "RegimeMixin":
        """
        Detecta el régimen de mercado actual usando pipeline PCA + GMM.

        Pipeline hedge-fund quality:
        1. 86+ features via ta.add_all_ta_features() + custom features
        2. Exclusión de features inútiles (binarias, precio absoluto, acumulativos)
        3. Limpieza (ffill, inf→NaN, drop columnas >50% NaN, dropna filas)
        4. VarianceThreshold para eliminar features near-constant
        5. Filtro de correlación (|r| > 0.95)
        6. PCA retiene 95% de varianza
        7. GMM robusto (n_init=10, full covariance)

        Parameters
        ----------
        n_regimes : int, default 3
            Número de regímenes a detectar.
            Default 3: Bull 🟢, Bear 🔴, Sideways 🟡.

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        RuntimeError
            Si no se ha ejecutado create_features() previamente.
        """
        self._require_features()

        df = self.data.copy()
        n = len(df)

        SHORT, MEDIUM, LONG = (
            REGIME_SHORT_WINDOW,
            REGIME_MEDIUM_WINDOW,
            REGIME_LONG_WINDOW,
        )
        STRUCTURAL_DESIRED = REGIME_STRUCTURAL_SMA
        long_structural = max(min(STRUCTURAL_DESIRED, n - LONG), LONG)

        EXCLUDE_COLS = {
            "Open", "High", "Low", "Close", "Volume",
            "volatility_bbhi", "volatility_bbli",
            "volatility_kchi", "volatility_kcli",
            "trend_psar_up_indicator", "trend_psar_down_indicator",
            "trend_ema_fast", "trend_ema_slow",
            "trend_sma_fast", "trend_sma_slow",
            "volatility_bbh", "volatility_bbl", "volatility_bbm",
            "volatility_kch", "volatility_kcl", "volatility_kcc",
            "volatility_dch", "volatility_dcl", "volatility_dcm",
            "trend_ichimoku_conv", "trend_ichimoku_base",
            "trend_ichimoku_a", "trend_ichimoku_b",
            "trend_visual_ichimoku_a", "trend_visual_ichimoku_b",
            "trend_psar_up", "trend_psar_down",
            "trend_dpo", "momentum_kama",
            "volume_adi", "volume_obv", "volume_vpt",
            "volume_nvi", "others_cr", "volume_vwap",
            "fgi_value",
        }

        df_ta = ta_lib.add_all_ta_features(
            df, open="Open", high="High", low="Low", close="Close", volume="Volume"
        )

        returns = df_ta["Close"].pct_change()
        df_ta["trend_7"] = returns.rolling(SHORT).mean()
        df_ta["trend_21"] = returns.rolling(MEDIUM).mean()
        df_ta["trend_50"] = returns.rolling(LONG).mean()
        df_ta["vol_7"] = returns.rolling(SHORT).std()
        df_ta["vol_21"] = returns.rolling(MEDIUM).std()
        df_ta["vol_50"] = returns.rolling(LONG).std()

        log_hl = np.log(df_ta["High"] / df_ta["Low"]) ** 2
        log_co = np.log(df_ta["Close"] / df_ta["Open"]) ** 2
        df_ta["gk_volatility"] = np.sqrt(
            (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(MEDIUM).mean()
        )

        df_ta["vol_ratio"] = df_ta["vol_7"] / df_ta["vol_50"]
        df_ta["volume_ratio"] = (
            df_ta["Volume"] / df_ta["Volume"].rolling(MEDIUM).mean()
        )

        sma_structural = df_ta["Close"].rolling(long_structural).mean()
        df_ta["dist_sma_structural"] = (
            (df_ta["Close"] - sma_structural) / sma_structural
        )

        rolling_high = df_ta["Close"].rolling(long_structural).max()
        df_ta["drawdown"] = (df_ta["Close"] - rolling_high) / rolling_high

        df_ta["_returns"] = returns

        exclude = EXCLUDE_COLS | {"_returns"}
        feature_cols = [c for c in df_ta.columns if c not in exclude]
        regime_features = df_ta[feature_cols].copy()

        regime_features.ffill(inplace=True)
        regime_features.replace([np.inf, -np.inf], np.nan, inplace=True)

        thresh = len(regime_features) * REGIME_NAN_THRESHOLD
        regime_features.dropna(axis=1, thresh=int(thresh), inplace=True)
        regime_features.dropna(inplace=True)

        n_raw = regime_features.shape[1]

        if len(regime_features) < n_regimes * 5:
            import warnings
            warnings.warn(
                f"Solo {len(regime_features)} filas válidas para {n_regimes} regímenes. "
                f"Considere usar más datos (last_n más alto)."
            )

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(regime_features)

        var_selector = VarianceThreshold(threshold=REGIME_VARIANCE_THRESHOLD)
        X_selected = var_selector.fit_transform(X_scaled)

        corr = pd.DataFrame(X_selected).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [
            col for col in upper.columns
            if any(upper[col] > REGIME_CORRELATION_THRESHOLD)
        ]
        X_curated = np.delete(X_selected, to_drop, axis=1)

        n_after_curation = X_curated.shape[1]

        pca = PCA(n_components=REGIME_PCA_VARIANCE, random_state=42)
        X_pca = pca.fit_transform(X_curated)

        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type="full",
            n_init=REGIME_GMM_N_INIT,
            random_state=42,
        )
        gmm.fit(X_pca)

        labels = gmm.predict(X_pca)

        aligned_returns = df_ta.loc[regime_features.index, "_returns"]
        cluster_returns = pd.DataFrame(
            {"cluster": labels, "returns": aligned_returns.values}
        )
        mean_returns = cluster_returns.groupby("cluster")["returns"].mean()

        sorted_clusters = mean_returns.sort_values().index.tolist()
        cluster_to_regime = {
            sorted_clusters[0]: 0,
            sorted_clusters[1]: 1,
            sorted_clusters[2]: 2,
        }

        mapped_labels = pd.Series(
            [cluster_to_regime[c] for c in labels],
            index=regime_features.index,
        )
        self.features["regime"] = np.nan
        common_idx = self.features.index.intersection(mapped_labels.index)
        self.features.loc[common_idx, "regime"] = mapped_labels.loc[common_idx].values

        last_point = X_pca[-1].reshape(1, -1)
        proba = gmm.predict_proba(last_point)[0]

        regime_probs = {}
        for cluster_id, regime_id in cluster_to_regime.items():
            label = REGIME_LABELS[regime_id]
            regime_probs[label] = proba[cluster_id]

        last_regime_id = cluster_to_regime[labels[-1]]
        regime_label = REGIME_LABELS[last_regime_id]
        self.regime = regime_label.split()[0]
        self.regime_probabilities = regime_probs
        self.regime_model = gmm

        confidence = proba[labels[-1]]
        print(f"📊 Régimen detectado: {regime_label} (confianza: {confidence:.1%})")
        print(
            f"   Pipeline: {n_raw} features → {n_after_curation} curados → {X_pca.shape[1]} PCA"
        )
        print(f"   Períodos analizados: {len(regime_features)} de {n} disponibles")
        print(f"   BIC: {gmm.bic(X_pca):.0f}")

        return self

    def regime_report(self) -> None:
        """
        Visualización detallada del régimen actual.
        """
        self._require_regime()

        print("=" * 50)
        print("📊 REPORTE DE RÉGIMEN DE MERCADO")
        print("=" * 50)
        emoji = {"Bull": "🟢", "Bear": "🔴", "Sideways": "🟡"}.get(self.regime, "")
        print(f"\nRégimen actual: {self.regime} {emoji}")
        print(f"\n{'Régimen':<20} {'Probabilidad':>12}")
        print("-" * 34)
        for label, prob in self.regime_probabilities.items():
            bar = "█" * int(prob * 20)
            print(f"{label:<20} {prob:>10.1%}  {bar}")

        df = self.features.dropna(subset=["regime"]).copy()
        print(
            f"\n{'Régimen':<20} {'Retorno Prom':>14} {'Volatilidad':>14} {'Períodos':>10} {'Duración Prom':>15}"
        )
        print("-" * 75)

        for regime_id, label in REGIME_LABELS.items():
            mask = df["regime"] == regime_id
            subset = df[mask]
            if len(subset) == 0:
                continue

            avg_return = subset["returns"].mean()
            avg_vol = subset["volatility_20"].mean()
            n_periods = len(subset)

            regime_series = mask.astype(int)
            changes = regime_series.diff().fillna(0) != 0
            groups = changes.cumsum()
            streaks = regime_series.groupby(groups).sum()
            streaks = streaks[streaks > 0]
            avg_duration = streaks.mean() if len(streaks) > 0 else 0

            print(
                f"{label:<20} {avg_return:>13.4%} {avg_vol:>13.4f} {n_periods:>10} {avg_duration:>14.1f}"
            )

        regime_colors = {
            0: COLOR_PALETTE["red"],
            1: COLOR_PALETTE["yellow"],
            2: COLOR_PALETTE["green"],
        }

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                mode="lines",
                name="Precio",
                line=dict(color="white", width=1.5),
            )
        )

        regime_col = df["regime"].dropna()
        if len(regime_col) > 0:
            current_regime = regime_col.iloc[0]
            start_idx = regime_col.index[0]

            for i in range(1, len(regime_col)):
                if regime_col.iloc[i] != current_regime or i == len(regime_col) - 1:
                    end_idx = regime_col.index[i]
                    fig.add_vrect(
                        x0=start_idx,
                        x1=end_idx,
                        fillcolor=regime_colors.get(
                            int(current_regime), COLOR_PALETTE["gray"]
                        ),
                        opacity=0.15,
                        line_width=0,
                    )
                    current_regime = regime_col.iloc[i]
                    start_idx = regime_col.index[i]

        fig.update_layout(
            title=f"Precio con Regímenes de Mercado — {getattr(self, 'symbol', '')}",
            xaxis_title="Fecha",
            yaxis_title="Precio (USDT)",
            template="plotly_dark",
            plot_bgcolor=COLOR_PALETTE["dark"],
            paper_bgcolor=COLOR_PALETTE["dark"],
            height=CHART_HEIGHT_SECONDARY,
        )

        fig.show()

    def recommend_strategies(self, auto_select: bool = True) -> "RegimeMixin":
        """
        Recomienda estrategias de trading basadas en el régimen actual.

        Clasifica cada estrategia del registry según compatibilidad
        con el régimen detectado (best_regimes / worst_regimes) y
        muestra el rationale de cada una.

        Si auto_select=True, selecciona automáticamente la mejor estrategia.
        """
        self._require_regime()

        recommended = []
        neutral = []
        not_recommended = []

        for key, info in STRATEGY_REGISTRY.items():
            entry = {"key": key, **info}
            if self.regime in info["best_regimes"]:
                entry["signal"] = "🟢 Recomendada"
                recommended.append(entry)
            elif self.regime in info["worst_regimes"]:
                entry["signal"] = "🔴 No recomendada"
                not_recommended.append(entry)
            else:
                entry["signal"] = "🟡 Neutral"
                neutral.append(entry)

        ordered = recommended + neutral + not_recommended

        print("=" * 70)
        print(f"📈 ESTRATEGIAS RECOMENDADAS — Régimen: {self.regime}")
        print("=" * 70)

        for entry in ordered:
            print(f"\n  {entry['signal']}  {entry['name']}")
            print(f"  {entry['rationale']}")

        print("=" * 70)

        if auto_select:
            if recommended:
                best = recommended[0]
            elif neutral:
                best = neutral[0]
            else:
                best = not_recommended[0]

            print("\n🤖 Seleccionando automáticamente la mejor estrategia...")
            self.select_strategy(best["key"])

        return self

    def select_strategy(self, strategy: str) -> "RegimeMixin":
        """
        Selecciona una estrategia de trading.

        Parameters
        ----------
        strategy : str
            Nombre de la estrategia. Opciones:
            "trend_following", "mean_reversion", "momentum",
            "breakout", "stat_arb", "volatility".

        Returns
        -------
        CryptoBot
            Retorna self para permitir method chaining.

        Raises
        ------
        ValueError
            Si la estrategia no está en el registry.
        """
        if strategy not in STRATEGY_REGISTRY:
            available = list(STRATEGY_REGISTRY.keys())
            raise ValueError(
                f"Estrategia '{strategy}' no válida. Opciones: {available}"
            )

        self.selected_strategy = strategy
        info = STRATEGY_REGISTRY[strategy]
        print(f"✅ Estrategia seleccionada: {info['name']}")
        print(f"   {info['description']}")
        return self