"""
Additional Strategy Classes for Backtesting Framework

This module extends the base strategy framework with model-based and
multi-asset strategies that were incorrectly implemented in frontend.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from .strategies import BaseStrategy, Signal, SignalType
from .engine import OrderSide, OrderType

logger = logging.getLogger(__name__)


class ModelBasedStrategy(BaseStrategy):
    """
    Model-based trading strategy using machine learning predictions.

    This strategy uses trained ML models to generate trading signals
    based on feature extraction from market data.
    """

    def __init__(
        self,
        model: Any,
        symbols: List[str],
        confidence_threshold: float = 0.7,
        position_sizing: str = "Fixed",
        **kwargs,
    ):
        super().__init__("ModelBased", symbols, **kwargs)
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.position_sizing = position_sizing
        self.last_signals = {}

    def on_start(self):
        """Initialize strategy"""
        self.is_initialized = True
        logger.info(f"Starting {self.name} strategy with {len(self.symbols)} symbols")

    def on_data(self, current_time: datetime) -> List[Signal]:
        """Generate signals based on model predictions"""
        signals = []

        for symbol in self.symbols:
            try:
                # Get current price
                current_price = self.get_current_price(symbol)
                if current_price is None:
                    continue

                # Extract features for prediction
                features = self._extract_features(symbol, current_time)
                if features is None:
                    continue

                # Make prediction
                prediction, confidence = self._make_prediction(features)

                # Generate signal if confidence is high enough
                if confidence >= self.confidence_threshold:
                    position_size = self._calculate_position_size(symbol, confidence)

                    if prediction > 0.5:  # Buy signal
                        signal = self.generate_signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=confidence,
                            confidence=confidence,
                            metadata={
                                "model_prediction": prediction,
                                "features": features.tolist(),
                            },
                        )
                        signals.append(signal)
                        self.execute_signal(signal)

                    elif prediction < 0.5:  # Sell signal
                        current_position = self.get_position(symbol)
                        if current_position and current_position.quantity > 0:
                            signal = self.generate_signal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                strength=confidence,
                                confidence=confidence,
                                metadata={
                                    "model_prediction": prediction,
                                    "features": features.tolist(),
                                },
                            )
                            signals.append(signal)
                            self.execute_signal(signal)

            except Exception as e:
                logger.warning(f"Error processing {symbol}: {e}")
                continue

        return signals

    def _extract_features(
        self, symbol: str, current_time: datetime
    ) -> Optional[np.ndarray]:
        """Extract features for model prediction"""
        try:
            # Get historical data for feature calculation
            data = self.get_market_data(symbol, periods=50)
            if data is None or len(data) < 10:
                return None

            # Calculate technical features
            close_prices = data["Close"]

            # Moving averages
            sma_5 = close_prices.rolling(5).mean().iloc[-1]
            sma_20 = close_prices.rolling(20).mean().iloc[-1]

            # Price ratios
            price_to_sma5 = close_prices.iloc[-1] / sma_5 if sma_5 > 0 else 1
            price_to_sma20 = close_prices.iloc[-1] / sma_20 if sma_20 > 0 else 1

            # Volatility
            returns = close_prices.pct_change().dropna()
            volatility = returns.rolling(10).std().iloc[-1] if len(returns) > 10 else 0

            # RSI-like momentum
            gains = returns[returns > 0].rolling(14).mean().iloc[-1] or 0
            losses = abs(returns[returns < 0]).rolling(14).mean().iloc[-1] or 0.01
            rsi_like = gains / (gains + losses)

            features = np.array([price_to_sma5, price_to_sma20, volatility, rsi_like])

            # Ensure no NaN values
            features = np.nan_to_num(features, 0)

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed for {symbol}: {e}")
            return None

    def _make_prediction(self, features: np.ndarray) -> tuple[float, float]:
        """Make model prediction and extract confidence"""
        try:
            if hasattr(self.model, "predict_proba"):
                # Classification with probability
                probabilities = self.model.predict_proba([features])[0]
                if len(probabilities) > 1:
                    confidence = max(probabilities)
                    prediction = np.argmax(probabilities)
                else:
                    confidence = probabilities[0]
                    prediction = 1 if confidence > 0.5 else 0
            else:
                # Direct prediction
                prediction = self.model.predict([features])[0]
                confidence = (
                    abs(prediction) if isinstance(prediction, (int, float)) else 0.7
                )

            return float(prediction), float(confidence)

        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            return 0.5, 0.0

    def _calculate_position_size(self, symbol: str, confidence: float) -> int:
        """Calculate position size based on portfolio and confidence"""
        try:
            portfolio_value = self.get_portfolio_value()
            current_price = self.get_current_price(symbol)

            if self.position_sizing == "Fixed":
                position_value = portfolio_value * 0.1  # 10% of portfolio
            elif self.position_sizing == "Kelly":
                position_value = portfolio_value * confidence * 0.2
            else:  # Risk Parity
                position_value = portfolio_value * 0.05  # Conservative 5%

            position_size = (
                int(position_value / current_price) if current_price > 0 else 0
            )
            return max(1, position_size)

        except Exception:
            return 100  # Default position size

    def on_finish(self):
        """Clean up strategy"""
        logger.info(f"Finished {self.name} strategy")


class MultiAssetStrategy(BaseStrategy):
    """
    Multi-asset portfolio strategy with rebalancing.

    This strategy manages a portfolio of multiple assets with
    periodic rebalancing based on target weights.
    """

    def __init__(
        self,
        symbols: List[str],
        rebalance_frequency: str = "Monthly",
        risk_model: str = "Equal Weight",
        max_position: float = 0.2,
        **kwargs,
    ):
        super().__init__("MultiAsset", symbols, **kwargs)
        self.rebalance_frequency = rebalance_frequency
        self.risk_model = risk_model
        self.max_position = max_position
        self.last_rebalance = None
        self.target_weights = {}

    def on_start(self):
        """Initialize strategy"""
        self.is_initialized = True

        if self.risk_model == "Equal Weight":
            weight = min(1.0 / len(self.symbols), self.max_position)
            self.target_weights = {symbol: weight for symbol in self.symbols}

        logger.info(f"Starting {self.name} strategy with {len(self.symbols)} assets")

    def on_data(self, current_time: datetime) -> List[Signal]:
        """Rebalance portfolio based on frequency"""
        signals = []

        if self._should_rebalance(current_time):
            # Calculate current weights
            current_weights = self._get_current_weights()

            # Generate rebalancing signals
            for symbol in self.symbols:
                current_weight = current_weights.get(symbol, 0)
                target_weight = self.target_weights.get(symbol, 0)

                weight_diff = target_weight - current_weight

                if abs(weight_diff) > 0.01:  # Only rebalance if significant difference
                    portfolio_value = self.get_portfolio_value()
                    current_price = self.get_current_price(symbol)

                    if current_price and current_price > 0:
                        target_value = portfolio_value * target_weight
                        current_position = self.get_position(symbol)
                        current_value = (
                            (current_position.quantity * current_price)
                            if current_position
                            else 0
                        )

                        value_diff = target_value - current_value
                        quantity_diff = int(value_diff / current_price)

                        if quantity_diff > 0:
                            signal = self.generate_signal(
                                symbol=symbol,
                                signal_type=SignalType.BUY,
                                strength=min(1.0, abs(weight_diff) * 10),
                                confidence=0.8,
                                metadata={
                                    "rebalance": True,
                                    "target_weight": target_weight,
                                },
                            )
                            signals.append(signal)
                            self.execute_signal(signal)

                        elif quantity_diff < 0:
                            signal = self.generate_signal(
                                symbol=symbol,
                                signal_type=SignalType.SELL,
                                strength=min(1.0, abs(weight_diff) * 10),
                                confidence=0.8,
                                metadata={
                                    "rebalance": True,
                                    "target_weight": target_weight,
                                },
                            )
                            signals.append(signal)
                            self.execute_signal(signal)

            self.last_rebalance = current_time

        return signals

    def _should_rebalance(self, current_time: datetime) -> bool:
        """Check if it's time to rebalance"""
        if self.last_rebalance is None:
            return True

        if self.rebalance_frequency == "Daily":
            return True
        elif self.rebalance_frequency == "Weekly":
            return (current_time - self.last_rebalance).days >= 7
        elif self.rebalance_frequency == "Monthly":
            return (current_time - self.last_rebalance).days >= 30

        return False

    def _get_current_weights(self) -> Dict[str, float]:
        """Get current portfolio weights"""
        portfolio_value = self.get_portfolio_value()
        weights = {}

        for symbol in self.symbols:
            position = self.get_position(symbol)
            if position:
                current_price = self.get_current_price(symbol)
                if current_price:
                    position_value = position.quantity * current_price
                    weights[symbol] = (
                        position_value / portfolio_value if portfolio_value > 0 else 0
                    )
            else:
                weights[symbol] = 0

        return weights

    def on_finish(self):
        """Clean up strategy"""
        logger.info(f"Finished {self.name} strategy")
