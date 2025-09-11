"""
Data Acquisition Utilities

This module provides utility functions for data acquisition workflows,
separated from UI components for better testability and maintainability.

Design Principles:
- Separation of Concerns: UI logic vs Business logic
- Testability: Pure Python functions without Streamlit dependencies
- Reusability: Functions can be used across different pages
- Maintainability: Easy to modify business logic without touching UI
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import traceback
import time
from pathlib import Path
import sys

# Add src directory to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.data.collectors import YahooFinanceCollector, AlphaVantageCollector
    from src.data.storage import DataStorage
    from src.data.validation import DataQualityValidator
    from src.config import settings
except ImportError as e:
    # Handle import errors gracefully for testing
    print(f"Import warning in data_utils: {e}")


class DataAcquisitionManager:
    """Manager class for data acquisition operations"""

    def __init__(self):
        self.storage = DataStorage()
        self.validator = DataQualityValidator()

    def initialize_session_state(self, session_state: Dict) -> None:
        """Initialize session state for data acquisition"""

        if "data_cache" not in session_state:
            session_state["data_cache"] = {}

        if "collection_status" not in session_state:
            session_state["collection_status"] = {}

        if "validation_results" not in session_state:
            session_state["validation_results"] = {}

        if "batch_operations" not in session_state:
            session_state["batch_operations"] = {
                "collection_queue": [],
                "validation_queue": [],
                "save_queue": [],
            }

    def start_yahoo_finance_collection(
        self, config: Dict[str, Any], session_state: Dict
    ) -> Tuple[bool, str]:
        """
        Start Yahoo Finance data collection

        Args:
            config: Collection configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            # Validate configuration
            validation_result = self._validate_collection_config(config)
            if not validation_result[0]:
                return validation_result

            # Initialize collector
            collector = YahooFinanceCollector()

            # Extract parameters
            symbols = [s.strip().upper() for s in config["symbols"].split(",")]
            period = config["period"]
            interval = config["interval"]

            # Update status
            session_state["collection_status"] = {
                "active": True,
                "source": "Yahoo Finance",
                "symbols": symbols,
                "start_time": datetime.now(),
                "progress": 0.0,
                "current_symbol": "",
                "message": "Initializing collection...",
            }

            # Collect data for each symbol
            collected_data = {}
            total_symbols = len(symbols)

            for i, symbol in enumerate(symbols):
                try:
                    # Update progress
                    progress = i / total_symbols
                    session_state["collection_status"].update(
                        {
                            "progress": progress,
                            "current_symbol": symbol,
                            "message": f"Collecting {symbol}... ({i+1}/{total_symbols})",
                        }
                    )

                    # Collect data
                    data = collector.collect_data(
                        symbol=symbol, period=period, interval=interval
                    )

                    if data is not None and not data.empty:
                        # Store in cache
                        cache_key = f"{symbol}_{period}_{interval}"
                        session_state["data_cache"][cache_key] = {
                            "data": data,
                            "symbol": symbol,
                            "source": "Yahoo Finance",
                            "period": period,
                            "interval": interval,
                            "collected_at": datetime.now(),
                            "rows": len(data),
                            "columns": list(data.columns),
                            "date_range": {
                                "start": data.index.min().strftime("%Y-%m-%d"),
                                "end": data.index.max().strftime("%Y-%m-%d"),
                            },
                        }

                        collected_data[symbol] = data

                    # Small delay to prevent rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"Error collecting {symbol}: {str(e)}")
                    continue

            # Final update
            session_state["collection_status"].update(
                {
                    "active": False,
                    "progress": 1.0,
                    "message": f"Collection completed! Collected {len(collected_data)} symbols",
                    "completed_at": datetime.now(),
                }
            )

            if collected_data:
                return (
                    True,
                    f"Successfully collected data for {len(collected_data)} symbols",
                )
            else:
                return False, "No data was collected successfully"

        except Exception as e:
            session_state["collection_status"]["active"] = False
            return False, f"Collection failed: {str(e)}"

    def start_custom_upload_collection(
        self, uploaded_files: List, config: Dict[str, Any], session_state: Dict
    ) -> Tuple[bool, str]:
        """
        Process custom uploaded files

        Args:
            uploaded_files: List of uploaded files
            config: Upload configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            if not uploaded_files:
                return False, "No files uploaded"

            processed_files = 0
            total_files = len(uploaded_files)

            # Update status
            session_state["collection_status"] = {
                "active": True,
                "source": "Custom Upload",
                "files": [f.name for f in uploaded_files],
                "start_time": datetime.now(),
                "progress": 0.0,
                "current_file": "",
                "message": "Processing uploaded files...",
            }

            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = i / total_files
                    session_state["collection_status"].update(
                        {
                            "progress": progress,
                            "current_file": uploaded_file.name,
                            "message": f"Processing {uploaded_file.name}... ({i+1}/{total_files})",
                        }
                    )

                    # Read file based on extension
                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    if file_extension == "csv":
                        data = pd.read_csv(uploaded_file)
                    elif file_extension in ["xlsx", "xls"]:
                        data = pd.read_excel(uploaded_file)
                    else:
                        continue

                    # Process data
                    processed_data = self._process_uploaded_data(data, config)

                    if processed_data is not None:
                        # Store in cache
                        file_name = uploaded_file.name.split(".")[0]
                        cache_key = f"{file_name}_upload"

                        session_state["data_cache"][cache_key] = {
                            "data": processed_data,
                            "symbol": file_name,
                            "source": "Custom Upload",
                            "filename": uploaded_file.name,
                            "collected_at": datetime.now(),
                            "rows": len(processed_data),
                            "columns": list(processed_data.columns),
                            "date_range": {
                                "start": (
                                    processed_data.index.min().strftime("%Y-%m-%d")
                                    if hasattr(processed_data.index, "min")
                                    else "N/A"
                                ),
                                "end": (
                                    processed_data.index.max().strftime("%Y-%m-%d")
                                    if hasattr(processed_data.index, "max")
                                    else "N/A"
                                ),
                            },
                        }

                        processed_files += 1

                except Exception as e:
                    print(f"Error processing {uploaded_file.name}: {str(e)}")
                    continue

            # Final update
            session_state["collection_status"].update(
                {
                    "active": False,
                    "progress": 1.0,
                    "message": f"Upload completed! Processed {processed_files} files",
                    "completed_at": datetime.now(),
                }
            )

            if processed_files > 0:
                return True, f"Successfully processed {processed_files} files"
            else:
                return False, "No files were processed successfully"

        except Exception as e:
            session_state["collection_status"]["active"] = False
            return False, f"Upload processing failed: {str(e)}"

    def validate_selected_data(
        self, symbols: List[str], level: str, report: bool, session_state: Dict
    ) -> Tuple[bool, str]:
        """
        Validate selected data entries

        Args:
            symbols: List of symbols to validate
            level: Validation level
            report: Whether to generate detailed report
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            if not symbols:
                return False, "No symbols selected for validation"

            validation_results = {}

            for symbol in symbols:
                if symbol in session_state["data_cache"]:
                    data = session_state["data_cache"][symbol]["data"]

                    # Run validation
                    result = self.validator.validate_data(
                        data=data, level=level, generate_report=report
                    )

                    validation_results[symbol] = result

            # Store results
            session_state["validation_results"].update(validation_results)

            # Calculate summary
            total_validated = len(validation_results)
            passed_validation = sum(
                1 for r in validation_results.values() if r.overall_quality == "PASS"
            )

            return (
                True,
                f"Validated {total_validated} symbols. {passed_validation} passed validation.",
            )

        except Exception as e:
            return False, f"Validation failed: {str(e)}"

    def validate_all_data(
        self, level: str, report: bool, session_state: Dict
    ) -> Tuple[bool, str]:
        """
        Validate all cached data

        Args:
            level: Validation level
            report: Whether to generate detailed report
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        all_symbols = list(session_state["data_cache"].keys())
        return self.validate_selected_data(all_symbols, level, report, session_state)

    def execute_batch_collection(
        self, config: Dict[str, Any], session_state: Dict
    ) -> Tuple[bool, str]:
        """
        Execute batch collection operation

        Args:
            config: Batch configuration
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            queue = session_state["batch_operations"]["collection_queue"]

            if not queue:
                return False, "No items in collection queue"

            results = []
            for item in queue:
                if item["source"] == "Yahoo Finance":
                    success, message = self.start_yahoo_finance_collection(
                        item["config"], session_state
                    )
                    results.append((success, message))

            # Clear queue
            session_state["batch_operations"]["collection_queue"] = []

            successful = sum(1 for r in results if r[0])
            total = len(results)

            return True, f"Batch collection completed: {successful}/{total} successful"

        except Exception as e:
            return False, f"Batch collection failed: {str(e)}"

    def execute_batch_validation(self, session_state: Dict) -> Tuple[bool, str]:
        """
        Execute batch validation operation

        Args:
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            queue = session_state["batch_operations"]["validation_queue"]

            if not queue:
                return False, "No items in validation queue"

            all_symbols = []
            for item in queue:
                all_symbols.extend(item["symbols"])

            # Remove duplicates
            unique_symbols = list(set(all_symbols))

            # Validate all
            success, message = self.validate_selected_data(
                unique_symbols, "standard", True, session_state
            )

            # Clear queue
            session_state["batch_operations"]["validation_queue"] = []

            return success, f"Batch validation: {message}"

        except Exception as e:
            return False, f"Batch validation failed: {str(e)}"

    def execute_batch_save(self, session_state: Dict) -> Tuple[bool, str]:
        """
        Execute batch save operation

        Args:
            session_state: Streamlit session state

        Returns:
            Tuple of (success, message)
        """

        try:
            queue = session_state["batch_operations"]["save_queue"]

            if not queue:
                return False, "No items in save queue"

            saved_count = 0

            for item in queue:
                symbol = item["symbol"]
                if symbol in session_state["data_cache"]:
                    data = session_state["data_cache"][symbol]["data"]

                    # Save data
                    success = self.storage.save_data(
                        data=data,
                        symbol=symbol,
                        metadata=session_state["data_cache"][symbol],
                    )

                    if success:
                        saved_count += 1

            # Clear queue
            session_state["batch_operations"]["save_queue"] = []

            return True, f"Batch save completed: {saved_count} items saved"

        except Exception as e:
            return False, f"Batch save failed: {str(e)}"

    def _validate_collection_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate collection configuration"""

        required_fields = ["symbols", "period", "interval"]

        for field in required_fields:
            if field not in config or not config[field]:
                return False, f"Missing required field: {field}"

        # Validate symbols
        symbols = [s.strip() for s in config["symbols"].split(",")]
        if not symbols or any(len(s) == 0 for s in symbols):
            return False, "Invalid symbols provided"

        return True, "Configuration valid"

    def _process_uploaded_data(
        self, data: pd.DataFrame, config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """Process uploaded data according to configuration"""

        try:
            # Basic processing
            processed_data = data.copy()

            # Try to set date index
            if config.get("date_column"):
                date_col = config["date_column"]
                if date_col in processed_data.columns:
                    processed_data[date_col] = pd.to_datetime(processed_data[date_col])
                    processed_data.set_index(date_col, inplace=True)

            # Clean data
            processed_data = processed_data.dropna()

            return processed_data

        except Exception as e:
            print(f"Error processing uploaded data: {str(e)}")
            return None

    def get_collection_progress(self, session_state: Dict) -> Dict[str, Any]:
        """Get current collection progress"""

        return session_state.get(
            "collection_status",
            {"active": False, "progress": 0.0, "message": "No active collection"},
        )

    def get_available_data_summary(self, session_state: Dict) -> Dict[str, Any]:
        """Get summary of available data"""

        data_cache = session_state.get("data_cache", {})

        if not data_cache:
            return {"total_symbols": 0, "total_rows": 0, "sources": []}

        total_symbols = len(data_cache)
        total_rows = sum(item["rows"] for item in data_cache.values())
        sources = list(set(item["source"] for item in data_cache.values()))

        return {
            "total_symbols": total_symbols,
            "total_rows": total_rows,
            "sources": sources,
            "symbols": list(data_cache.keys()),
        }
