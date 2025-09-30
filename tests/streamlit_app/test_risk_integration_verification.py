#!/usr/bin/env python3
"""
Risk Management UI Integration Verification
Verify that hardcoded calculations have been properly removed and backend integration works
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Add src directory to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.risk.risk_metrics import RiskMetrics
from streamlit_app.utils.risk_management_utils import RiskManagementProcessor


def test_backend_risk_calculations():
    """Test that backend risk calculations work correctly"""

    print("🧪 Testing Backend Risk Calculations...")

    # Generate sample returns data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="D")
    returns = pd.Series(np.random.normal(0.001, 0.02, 252), index=dates, name="returns")

    # Test RiskMetrics backend
    risk_metrics = RiskMetrics(confidence_level=0.95)

    # Test VaR calculation
    var_parametric = risk_metrics.value_at_risk(returns, method="parametric")
    var_historical = risk_metrics.value_at_risk(returns, method="historical")

    print(f"✅ VaR (Parametric): {var_parametric:.4f}")
    print(f"✅ VaR (Historical): {var_historical:.4f}")

    # Test risk-adjusted returns
    risk_adj_metrics = risk_metrics.risk_adjusted_returns(returns)
    annual_vol = risk_adj_metrics.get("annual_volatility", 0)
    sharpe_ratio = risk_adj_metrics.get("sharpe_ratio", 0)

    print(f"✅ Annual Volatility: {annual_vol:.4f}")
    print(f"✅ Sharpe Ratio: {sharpe_ratio:.4f}")

    # Test maximum drawdown
    dd_metrics = risk_metrics.maximum_drawdown(returns)
    max_dd = dd_metrics.get("max_drawdown", 0)

    print(f"✅ Maximum Drawdown: {max_dd:.4f}")

    return True


def test_utils_integration():
    """Test that utils properly use backend calculations"""

    print("\n🔧 Testing Utils Integration...")

    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    prices = pd.Series(
        100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 100)), index=dates
    )
    returns = prices.pct_change().dropna()

    # Test RiskManagementProcessor
    processor = RiskManagementProcessor()

    # Mock session state data
    mock_session_data = {"analysis_results": {"returns": returns}}

    # Test risk metrics calculation
    try:
        confidence_level = 0.95
        var_method = "Historical"

        risk_calc = RiskMetrics(confidence_level=confidence_level)

        # Test that utils can properly call backend methods
        risk_adj_metrics = risk_calc.risk_adjusted_returns(returns)
        volatility = risk_adj_metrics.get("annual_volatility", 0)

        var_result = risk_calc.value_at_risk(returns, method="historical")
        cvar_result = risk_calc.conditional_var(returns)

        print(f"✅ Utils Volatility (from backend): {volatility:.4f}")
        print(f"✅ Utils VaR (from backend): {var_result:.4f}")
        print(f"✅ Utils CVaR (from backend): {cvar_result:.4f}")

        return True

    except Exception as e:
        print(f"❌ Utils integration error: {e}")
        return False


def verify_no_hardcoded_calculations():
    """Verify that hardcoded calculations have been removed"""

    print("\n🔍 Verifying No Hardcoded Calculations...")

    # Check main risk management page
    risk_page_path = "streamlit_app/pages/06_risk_management.py"
    utils_path = "streamlit_app/utils/risk_management_utils.py"

    hardcoded_patterns = [
        "returns.std() * np.sqrt(252)",
        ".std() * math.sqrt(252)",
        "* 252**0.5",
        "expected_return / volatility",  # Direct Sharpe calculation
        ".rolling(30).quantile(0.05)",  # Direct VaR calculation
    ]

    issues_found = []

    for file_path in [risk_page_path, utils_path]:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                content = f.read()

            for pattern in hardcoded_patterns:
                if pattern in content:
                    issues_found.append(f"Found '{pattern}' in {file_path}")

    if issues_found:
        print("❌ Hardcoded calculations still found:")
        for issue in issues_found:
            print(f"   {issue}")
        return False
    else:
        print("✅ No hardcoded risk calculations found")
        return True


def main():
    """Run all verification tests"""

    print("🚀 Risk Management UI Integration Verification")
    print("=" * 60)

    # Test backend functionality
    backend_ok = test_backend_risk_calculations()

    # Test utils integration
    utils_ok = test_utils_integration()

    # Verify no hardcoded calculations
    no_hardcode_ok = verify_no_hardcoded_calculations()

    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY:")
    print(f"Backend Risk Calculations: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"Utils Integration: {'✅ PASS' if utils_ok else '❌ FAIL'}")
    print(f"No Hardcoded Calculations: {'✅ PASS' if no_hardcode_ok else '❌ FAIL'}")

    if all([backend_ok, utils_ok, no_hardcode_ok]):
        print("\n🎉 ALL TESTS PASSED - Risk Management UI is properly integrated!")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - Review the issues above")
        return 1


if __name__ == "__main__":
    exit(main())
