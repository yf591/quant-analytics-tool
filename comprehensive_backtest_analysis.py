#!/usr/bin/env python3
"""
Comprehensive Backtest Architecture Violation Analysis
Find all business logic that should be in backend but is implemented in frontend
"""

import sys
import re
from pathlib import Path

project_root = Path(__file__).parent
sys.path.append(str(project_root))


def analyze_backend_modules():
    """Analyze what's implemented in backend"""

    backend_features = {
        "strategies": [],
        "engines": [],
        "calculators": [],
        "portfolios": [],
        "metrics": [],
        "execution": [],
    }

    # Analyze strategies
    strategies_file = project_root / "src/backtesting/strategies.py"
    if strategies_file.exists():
        with open(strategies_file, "r") as f:
            content = f.read()
            # Find strategy classes
            strategy_matches = re.findall(r"class\s+(\w+Strategy)\s*\(", content)
            backend_features["strategies"] = strategy_matches

    # Analyze engine
    engine_file = project_root / "src/backtesting/engine.py"
    if engine_file.exists():
        with open(engine_file, "r") as f:
            content = f.read()
            engine_matches = re.findall(
                r"class\s+(\w+Engine|\w+Order|\w+Position)\s*\(", content
            )
            backend_features["engines"] = engine_matches

    # Analyze metrics
    metrics_file = project_root / "src/backtesting/metrics.py"
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            content = f.read()
            metrics_matches = re.findall(
                r"class\s+(\w+Calculator|\w+Metrics)\s*\(", content
            )
            backend_features["calculators"] = metrics_matches

            # Find calculation methods
            method_matches = re.findall(
                r"def\s+(_?calculate_\w+|_?\w+_ratio|_?\w+_var)\s*\(", content
            )
            backend_features["metrics"] = method_matches[:10]  # First 10 methods

    return backend_features


def analyze_frontend_violations():
    """Analyze what business logic exists in frontend files"""

    violations = []

    frontend_files = [
        "streamlit_app/pages/05_backtesting.py",
        "streamlit_app/utils/backtest_utils.py",
    ]

    for file_path in frontend_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue

        violations.extend(analyze_file_violations(str(full_path), file_path))

    return violations


def analyze_file_violations(file_path: str, relative_path: str):
    """Analyze violations in a specific file"""

    violations = []

    # Patterns that indicate business logic (not UI)
    business_logic_patterns = {
        "strategy_classes": r"class\s+(\w*Strategy)\s*[:\(]",
        "calculation_methods": r"def\s+(calculate_\w+|_calculate_\w+)\s*\(",
        "financial_formulas": r"(sharpe_ratio|sortino_ratio|calmar_ratio|max_drawdown|volatility)\s*=",
        "signal_generation": r"(buy_signal|sell_signal|signal_type|generate.*signal)",
        "portfolio_logic": r"(portfolio_value|position_size|rebalance|weight.*portfolio)",
        "strategy_logic": r"(on_start|on_data|on_finish|moving_average|bollinger|momentum)",
        "engine_operations": r"(backtest.*engine|run_backtest|engine\.|add_data)",
        "complex_algorithms": r"(rsi_like|kelly|risk_parity|model_predict|extract_features)",
    }

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_num, line in enumerate(lines, 1):
            for violation_type, pattern in business_logic_patterns.items():
                matches = re.findall(pattern, line, re.IGNORECASE)
                if matches:
                    violations.append(
                        {
                            "file": relative_path,
                            "line": line_num,
                            "type": violation_type,
                            "content": line.strip(),
                            "matches": matches,
                            "severity": (
                                "HIGH"
                                if violation_type
                                in [
                                    "strategy_classes",
                                    "calculation_methods",
                                    "financial_formulas",
                                ]
                                else "MEDIUM"
                            ),
                        }
                    )

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

    return violations


def count_business_logic_lines():
    """Count how many lines are business logic vs UI logic"""

    frontend_files = [
        "streamlit_app/pages/05_backtesting.py",
        "streamlit_app/utils/backtest_utils.py",
    ]

    line_counts = {}

    for file_path in frontend_files:
        full_path = project_root / file_path
        if not full_path.exists():
            continue

        try:
            with open(full_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            ui_lines = 0
            business_lines = 0
            comment_lines = 0

            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    comment_lines += 1
                elif any(
                    ui_keyword in stripped.lower()
                    for ui_keyword in [
                        "st.",
                        "streamlit",
                        "display",
                        "render",
                        "widget",
                    ]
                ):
                    ui_lines += 1
                elif any(
                    biz_keyword in stripped.lower()
                    for biz_keyword in [
                        "calculate",
                        "strategy",
                        "signal",
                        "engine",
                        "backtest",
                        "portfolio",
                        "class ",
                    ]
                ):
                    business_lines += 1
                else:
                    # Neutral lines (imports, etc.)
                    pass

            line_counts[file_path] = {
                "total": len(lines),
                "ui_lines": ui_lines,
                "business_lines": business_lines,
                "comment_lines": comment_lines,
                "business_ratio": business_lines / len(lines) if len(lines) > 0 else 0,
            }

        except Exception as e:
            print(f"Error counting lines in {file_path}: {e}")

    return line_counts


def main():
    """Run comprehensive analysis"""

    print("🔍 COMPREHENSIVE BACKTEST ARCHITECTURE VIOLATION ANALYSIS")
    print("=" * 80)

    # Analyze backend features
    print("\n📁 BACKEND ANALYSIS:")
    print("-" * 40)
    backend_features = analyze_backend_modules()

    print(f"Backend Strategies: {backend_features['strategies']}")
    print(f"Backend Engines: {backend_features['engines']}")
    print(f"Backend Calculators: {backend_features['calculators']}")
    print(f"Backend Methods (sample): {backend_features['metrics'][:5]}...")

    # Count line ratios
    print("\n📊 FRONTEND LINE ANALYSIS:")
    print("-" * 40)
    line_counts = count_business_logic_lines()

    total_ui_lines = 0
    total_business_lines = 0
    total_lines = 0

    for file_path, counts in line_counts.items():
        print(f"\n{file_path}:")
        print(f"  Total: {counts['total']} lines")
        print(
            f"  UI Logic: {counts['ui_lines']} lines ({counts['ui_lines']/counts['total']*100:.1f}%)"
        )
        print(
            f"  Business Logic: {counts['business_lines']} lines ({counts['business_lines']/counts['total']*100:.1f}%)"
        )
        print(f"  Business Ratio: {counts['business_ratio']*100:.1f}%")

        total_ui_lines += counts["ui_lines"]
        total_business_lines += counts["business_lines"]
        total_lines += counts["total"]

    print(f"\n📈 OVERALL FRONTEND COMPOSITION:")
    print(f"  Total Lines: {total_lines}")
    print(f"  UI Logic: {total_ui_lines} lines ({total_ui_lines/total_lines*100:.1f}%)")
    print(
        f"  Business Logic: {total_business_lines} lines ({total_business_lines/total_lines*100:.1f}%)"
    )

    # Analyze violations
    print("\n🚨 ARCHITECTURE VIOLATIONS:")
    print("-" * 40)
    violations = analyze_frontend_violations()

    violation_counts = {}
    high_severity_count = 0

    for violation in violations:
        v_type = violation["type"]
        if v_type not in violation_counts:
            violation_counts[v_type] = 0
        violation_counts[v_type] += 1

        if violation["severity"] == "HIGH":
            high_severity_count += 1

    print(f"Total Violations: {len(violations)}")
    print(f"High Severity: {high_severity_count}")
    print("\nViolation Types:")
    for v_type, count in violation_counts.items():
        print(f"  {v_type}: {count} instances")

    # Show worst violations
    print(f"\n🔥 WORST VIOLATIONS (First 10):")
    print("-" * 40)
    high_violations = [v for v in violations if v["severity"] == "HIGH"][:10]

    for violation in high_violations:
        print(f"❌ {violation['file']}:{violation['line']}")
        print(f"   Type: {violation['type']}")
        print(f"   Code: {violation['content'][:100]}...")
        print()

    # Summary assessment
    print("=" * 80)
    print("🏁 ARCHITECTURE ASSESSMENT:")

    business_ratio = total_business_lines / total_lines

    if business_ratio > 0.3:
        print("❌ SEVERE ARCHITECTURE VIOLATIONS")
        print(f"   Frontend contains {business_ratio*100:.1f}% business logic")
        print("   This violates separation of concerns principles")
    elif business_ratio > 0.1:
        print("⚠️  MODERATE ARCHITECTURE ISSUES")
        print(f"   Frontend contains {business_ratio*100:.1f}% business logic")
    else:
        print("✅ CLEAN ARCHITECTURE")
        print(f"   Frontend contains only {business_ratio*100:.1f}% business logic")

    print(f"\nViolations Found: {len(violations)}")
    print(f"High Severity Issues: {high_severity_count}")

    if high_severity_count > 0:
        print("\n❌ IMMEDIATE ACTION REQUIRED")
        print(
            "   Move all strategy classes and calculation methods to src/backtesting/"
        )
        return False
    else:
        print("\n✅ ARCHITECTURE ACCEPTABLE")
        return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
