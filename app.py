"""
Supply Chain Analytics: Inventory Decision Agent Application
============================================================
Dual-Mode Application:
  1. Streamlit Web App: Run with `python -m streamlit run app.py` or `streamlit run app.py`
  2. Terminal CLI / Built-in Server: Run with `python app.py`

Models Supported:
  1. Economic Order Quantity (EOQ)
  2. Quantity Discount (All-Units)
  3. Safety Stock & Reorder Point (Continuous Review / (s, Q))
  4. Fixed-Period Review (Periodic Review / (T, S))
  5. Newsvendor (Single-Period Stochastic)

Architecture:
  Customer Question -> Identify Model -> Extract Inputs -> Select Tool -> Calculate -> Verify -> Explain
"""

from __future__ import annotations

import argparse
import http.server
import importlib
import io
import json
from pathlib import Path
import socketserver
import sys
import unittest
import urllib.parse
from typing import Any, Dict

# Ensure the app's directory is always first in sys.path
_app_dir = str(Path(__file__).parent.resolve())
if _app_dir not in sys.path:
    sys.path.insert(0, _app_dir)

import pandas as pd

# Hot-reload check to ensure clean restarts in Streamlit
import inventory_tools as it_mod
if not hasattr(it_mod, "classify_inventory_multicriteria"):
    it_mod = importlib.reload(it_mod)

from inventory_tools import (
    ClassificationResult,
    classify_inventory_multicriteria,
)

from inventory_agent import InventoryDecisionAgent


# ============================================================================
# PRE-LOADED CASE STUDIES
# ============================================================================

PRESET_CASE_STUDIES = [
    {
        "name": "1. Economic Order Quantity (EOQ)",
        "question": "Our annual demand is 12,000 units, ordering cost is $50 per order, and holding cost is $3 per unit/year. What is our optimal EOQ, order frequency, and cycle time?",
    },
    {
        "name": "2. All-Units Quantity Discount",
        "question": "Annual demand is 5,000 units, ordering cost is $49, and holding rate is 20%. The supplier offers the following discount schedule: 1-499 units at $12, 500-999 units at $11, and 1000+ units at $10. Which tier and order quantity minimizes total acquisition cost?",
    },
    {
        "name": "3. Safety Stock & Reorder Point (Continuous Review)",
        "question": "Daily demand averages 80 units with standard deviation 15. Supplier lead time is 5 days. What safety stock and reorder point do we need for a 95% cycle service level?",
    },
    {
        "name": "4. Fixed-Period Review (Periodic Review / (T, S))",
        "question": "We review inventory every 14 days. Supplier lead time is 4 days. Average daily demand is 25 units with a standard deviation of 5. Our service level target is 98%. We currently have 85 units on hand and 40 on order. What is our order-up-to level and how many units should we order today?",
    },
    {
        "name": "5. Newsvendor (Single-Period Stochastic)",
        "question": "We sell a seasonal winter jacket for $150 that costs $60 to purchase. Unsold coats can be liquidated at the end of the season for $25. Demand is normally distributed with mean 2,000 units and standard deviation 450. How many coats should we stock?",
    },
]


# ============================================================================
# STREAMLIT APPLICATION (Runs when invoked via `streamlit run app.py`)
# ============================================================================

def is_running_streamlit() -> bool:
    """Detects whether this script is currently being run by Streamlit."""
    try:
        import streamlit as st
        # Modern Streamlit runtime check
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            return True
    except Exception:
        pass
    # Fallback check on command line arguments
    if "streamlit" in sys.modules and any("streamlit" in arg for arg in sys.argv):
        return True
    return False


def render_streamlit_app():
    """Renders the comprehensive Streamlit Web Dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Supply Chain Inventory Decision Agent",
        page_icon="📦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    agent = InventoryDecisionAgent()

    # Header & Banner
    st.title("📦 Supply Chain Analytics: Inventory Decision Agent")
    st.markdown(
        "**7-Stage Decision Pipeline**: "
        "`Customer Question` ➔ `Identify Model` ➔ `Extract Inputs` ➔ "
        "`Select Python Tool` ➔ `Calculate` ➔ `Verify` ➔ `Explain`"
    )
    st.markdown("---")

    # Sidebar: Case Studies & Settings
    with st.sidebar:
        st.header("📚 Case Studies & Controls")
        st.markdown("Select a textbook case study or enter a custom query:")

        preset_options = ["(Custom Query)"] + [p["name"] for p in PRESET_CASE_STUDIES]
        selected_preset = st.selectbox("Pre-loaded Scenarios:", preset_options)

        default_text = ""
        if selected_preset != "(Custom Query)":
            for p in PRESET_CASE_STUDIES:
                if p["name"] == selected_preset:
                    default_text = p["question"]
                    break

        st.markdown("---")
        st.subheader("⚙️ Model Override (Optional)")
        model_override = st.selectbox(
            "Model Mode:",
            [
                "Auto-Detect from Question",
                "Economic Order Quantity (EOQ)",
                "Quantity Discount (All-Units)",
                "Safety Stock & Reorder Point",
                "Fixed-Period Review ((T, S))",
                "Single-Period Stochastic (Newsvendor)",
            ],
        )

        st.markdown("---")
        if st.button("🧪 Run Automated Unit Tests"):
            try:
                import tests.test_inventory_tools as test_mod
            except ImportError:
                import test_inventory_tools as test_mod
            suite = unittest.defaultTestLoader.loadTestsFromModule(test_mod)
            res = unittest.TextTestRunner(verbosity=0).run(suite)
            if res.wasSuccessful():
                st.success(f"✅ All {res.testsRun} unit tests passed successfully!")
            else:
                st.error(f"❌ {len(res.failures)} failures, {len(res.errors)} errors.")

        # Data Upload Section (Last section in the sidebar)
        st.markdown("---")
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    # Main Area: Input Form
    user_query = st.text_area(
        "Enter your Natural Language Inventory Question:",
        value=default_text,
        height=120,
        placeholder="e.g. Our annual demand is 12,000 units, ordering cost is $50, and holding cost is $3 per unit/year. What is our EOQ and cycle time?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        solve_btn = st.button("🚀 Solve Question", type="primary", use_container_width=True)

    if solve_btn and not user_query.strip():
        st.warning("Please enter an inventory question above or select a pre-loaded case study.")

    if user_query.strip():
        explicit_model = None
        if model_override == "Economic Order Quantity (EOQ)":
            explicit_model = agent.MODEL_EOQ
        elif model_override == "Quantity Discount (All-Units)":
            explicit_model = agent.MODEL_QUANTITY_DISCOUNT
        elif model_override == "Safety Stock & Reorder Point":
            explicit_model = agent.MODEL_SAFETY_STOCK_ROP
        elif model_override == "Fixed-Period Review ((T, S))":
            explicit_model = agent.MODEL_FIXED_PERIOD_REVIEW
        elif model_override == "Single-Period Stochastic (Newsvendor)":
            explicit_model = agent.MODEL_NEWSVENDOR

        with st.spinner("Processing through 7-Stage Agent Pipeline..."):
            decision = agent.solve(user_query, explicit_model=explicit_model)

        # Verification Status Header
        status_col1, status_col2 = st.columns([2, 2])
        with status_col1:
            st.info(f"🏷️ **Model Identified:** {decision.model_name}")
        with status_col2:
            if decision.verification_passed:
                st.success(f"✅ **Verification Status:** {decision.verification_details.get('status', 'PASSED')}")
            else:
                st.warning(f"⚠️ **Verification Status:** {decision.error or 'Needs Clarification'}")

        # Primary KPI Metrics
        raw = decision.raw_result
        if raw is not None:
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)

            if decision.model_identified == agent.MODEL_EOQ:
                mcol1.metric("Optimal EOQ (Q*)", f"{raw.eoq_rounded:,} units", f"{raw.eoq:.2f} exact")
                mcol2.metric("Orders per Year", f"{raw.orders_per_year:.2f}")
                mcol3.metric("Cycle Time", f"{raw.cycle_time_days:.1f} days")
                mcol4.metric("Total Inventory Cost", f"${raw.annual_inventory_cost:,.2f}")

            elif decision.model_identified == agent.MODEL_QUANTITY_DISCOUNT:
                mcol1.metric("Recommended Tier", f"Tier {raw.optimal_tier_index + 1}")
                mcol2.metric("Order Quantity", f"{raw.optimal_order_qty:,.0f} units")
                mcol3.metric("Unit Price", f"${raw.optimal_unit_price:,.2f}")
                mcol4.metric("Net Annual Savings", f"${raw.net_savings:,.2f}", f"{raw.net_savings_pct:.2f}%")

            elif decision.model_identified == agent.MODEL_SAFETY_STOCK_ROP:
                mcol1.metric("Reorder Point (ROP)", f"{raw.reorder_point_rounded:,} units", f"{raw.reorder_point:.2f} exact")
                mcol2.metric("Safety Stock (SS)", f"{raw.safety_stock_rounded:,} units", f"{raw.safety_stock:.2f} exact")
                mcol3.metric("Service Level (CSL)", f"{raw.service_level * 100:.1f}%", f"z = {raw.z_score:.2f}")
                mcol4.metric("Exp. Shortage / Cycle", f"{raw.expected_shortage_per_cycle:.4f} units")

            elif decision.model_identified == agent.MODEL_FIXED_PERIOD_REVIEW:
                mcol1.metric("Order-Up-To Level (S)", f"{raw.order_up_to_level_rounded:,} units")
                mcol2.metric("Order Quantity Now", f"{raw.recommended_order_quantity:,.0f} units")
                mcol3.metric("Safety Stock", f"{raw.safety_stock_rounded:,} units")
                mcol4.metric("Protection Period", f"{raw.protection_period_days:.0f} days (T+L)")

            elif decision.model_identified == agent.MODEL_NEWSVENDOR:
                mcol1.metric("Optimal Order (Q*)", f"{raw.optimal_quantity_rounded:,} units", f"{raw.optimal_quantity:.2f} exact")
                mcol2.metric("Critical Ratio (CR)", f"{raw.critical_ratio * 100:.2f}%", f"z = {raw.z_score:.2f}")
                mcol3.metric("Expected Profit", f"${raw.expected_profit:,.2f}")
                mcol4.metric("EVPI", f"${raw.expected_value_of_perfect_info:,.2f}")

        # Detailed Tabs
        tab_explain, tab_data, tab_verify, tab_json = st.tabs([
            "📋 Executive Explanation",
            "📊 Analytical Data Tables",
            "🛡️ Mathematical Verification",
            "🔍 Structured JSON Payload"
        ])

        with tab_explain:
            st.markdown(decision.explanation)

        with tab_data:
            if decision.model_identified == agent.MODEL_EOQ and raw is not None:
                st.subheader("Sensitivity Analysis (Deviations from Optimal EOQ)")
                table_rows = []
                for label, sdata in raw.sensitivity.items():
                    table_rows.append({
                        "Deviation": label,
                        "Order Quantity": f"{sdata['order_quantity']:,}",
                        "Ordering Cost": f"${sdata['order_cost']:,.2f}",
                        "Holding Cost": f"${sdata['holding_cost']:,.2f}",
                        "Total Inventory Cost": f"${sdata['total_inventory_cost']:,.2f}",
                        "Cost Penalty": f"{sdata['cost_penalty_pct']:+.2f}%"
                    })
                st.dataframe(table_rows, use_container_width=True)

            elif decision.model_identified == agent.MODEL_QUANTITY_DISCOUNT and raw is not None:
                st.subheader("Tier-by-Tier Evaluation Table")
                tier_rows = []
                for t in raw.tiers:
                    rng = f"[{t.min_qty:,}, {'∞' if t.max_qty is None else f'{t.max_qty:,}'}]"
                    adj = "Yes (Breakpoint)" if t.is_breakpoint_used else "No (Interior)"
                    tier_rows.append({
                        "Tier": f"Tier {t.tier_index + 1}" + (" ⭐ BEST" if t.tier_index == raw.optimal_tier_index else ""),
                        "Bracket Range": rng,
                        "Unit Price": f"${t.unit_price:,.2f}",
                        "Raw EOQ": f"{t.raw_eoq:,.1f}",
                        "Feasible Q": f"{t.feasible_qty:,.0f}",
                        "Breakpoint Used": adj,
                        "Annual Purchase Cost": f"${t.annual_purchase_cost:,.2f}",
                        "Annual Ordering Cost": f"${t.annual_order_cost:,.2f}",
                        "Annual Holding Cost": f"${t.annual_holding_cost:,.2f}",
                        "Total Cost (TAC)": f"${t.total_acquisition_cost:,.2f}"
                    })
                st.dataframe(tier_rows, use_container_width=True)
            else:
                st.json(decision.extracted_inputs)

        with tab_verify:
            st.subheader("Mathematical Verification Details")
            st.json(decision.verification_details)

        with tab_json:
            st.subheader("Complete Structured Decision Payload")
            st.json(decision.to_dict())

    # ========================================================================
    # CSV DATA UPLOAD & INVENTORY ANALYSIS / CLASSIFICATION SECTION
    # ========================================================================
    if uploaded_file is not None:
        st.markdown("---")

        try:
            # 1. Read CSV into a pandas dataframe using existing uploader
            df = pd.read_csv(uploaded_file)

            # Check if file has Multicriteria Classification columns (Sales & Hits)
            has_sales = any("sales" in c.lower() for c in df.columns)
            has_hits = any("hit" in c.lower() for c in df.columns)
            is_classification = has_sales and has_hits

            # Check if file has Basic Inventory Analysis columns (Product, Category, Units, Unit Cost)
            required_basic = ["Product", "Category", "Units", "Unit Cost"]
            is_basic = all(col in df.columns for col in required_basic)

            if is_classification:
                st.header("🏷️ Multicriteria Inventory Classification (Sales & Hits)")

                # Robust hot-reload fallback to guarantee clean restart in Streamlit
                try:
                    import inventory_tools as it_mod
                    if not hasattr(it_mod, "classify_inventory_multicriteria"):
                        it_mod = importlib.reload(it_mod)
                    classify_fn = getattr(it_mod, "classify_inventory_multicriteria", classify_inventory_multicriteria)
                except Exception:
                    classify_fn = classify_inventory_multicriteria

                clf_res = classify_fn(df)
                classified_df = clf_res.classified_df

                # DATA PROCESSING (Dimensions of the uploaded dataframe)
                st.subheader("📋 Dataset Overview")
                dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
                dcol1.metric("Number of Rows (SKUs)", f"{len(df):,}")
                dcol2.metric("Number of Columns", f"{len(df.columns):,}")
                dcol3.markdown(f"**Column Names:** `{', '.join(list(df.columns))}`")

                # Step 4: Display uploaded dataframe
                st.subheader("📄 Uploaded Data")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Multicriteria Classification KPIs
                st.subheader("📈 Stratification Summary (Sales & Hits)")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Items (SKUs)", f"{clf_res.total_items:,}")
                k2.metric("Total Sales", f"${clf_res.total_sales:,.2f}")
                k3.metric("Total Hits", f"{clf_res.total_hits:,.0f}")
                k4.metric("Class A Items", f"{clf_res.class_distribution.get('A', 0):,}", f"{(clf_res.class_distribution.get('A', 0)/clf_res.total_items)*100:.1f}% of SKUs")

                # Class Breakdown Metrics A, B, C, D
                c1, c2, c3, c4 = st.columns(4)
                for col, c_name in zip([c1, c2, c3, c4], ["A", "B", "C", "D"]):
                    cnt = clf_res.class_distribution.get(c_name, 0)
                    pct = (cnt / clf_res.total_items) * 100 if clf_res.total_items > 0 else 0
                    col.metric(f"Class {c_name} Items", f"{cnt:,}", f"{pct:.1f}% of catalog")

                # Setup column configuration for classified dataframe display
                col_config = {
                    "Item Number": st.column_config.TextColumn("Item Number"),
                    "Item Name": st.column_config.TextColumn("Item Name"),
                    "UoM": st.column_config.TextColumn("UoM"),
                    "Sales ($)": st.column_config.NumberColumn("Sales ($)", format="$%.2f"),
                    "Sales": st.column_config.NumberColumn("Sales", format="$%.2f"),
                    "Hits": st.column_config.NumberColumn("Hits", format="%d"),
                    "Cumulative Sales %": st.column_config.NumberColumn("Cumulative Sales %", format="%.2f%%"),
                    "Sales Class": st.column_config.TextColumn("Sales Class"),
                    "Hits Class": st.column_config.TextColumn("Hits Class"),
                    "Combined Weighted Score": st.column_config.NumberColumn("Combined Weighted Score", format="%.2f"),
                    "Final Inventory Class": st.column_config.TextColumn("Final Inventory Class"),
                }

                # 1. Main DataFrame Display (Preserving original Sales ($) descending order)
                st.subheader("📄 Classified Inventory Items (Sales Descending Order Preserved)")
                st.markdown(
                    "Main dataframe sorted by **Sales ($) descending** (mergesort preserved), with appended "
                    "**Sales Class**, **Hits Class**, **Combined Weighted Score**, and **Final Inventory Class**:"
                )
                st.dataframe(
                    classified_df,
                    use_container_width=True,
                    column_config=col_config,
                    hide_index=True,
                )

                # 2. Portfolio Summary Table (Grouped by Final Inventory Class A, B, C, D)
                st.subheader("📊 Portfolio Summary Table (Grouped by Final Inventory Class)")
                st.markdown(
                    "Portfolio summary grouped by **Final Inventory Class (A, B, C, D)** showing the exact "
                    "**Item Count** and **Total Sales ($)** per class, plus totals:"
                )
                st.dataframe(
                    clf_res.portfolio_summary_table,
                    use_container_width=True,
                    column_config={
                        "Final Inventory Class": st.column_config.TextColumn("Final Inventory Class"),
                        "Item Count": st.column_config.NumberColumn("Item Count", format="%d"),
                        "Total Sales ($)": st.column_config.NumberColumn("Total Sales ($)", format="$%.2f"),
                        "% of Items": st.column_config.NumberColumn("% of Items", format="%.2f%%"),
                        "% of Total Sales": st.column_config.NumberColumn("% of Total Sales", format="%.2f%%"),
                    },
                    hide_index=True,
                )

                # 3. Top 5 Items by Sales ($) for each Final Inventory Class (A, B, C, and D)
                st.subheader("🏆 Top 5 Items by Sales ($) for Each Final Inventory Class")
                st.markdown(
                    "Top 5 revenue-generating items within each **Final Inventory Class (A, B, C, D)**:"
                )
                tab_a, tab_b, tab_c, tab_d = st.tabs([
                    "⭐ Class A (Top 5)",
                    "⭐ Class B (Top 5)",
                    "⭐ Class C (Top 5)",
                    "⭐ Class D (Top 5)",
                ])

                for tab_obj, c_name in zip([tab_a, tab_b, tab_c, tab_d], ["A", "B", "C", "D"]):
                    with tab_obj:
                        top_df = clf_res.top_5_items_by_class.get(c_name)
                        if top_df is not None and not top_df.empty:
                            st.markdown(f"**Top {len(top_df)} items in Final Inventory Class {c_name} (by Sales):**")
                            st.dataframe(
                                top_df,
                                use_container_width=True,
                                column_config=col_config,
                                hide_index=True,
                            )
                        else:
                            st.info(f"No items categorized under Final Inventory Class {c_name}.")

                # 4. Multicriteria Matrix
                st.subheader("🧩 Multicriteria Matrix (Sales Class × Hits Class)")
                st.markdown("Distribution of items across **Sales Class (Velocity)** and **Hits Class (Transaction Frequency)**:")
                st.dataframe(clf_res.matrix_distribution, use_container_width=True)

                # 5. Download Enriched CSV
                csv_bytes = classified_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Multicriteria Classified Inventory CSV",
                    data=csv_bytes,
                    file_name="multicriteria_classified_inventory.csv",
                    mime="text/csv",
                )

            elif is_basic:
                st.header("📊 Uploaded CSV Data & Inventory Analysis")

                # Capture dimensions of the uploaded dataframe (4 columns: Product, Category, Units, Unit Cost)
                num_rows = len(df)
                num_cols = len(df.columns)
                col_names = list(df.columns)

                # DATA PROCESSING (Step 4 & 5: Dataset Overview on uploaded data)
                st.subheader("📋 Dataset Overview")
                dcol1, dcol2, dcol3 = st.columns([1, 1, 2])
                dcol1.metric("Number of Rows", f"{num_rows:,}")
                dcol2.metric("Number of Columns", f"{num_cols:,}")
                dcol3.markdown(f"**Column Names:** `{', '.join(col_names)}`")

                # Step 4: Display the uploaded dataframe (4 columns)
                st.dataframe(
                    df[["Product", "Category", "Units", "Unit Cost"]],
                    use_container_width=True,
                    column_config={
                        "Product": st.column_config.TextColumn("Product"),
                        "Category": st.column_config.TextColumn("Category"),
                        "Units": st.column_config.NumberColumn("Units", format="%d"),
                        "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                    },
                    hide_index=True,
                )

                # Convert Units and Unit Cost to numeric using pandas
                if df["Units"].dtype == object:
                    df["Units"] = pd.to_numeric(df["Units"].astype(str).str.replace(",", "").str.strip(), errors="coerce")
                else:
                    df["Units"] = pd.to_numeric(df["Units"], errors="coerce")

                if df["Unit Cost"].dtype == object:
                    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"].astype(str).str.replace("$", "").str.replace(",", "").str.strip(), errors="coerce")
                else:
                    df["Unit Cost"] = pd.to_numeric(df["Unit Cost"], errors="coerce")

                # DATA ANALYSIS USING PANDAS
                # 1. Total Units
                total_units = df["Units"].sum()

                # 2. Average Unit Cost
                avg_unit_cost = df["Unit Cost"].mean()

                # 3. Product with highest Units
                max_units_idx = df["Units"].idxmax()
                product_highest_units = df.loc[max_units_idx, "Product"]
                max_units_val = df.loc[max_units_idx, "Units"]

                # 4. Group data by Category and calculate total Units for each Category
                units_by_category = df.groupby("Category", as_index=False)["Units"].sum().sort_values(by="Units", ascending=False)

                # 5. Create new calculated column: Inventory Value = Units * Unit Cost
                df["Inventory Value"] = df["Units"] * df["Unit Cost"]

                # 6. Calculate total Inventory Value across all products
                total_inventory_value = df["Inventory Value"].sum()

                # RESULTS DISPLAY
                # Summary KPIs
                st.subheader("📈 Summary Metrics")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Total Units", f"{total_units:,.0f}")
                k2.metric("Average Unit Cost", f"${avg_unit_cost:,.2f}")
                k3.metric("Product with Highest Units", f"{product_highest_units}", f"{max_units_val:,.0f} units")
                k4.metric("Total Inventory Value", f"${total_inventory_value:,.2f}")

                # Total Units by Category
                st.subheader("📂 Total Units by Category")
                st.dataframe(
                    units_by_category,
                    use_container_width=True,
                    column_config={
                        "Category": st.column_config.TextColumn("Category"),
                        "Units": st.column_config.NumberColumn("Total Units", format="%d"),
                    },
                    hide_index=True,
                )

                # Display the dataframe including the new Inventory Value column
                st.subheader("📄 Uploaded Dataframe (with Calculated Inventory Value)")
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        "Product": st.column_config.TextColumn("Product"),
                        "Category": st.column_config.TextColumn("Category"),
                        "Units": st.column_config.NumberColumn("Units", format="%d"),
                        "Unit Cost": st.column_config.NumberColumn("Unit Cost", format="$%.2f"),
                        "Inventory Value": st.column_config.NumberColumn("Inventory Value", format="$%.2f"),
                    },
                    hide_index=True,
                )

            else:
                missing_cols = [col for col in required_basic if col not in df.columns]
                st.error(
                    f"❌ **Unrecognized CSV Format:**\n\n"
                    f"The uploaded CSV does not match either supported inventory schema:\n\n"
                    f"1. **Inventory Classification:** Requires `Item Number`, `Item Name`, `UoM`, `Sales ($)`, `Hits`\n"
                    f"2. **Basic Inventory Analysis:** Missing `{', '.join(missing_cols)}` (Requires: `{', '.join(required_basic)}`)"
                )

        except Exception as e:
            st.error(f"❌ Error reading or processing CSV file: {str(e)}")


# ============================================================================
# CLI RUNNER & REPL
# ============================================================================

def print_banner():
    print("=" * 78)
    print("  SUPPLY CHAIN ANALYTICS: INVENTORY DECISION AGENT")
    print("  Models: EOQ | Quantity Discount | Safety Stock & ROP | (T, S) | Newsvendor")
    print("  Architecture: Question -> Identify -> Extract -> Tool -> Calc -> Verify -> Explain")
    print("=" * 78)


def run_single_query(agent: InventoryDecisionAgent, question: str):
    print("\n[Input Question]")
    print(f"  {question}\n")
    print("--- Executing 7-Stage Decision Pipeline ---")
    decision = agent.solve(question)

    print(f"1. Model Identified : {decision.model_name} ({decision.model_identified})")
    print(f"2. Extracted Inputs  : {json.dumps(decision.extracted_inputs, default=str)}")
    if decision.missing_inputs:
        print(f"   Missing Inputs   : {decision.missing_inputs}")
    print(f"3. Verification     : {'PASSED' if decision.verification_passed else 'FAILED / PENDING'}")
    print(f"   Status Details   : {decision.verification_details.get('status', 'N/A')}")
    print("\n" + "=" * 78)
    print(decision.explanation)
    print("=" * 78 + "\n")


def run_interactive_cli(agent: InventoryDecisionAgent):
    print_banner()
    while True:
        print("\nMain Menu:")
        print("  [1] Run Pre-loaded Case Study 1: Economic Order Quantity (EOQ)")
        print("  [2] Run Pre-loaded Case Study 2: Quantity Discount (All-Units)")
        print("  [3] Run Pre-loaded Case Study 3: Safety Stock & Reorder Point")
        print("  [4] Run Pre-loaded Case Study 4: Fixed-Period Review (Periodic / (T,S))")
        print("  [5] Run Pre-loaded Case Study 5: Single-Period Stochastic (Newsvendor)")
        print("  [6] Enter Custom Natural Language Inventory Question")
        print("  [7] Run Full Test Suite")
        print("  [Q] Exit")

        choice = input("\nSelect an option [1-7 or Q]: ").strip()
        if choice.lower() in ["q", "exit", "quit"]:
            print("Exiting Supply Chain Decision Agent. Goodbye!")
            break
        elif choice in ["1", "2", "3", "4", "5"]:
            preset = PRESET_CASE_STUDIES[int(choice) - 1]
            print(f"\n--- Loading {preset['name']} ---")
            run_single_query(agent, preset["question"])
        elif choice == "6":
            user_q = input("\nEnter your inventory question:\n> ").strip()
            if user_q:
                run_single_query(agent, user_q)
        elif choice == "7":
            print("\nRunning unit test suite...")
            import tests.test_inventory_tools as test_mod
            suite = unittest.defaultTestLoader.loadTestsFromModule(test_mod)
            unittest.TextTestRunner(verbosity=2).run(suite)
        else:
            print("Invalid option. Please enter 1-7 or Q.")


# ============================================================================
# BUILT-IN FALLBACK WEB SERVER (http.server)
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Supply Chain Analytics: Inventory Decision Agent</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #0f172a; padding: 24px; margin: 0; }
    .container { max-width: 1000px; margin: 0 auto; }
    header { background: #1e3a8a; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 18px; margin-bottom: 16px; }
    textarea { width: 100%; height: 90px; padding: 10px; border: 1px solid #cbd5e1; border-radius: 6px; box-sizing: border-box; }
    .btn { background: #1e3a8a; color: white; border: none; padding: 8px 16px; border-radius: 6px; font-weight: 600; cursor: pointer; }
    .btn:hover { background: #2563eb; }
    .output { background: #fafafa; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; white-space: pre-wrap; font-size: 14px; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h2>📦 Supply Chain Analytics: Inventory Decision Agent</h2>
      <p>Natural-language inventory decision solver with deterministic calculations.</p>
    </header>
    <div class="card">
      <h3>💬 Ask Question</h3>
      <textarea id="q" placeholder="Enter inventory question..."></textarea><br><br>
      <button class="btn" onclick="ask()">Solve</button>
    </div>
    <div class="card">
      <h3>📋 Decision Output</h3>
      <div id="out" class="output">Ready. Enter a question and click Solve.</div>
    </div>
  </div>
  <script>
    async function ask() {
      const q = document.getElementById('q').value.trim();
      if(!q) return;
      document.getElementById('out').innerText = "Processing...";
      const r = await fetch('/api/solve', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({question: q})
      });
      const d = await r.json();
      document.getElementById('out').innerText = d.explanation || JSON.stringify(d, null, 2);
    }
  </script>
</body>
</html>
"""

class AgentHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    agent = InventoryDecisionAgent()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/presets":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(PRESET_CASE_STUDIES).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/solve":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            try:
                payload = json.loads(post_data.decode("utf-8"))
                question = payload.get("question", "")
                explicit_inputs = payload.get("inputs")
                explicit_model = payload.get("model")

                decision = self.agent.solve(
                    question=question,
                    explicit_model=explicit_model,
                    explicit_inputs=explicit_inputs,
                )
                response_json = json.dumps(decision.to_dict(), default=str)

                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(response_json.encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def run_web_server(port: int = 8080):
    server_address = ("", port)
    with socketserver.TCPServer(server_address, AgentHTTPRequestHandler) as httpd:
        print_banner()
        print(f"\n🌐 Web Server Running at: http://localhost:{port}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down web server...")


# ============================================================================
# ENTRY POINT
# ============================================================================

# If Streamlit is running this file, render Streamlit UI immediately!
if is_running_streamlit():
    render_streamlit_app()
else:
    def main():
        parser = argparse.ArgumentParser(description="Supply Chain Analytics Inventory Decision Agent")
        parser.add_argument("query", nargs="?", type=str, help="Natural language inventory question to solve")
        parser.add_argument("--web", nargs="?", const=8080, type=int, help="Start web interface on port (default: 8080)")
        parser.add_argument("--test", action="store_true", help="Run the full test suite")
        parser.add_argument("--interactive", action="store_true", help="Run interactive CLI menu")

        args = parser.parse_args()
        agent = InventoryDecisionAgent()

        if args.test:
            try:
                import tests.test_inventory_tools as test_mod
            except ImportError:
                import test_inventory_tools as test_mod
            suite = unittest.defaultTestLoader.loadTestsFromModule(test_mod)
            runner = unittest.TextTestRunner(verbosity=2)
            res = runner.run(suite)
            sys.exit(0 if res.wasSuccessful() else 1)

        if args.web is not None:
            run_web_server(port=args.web)
            return

        if args.query:
            run_single_query(agent, args.query)
            return

        run_interactive_cli(agent)

    if __name__ == "__main__":
        main()
