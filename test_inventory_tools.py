"""
Comprehensive Unit Tests for Deterministic Inventory Tools
===========================================================
Tests each deterministic Python tool:
  1. calculate_eoq()
  2. calculate_quantity_discount()
  3. calculate_rop() (All 4 operational cases)
  4. calculate_periodic_review()
  5. calculate_newsvendor()

Validates:
  - Input validation and rejection of missing inputs (never invents inputs)
  - Intermediate calculations
  - Final answer accuracy
  - Explicit assumption documentation
  - Structured output
"""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

import inventory_tools as it


class TestEOQTool(unittest.TestCase):
    """Tests for calculate_eoq()."""

    def test_eoq_exact_formula_and_intermediates(self):
        # D = 10,000, S = $50, H = $2.00
        # Q = sqrt(2 * 10000 * 50 / 2) = sqrt(500,000) = 707.10678...
        res = it.calculate_eoq(demand=10000, order_cost=50, holding_cost=2.0)

        # Check structured result
        self.assertIsInstance(res, it.EOQResult)
        self.assertIn("optimal_order_quantity_Q", res.final_answer)
        self.assertAlmostEqual(res.final_answer["optimal_order_quantity_Q"], 707.11, places=2)
        self.assertEqual(res.final_answer["optimal_order_quantity_rounded"], 707)

        # Check intermediate calculations
        self.assertIn("annual_ordering_cost", res.intermediate_calculations)
        self.assertIn("annual_holding_cost", res.intermediate_calculations)
        self.assertAlmostEqual(
            res.intermediate_calculations["annual_ordering_cost"],
            res.intermediate_calculations["annual_holding_cost"],
            delta=0.01,
        )

        # Check assumptions returned
        self.assertIsInstance(res.assumptions, list)
        self.assertGreater(len(res.assumptions), 3)

        # Check verification
        self.assertTrue(res.verification["holding_equals_ordering_at_eoq"])

    def test_eoq_with_holding_rate(self):
        # D = 12,000, S = $60, i = 20%, C = $50 -> H = 0.20 * 50 = $10
        # Q = sqrt(2 * 12000 * 60 / 10) = 379.47
        res = it.calculate_eoq(demand=12000, order_cost=60, holding_rate=0.20, unit_cost=50.0)
        self.assertAlmostEqual(res.final_answer["optimal_order_quantity_Q"], 379.47, places=2)
        self.assertEqual(res.inputs["holding_cost"], 10.0)

    def test_eoq_never_invents_missing_numerical_inputs(self):
        # Missing demand
        with self.assertRaises(ValueError) as ctx:
            it.calculate_eoq(order_cost=50, holding_cost=2.0)
        self.assertIn("Missing required input", str(ctx.exception))

        # Missing order cost
        with self.assertRaises(ValueError) as ctx:
            it.calculate_eoq(demand=1000, holding_cost=2.0)
        self.assertIn("Missing required input", str(ctx.exception))

        # Missing both holding cost and rate
        with self.assertRaises(ValueError) as ctx:
            it.calculate_eoq(demand=1000, order_cost=50)
        self.assertIn("Missing required input", str(ctx.exception))


class TestQuantityDiscountTool(unittest.TestCase):
    """Tests for calculate_quantity_discount()."""

    def setUp(self):
        self.schedule = [
            (1, 499, 10.0),
            (500, 999, 9.5),
            (1000, None, 9.0),
        ]
        self.demand = 5000
        self.order_cost = 49

    def test_quantity_discount_percentage_holding_cost(self):
        res = it.calculate_quantity_discount(
            demand=self.demand,
            order_cost=self.order_cost,
            discount_schedule=self.schedule,
            holding_rate=0.20,
        )
        self.assertEqual(res.inputs["holding_mode"], "percentage")
        self.assertEqual(res.final_answer["optimal_tier_index"], 2)
        self.assertEqual(res.final_answer["recommended_order_quantity"], 1000.0)
        self.assertGreater(res.final_answer["net_annual_savings_vs_baseline"], 0.0)

        # Intermediates and assumptions
        self.assertIn("tier_evaluations", res.intermediate_calculations)
        self.assertEqual(len(res.intermediate_calculations["tier_evaluations"]), 3)
        self.assertIsInstance(res.assumptions, list)

    def test_quantity_discount_fixed_holding_cost(self):
        # Constant holding cost H = $2.00 across all tiers
        res = it.calculate_quantity_discount(
            demand=self.demand,
            order_cost=self.order_cost,
            discount_schedule=self.schedule,
            holding_cost=2.0,
        )
        self.assertEqual(res.inputs["holding_mode"], "fixed")
        self.assertIsNotNone(res.final_answer["optimal_total_acquisition_cost"])
        self.assertTrue(res.verification["is_global_minimum"])

    def test_quantity_discount_never_invents_missing_inputs(self):
        with self.assertRaises(ValueError):
            it.calculate_quantity_discount(order_cost=49, discount_schedule=self.schedule, holding_rate=0.20)
        with self.assertRaises(ValueError):
            it.calculate_quantity_discount(demand=5000, order_cost=49, discount_schedule=[])
        with self.assertRaises(ValueError):
            it.calculate_quantity_discount(demand=5000, order_cost=49, discount_schedule=self.schedule)


class TestROPTool(unittest.TestCase):
    """Tests for calculate_rop() across all 4 operational cases."""

    def test_case_1_constant_demand_constant_lead_time(self):
        # Case 1: sigma_d = 0, sigma_L = 0
        # d = 10, L = 5 -> sigma_LT = 0, SS = 0, ROP = 50
        res = it.calculate_rop(daily_demand=10, lead_time=5, sigma_d=0.0, sigma_L=0.0)
        self.assertIn("Case 1", res.case_name)
        self.assertEqual(res.intermediate_calculations["std_lead_time_demand_sigma_LT"], 0.0)
        self.assertEqual(res.final_answer["safety_stock"], 0.0)
        self.assertEqual(res.final_answer["reorder_point"], 50.0)

    def test_case_2_variable_demand_constant_lead_time(self):
        # Case 2: sigma_d > 0, sigma_L = 0
        # d = 100, L = 9, sigma_d = 20, sigma_L = 0, CSL = 95%
        # sigma_LT = 20 * sqrt(9) = 60
        # z = 1.64485 -> SS = 1.64485 * 60 = 98.69 -> ROP = 900 + 98.69 = 998.69
        res = it.calculate_rop(daily_demand=100, lead_time=9, sigma_d=20, sigma_L=0.0, service_level=0.95)
        self.assertIn("Case 2", res.case_name)
        self.assertAlmostEqual(res.intermediate_calculations["std_lead_time_demand_sigma_LT"], 60.0, places=2)
        self.assertAlmostEqual(res.final_answer["safety_stock"], 98.69, places=2)
        self.assertAlmostEqual(res.final_answer["reorder_point"], 998.69, places=2)

    def test_case_3_constant_demand_variable_lead_time(self):
        # Case 3: sigma_d = 0, sigma_L > 0
        # d = 50, L = 6, sigma_d = 0, sigma_L = 2, CSL = 95%
        # sigma_LT = d * sigma_L = 50 * 2 = 100
        # z = 1.64485 -> SS = 164.49 -> ROP = 300 + 164.49 = 464.49
        res = it.calculate_rop(daily_demand=50, lead_time=6, sigma_d=0.0, sigma_L=2.0, service_level=0.95)
        self.assertIn("Case 3", res.case_name)
        self.assertAlmostEqual(res.intermediate_calculations["std_lead_time_demand_sigma_LT"], 100.0, places=2)
        self.assertAlmostEqual(res.final_answer["safety_stock"], 164.49, places=2)
        self.assertAlmostEqual(res.final_answer["reorder_point"], 464.49, places=2)

    def test_case_4_variable_demand_variable_lead_time(self):
        # Case 4: sigma_d > 0, sigma_L > 0
        # Formula: sigma_LT = sqrt(L * sigma_d^2 + d^2 * sigma_L^2)
        # d = 50, L = 4, sigma_d = 10, sigma_L = 1, CSL = 95%
        # sigma_LT = sqrt(4 * 100 + 2500 * 1) = sqrt(2900) = 53.8516
        # SS = 1.64485 * 53.8516 = 88.58
        # ROP = 200 + 88.58 = 288.58
        res = it.calculate_rop(daily_demand=50, lead_time=4, sigma_d=10, sigma_L=1.0, service_level=0.95)
        self.assertIn("Case 4", res.case_name)
        expected_sigma = math.sqrt(4 * (10 ** 2) + (50 ** 2) * (1.0 ** 2))
        self.assertAlmostEqual(res.intermediate_calculations["std_lead_time_demand_sigma_LT"], expected_sigma, places=4)
        self.assertAlmostEqual(res.final_answer["reorder_point"], (50 * 4) + (res.intermediate_calculations["z_score"] * expected_sigma), places=2)

    def test_rop_never_invents_missing_inputs(self):
        with self.assertRaises(ValueError):
            it.calculate_rop(lead_time=5)
        with self.assertRaises(ValueError):
            it.calculate_rop(daily_demand=50)


class TestPeriodicReviewTool(unittest.TestCase):
    """Tests for calculate_periodic_review()."""

    def test_periodic_review_formulas_and_intermediates(self):
        # T = 14 days, L = 4 days -> Protection Period = 18 days
        # d = 25, sigma_d = 5, CSL = 95% -> z = 1.64485
        # Expected Demand during protection = 25 * 18 = 450
        # sigma_protection = 5 * sqrt(18) = 21.2132
        # SS = 1.64485 * 21.2132 = 34.89
        # Target inventory S = 450 + 34.89 = 484.89
        # Inventory position = 85 (on hand) + 40 (on order) = 125
        # Order quantity = max(0, 484.89 - 125) = 359.89
        res = it.calculate_periodic_review(
            review_period=14,
            lead_time=4,
            demand_mean=25,
            demand_std=5,
            service_level=0.95,
            on_hand=85,
            on_order=40,
        )

        self.assertEqual(res.intermediate_calculations["protection_period"], 18.0)
        self.assertAlmostEqual(res.intermediate_calculations["expected_demand_protection_period"], 450.0, places=2)
        self.assertAlmostEqual(res.final_answer["target_inventory_S"], 484.89, places=2)
        self.assertAlmostEqual(res.final_answer["order_quantity"], 359.89, places=2)
        self.assertIsInstance(res.assumptions, list)

    def test_order_quantity_zero_when_inventory_position_exceeds_target(self):
        # High inventory position -> Order quantity should truncate at 0
        res = it.calculate_periodic_review(
            review_period=7,
            lead_time=2,
            demand_mean=10,
            demand_std=2,
            inventory_position=500,
        )
        self.assertEqual(res.final_answer["order_quantity"], 0.0)

    def test_periodic_review_never_invents_missing_inputs(self):
        with self.assertRaises(ValueError):
            it.calculate_periodic_review(lead_time=4, demand_mean=25)
        with self.assertRaises(ValueError):
            it.calculate_periodic_review(review_period=14, demand_mean=25)


class TestNewsvendorTool(unittest.TestCase):
    """Tests for calculate_newsvendor()."""

    def test_newsvendor_calculations_and_intermediates(self):
        # p = $100, c = $40, s = $20, g = $0
        # Cu = 100 - 40 = 60
        # Co = 40 - 20 = 20
        # CR = 60 / (60 + 20) = 0.75
        # mu = 1000, sigma = 200
        # z for 0.75 = 0.67449
        # Q* = 1000 + 0.67449 * 200 = 1134.898
        res = it.calculate_newsvendor(
            selling_price=100,
            unit_cost=40,
            salvage_value=20,
            demand_mean=1000,
            demand_std=200,
        )

        self.assertEqual(res.final_answer["underage_cost"], 60.0)
        self.assertEqual(res.final_answer["overage_cost"], 20.0)
        self.assertAlmostEqual(res.final_answer["critical_ratio"], 0.75, places=4)
        self.assertAlmostEqual(res.final_answer["recommended_order_quantity_Q"], 1134.90, places=2)
        self.assertEqual(res.final_answer["recommended_order_quantity_rounded"], 1135)
        self.assertGreater(res.final_answer["expected_profit"], 0.0)

        # Assumptions & verification
        self.assertIsInstance(res.assumptions, list)
        self.assertTrue(res.verification["critical_ratio_valid_bounds"])

    def test_unprofitable_product_rejected(self):
        # selling price <= unit cost
        with self.assertRaises(ValueError):
            it.calculate_newsvendor(selling_price=30, unit_cost=40, demand_mean=100, demand_std=15)

    def test_salvage_exceeding_cost_rejected(self):
        with self.assertRaises(ValueError):
            it.calculate_newsvendor(selling_price=100, unit_cost=40, salvage_value=50, demand_mean=100, demand_std=15)

    def test_newsvendor_never_invents_missing_inputs(self):
        with self.assertRaises(ValueError):
            it.calculate_newsvendor(unit_cost=40, demand_mean=1000, demand_std=200)
        with self.assertRaises(ValueError):
            it.calculate_newsvendor(selling_price=100, demand_mean=1000, demand_std=200)


class TestNaturalLanguageExtraction(unittest.TestCase):
    """Tests the agent's natural-language input extraction and synonym recognition."""

    def setUp(self):
        from inventory_agent import InventoryDecisionAgent
        self.agent = InventoryDecisionAgent()

    def test_eoq_ordering_cost_synonyms(self):
        # Must recognize all specified synonyms as ordering cost:
        # - ordering cost
        # - order cost
        # - setup cost
        # - setup expense
        # - cost per order
        # - cost of placing an order
        # - ordering/setup cost
        synonyms = [
            "Annual demand is 10000 units, ordering cost is $50, holding cost is $2.",
            "Annual demand is 10000 units, order cost is $50, holding cost is $2.",
            "Annual demand is 10000 units, setup cost is $50, holding cost is $2.",
            "Annual demand is 10000 units, setup expense is $50, holding cost is $2.",
            "Annual demand is 10000 units, cost per order is $50, holding cost is $2.",
            "Annual demand is 10000 units, cost of placing an order is $50, holding cost is $2.",
            "Annual demand is 10000 units, ordering/setup cost is $50, holding cost is $2.",
        ]
        for query in synonyms:
            decision = self.agent.solve(query)
            self.assertEqual(decision.extracted_inputs.get("order_cost"), 50.0)
            self.assertTrue(decision.verification_passed)

    def test_rop_distinctions_demand_vs_lead_time_variability(self):
        # Must carefully distinguish:
        # - average demand
        # - demand standard deviation
        # - average lead time
        # - lead-time standard deviation
        # - service level
        query = (
            "Average daily demand is 60 units with demand standard deviation of 12. "
            "Average lead time is 5 days with lead-time standard deviation of 1.5 days. "
            "Our cycle service level target is 95%."
        )
        decision = self.agent.solve(query)
        self.assertEqual(decision.extracted_inputs.get("daily_demand"), 60.0)
        self.assertEqual(decision.extracted_inputs.get("sigma_d"), 12.0)
        self.assertEqual(decision.extracted_inputs.get("lead_time"), 5.0)
        self.assertEqual(decision.extracted_inputs.get("sigma_L"), 1.5)
        self.assertEqual(decision.extracted_inputs.get("service_level"), 0.95)
        self.assertIn("Case 4", decision.raw_result.case_name)
        self.assertTrue(decision.verification_passed)

    def test_quantity_discount_distinctions(self):
        # Distinguish fixed holding cost vs percentage holding cost & brackets
        query_pct = "Annual demand 6000, setup cost $50, holding rate 20%. Discount tiers: 1-499 @ $10, 500+ @ $9."
        d_pct = self.agent.solve(query_pct)
        self.assertEqual(d_pct.extracted_inputs.get("holding_rate"), 0.20)
        self.assertEqual(d_pct.extracted_inputs.get("order_cost"), 50.0)
        self.assertEqual(len(d_pct.extracted_inputs.get("discount_schedule")), 2)

        query_fix = "Annual demand 6000, setup expense $50, fixed holding cost $2.00. Schedule: 1-499 at $10, 500+ at $9."
        d_fix = self.agent.solve(query_fix)
        self.assertEqual(d_fix.extracted_inputs.get("holding_cost"), 2.00)
        self.assertEqual(d_fix.extracted_inputs.get("order_cost"), 50.0)

    def test_fixed_period_review_distinctions(self):
        # Distinguish demand, review period, lead time, on-hand, on-order, backorders
        query = (
            "Review period is 14 days. Lead time is 4 days. Average daily demand is 25 with "
            "demand standard deviation of 5. Service level 98%. On-hand inventory is 80 units, "
            "on-order inventory is 40 units, and backorders are 10 units."
        )
        decision = self.agent.solve(query)
        self.assertEqual(decision.extracted_inputs.get("review_period"), 14.0)
        self.assertEqual(decision.extracted_inputs.get("lead_time"), 4.0)
        self.assertEqual(decision.extracted_inputs.get("demand_mean"), 25.0)
        self.assertEqual(decision.extracted_inputs.get("demand_std"), 5.0)
        self.assertEqual(decision.extracted_inputs.get("on_hand"), 80.0)
        self.assertEqual(decision.extracted_inputs.get("on_order"), 40.0)
        self.assertEqual(decision.extracted_inputs.get("backorders"), 10.0)
        # Inventory position = 80 + 40 - 10 = 110
        self.assertEqual(decision.extracted_inputs.get("inventory_position"), 110.0)
        self.assertTrue(decision.verification_passed)

    def test_newsvendor_distinctions(self):
        # Distinguish selling price, purchase cost, salvage value, mean demand, demand std
        query = (
            "A seasonal product has a selling price of $150, purchase cost of $65, and "
            "salvage value of $20. Demand distribution is normal with mean demand of 1800 "
            "and demand standard deviation of 400."
        )
        decision = self.agent.solve(query)
        self.assertEqual(decision.extracted_inputs.get("selling_price"), 150.0)
        self.assertEqual(decision.extracted_inputs.get("unit_cost"), 65.0)
        self.assertEqual(decision.extracted_inputs.get("salvage_value"), 20.0)
        self.assertEqual(decision.extracted_inputs.get("demand_mean"), 1800.0)
        self.assertEqual(decision.extracted_inputs.get("demand_std"), 400.0)
        self.assertTrue(decision.verification_passed)


class TestCSVDataAnalysis(unittest.TestCase):
    """Tests the CSV data processing, validation, and pandas analytical calculations."""

    def test_csv_analysis_calculations(self):
        import io
        import pandas as pd

        csv_content = """Product,Category,Units,Unit Cost
Widget A,Electronics,100,10.50
Widget B,Electronics,250,5.00
Gizmo X,Hardware,80,25.00
Gizmo Y,Hardware,300,12.00
Tool Z,Accessories,50,40.00"""

        df = pd.read_csv(io.StringIO(csv_content))
        required_cols = ["Product", "Category", "Units", "Unit Cost"]
        missing = [c for c in required_cols if c not in df.columns]
        self.assertEqual(len(missing), 0)

        # Calculations
        total_units = df["Units"].sum()
        self.assertEqual(total_units, 780)

        avg_unit_cost = df["Unit Cost"].mean()
        self.assertAlmostEqual(avg_unit_cost, 18.50, places=2)

        max_idx = df["Units"].idxmax()
        highest_product = df.loc[max_idx, "Product"]
        self.assertEqual(highest_product, "Gizmo Y")

        # Category group
        cat_units = df.groupby("Category")["Units"].sum().to_dict()
        self.assertEqual(cat_units["Electronics"], 350)
        self.assertEqual(cat_units["Hardware"], 380)
        self.assertEqual(cat_units["Accessories"], 50)

        # Inventory Value
        df["Inventory Value"] = df["Units"] * df["Unit Cost"]
        self.assertEqual(df.loc[0, "Inventory Value"], 1050.0)
        total_inventory_val = df["Inventory Value"].sum()
        # 100*10.5 + 250*5 + 80*25 + 300*12 + 50*40 = 1050 + 1250 + 2000 + 3600 + 2000 = 9900.0
        self.assertEqual(total_inventory_val, 9900.0)

    def test_csv_missing_columns_validation(self):
        import io
        import pandas as pd

        bad_csv = """Item,Category,Quantity
Widget A,Electronics,100"""
        df = pd.read_csv(io.StringIO(bad_csv))
        required_cols = ["Product", "Category", "Units", "Unit Cost"]
        missing = [c for c in required_cols if c not in df.columns]
        self.assertIn("Product", missing)
        self.assertIn("Units", missing)
        self.assertIn("Unit Cost", missing)
        self.assertNotIn("Category", missing)


class TestMulticriteriaClassification(unittest.TestCase):
    """Tests the multicriteria inventory classification (Sales & Hits ABCD stratification)."""

    def test_multicriteria_classification_logic(self):
        import pandas as pd
        from inventory_tools import classify_inventory_multicriteria

        # Create sample data with known distribution
        data = {
            "Item Number": ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"],
            "Item Name": ["Item A", "Item B", "Item C", "Item D", "Item E"],
            "UoM": ["EA", "EA", "CS", "BOX", "EA"],
            "Sales ($)": [60000.0, 20000.0, 10000.0, 7000.0, 3000.0],
            "Hits": [500, 300, 100, 70, 30],
        }
        df = pd.DataFrame(data)

        res = classify_inventory_multicriteria(df)

        # Totals verification
        self.assertEqual(res.total_items, 5)
        self.assertEqual(res.total_sales, 100000.0)
        self.assertEqual(res.total_hits, 1000)

        # Enriched dataframe checks
        classified_df = res.classified_df
        expected_cols = [
            "Item Number", "Item Name", "UoM", "Sales ($)", "Hits",
            "Sales Class", "Hits Class", "Combined Weighted Score", "Final Inventory Class"
        ]
        for col in expected_cols:
            self.assertIn(col, classified_df.columns)

        # Verify Item A (60% sales, 50% hits) is A for both -> Combined Score 4.0 -> Final 'A'
        item_a = classified_df[classified_df["Item Number"] == "SKU-001"].iloc[0]
        self.assertEqual(item_a["Sales Class"], "A")
        self.assertEqual(item_a["Hits Class"], "A")
        self.assertEqual(item_a["Final Inventory Class"], "A")
        self.assertEqual(item_a["Combined Weighted Score"], 4.0)

        # Verify 4x4 matrix dimensions
        self.assertEqual(res.matrix_distribution.shape, (4, 4))
        self.assertEqual(res.matrix_distribution.index.tolist(), ["A", "B", "C", "D"])
        self.assertEqual(res.matrix_distribution.columns.tolist(), ["A", "B", "C", "D"])

        # Verify class distribution contains A, B, C, D
        for c in ["A", "B", "C", "D"]:
            self.assertIn(c, res.class_distribution)

    def test_multicriteria_benchmark_outputs_and_top_5_items(self):
        import pandas as pd
        from inventory_tools import classify_inventory_multicriteria

        # 20-SKU Benchmark Dataset matching course specification
        benchmark_data = {
            "Item Number": [
                "SKU-1001", "SKU-1002", "SKU-1003", "SKU-1004", "SKU-1005",
                "SKU-1006", "SKU-1007", "SKU-1008", "SKU-1009", "SKU-1010",
                "SKU-1011", "SKU-1012", "SKU-1013", "SKU-1014", "SKU-1015",
                "SKU-1016", "SKU-1017", "SKU-1018", "SKU-1019", "SKU-1020"
            ],
            "Item Name": [
                "Premium Hydraulic Pump", "Industrial Ball Bearing", "Heavy Duty Conveyor Belt",
                "Electric Motor 5HP", "Control Valve Assembly", "Steel Roller Bearing",
                "Air Filter Cartridge", "Lubricant Synthetic 5Gal", "Flange Coupling 2in",
                "Pressure Gauge 100PSI", "Rubber O-Ring Pack 50", "Gasket Sheet Silicone",
                "Drive Shaft 1.5in", "Replacement Fuse 20A", "Hex Bolt M12 Pack 100",
                "Terminal Block 12-Way", "LED Indicator Lamp", "Hydraulic Seal Kit",
                "Relay 24VDC 10A", "Safety Limit Switch"
            ],
            "UoM": ["EA", "EA", "EA", "EA", "EA", "EA", "BX", "PL", "EA", "EA",
                    "PK", "EA", "EA", "BX", "BX", "EA", "EA", "KT", "EA", "EA"],
            "Sales ($)": [
                4694447, 4551669, 4347291, 4192948, 3507850,
                1278763, 1273914, 1251174, 1216696, 1090841,
                1084760, 996214, 1043602, 19392, 799098,
                1059416, 627430, 1001627, 658002, 552678
            ],
            "Hits": [
                186, 55, 317, 48, 170,
                76, 10, 7, 227, 308,
                97, 92, 86, 78, 77,
                42, 35, 29, 18, 14
            ]
        }
        df = pd.DataFrame(benchmark_data)
        res = classify_inventory_multicriteria(df)

        # 1. Verification of Totals
        self.assertEqual(res.total_items, 20)
        self.assertEqual(res.total_sales, 35247812.0)
        self.assertEqual(res.total_hits, 1972)

        # 2. Verification of Mergesort Preserved Sales Descending Order
        cdf = res.classified_df
        sales_vals = cdf["Sales ($)"].tolist()
        self.assertEqual(sales_vals, sorted(sales_vals, reverse=True))

        # 3. Verification of Specific Benchmark Classifications & Combined Scores
        item_map = cdf.set_index("Item Number").to_dict(orient="index")

        # SKU-1001: Sales A (4), Hits A (4) -> Combined 4.00 -> Final A
        self.assertEqual(item_map["SKU-1001"]["Sales Class"], "A")
        self.assertEqual(item_map["SKU-1001"]["Hits Class"], "A")
        self.assertEqual(item_map["SKU-1001"]["Combined Weighted Score"], 4.00)
        self.assertEqual(item_map["SKU-1001"]["Final Inventory Class"], "A")

        # SKU-1002: Sales A (4), Hits C (2) -> Combined 3.30 -> Final A
        self.assertEqual(item_map["SKU-1002"]["Sales Class"], "A")
        self.assertEqual(item_map["SKU-1002"]["Hits Class"], "C")
        self.assertEqual(item_map["SKU-1002"]["Combined Weighted Score"], 3.30)
        self.assertEqual(item_map["SKU-1002"]["Final Inventory Class"], "A")

        # SKU-1004: Sales A (4), Hits D (1) -> Combined 2.95 -> Final B
        self.assertEqual(item_map["SKU-1004"]["Sales Class"], "A")
        self.assertEqual(item_map["SKU-1004"]["Hits Class"], "D")
        self.assertEqual(item_map["SKU-1004"]["Combined Weighted Score"], 2.95)
        self.assertEqual(item_map["SKU-1004"]["Final Inventory Class"], "B")

        # SKU-1009: Sales B (3), Hits A (4) -> Combined 3.35 -> Final A
        self.assertEqual(item_map["SKU-1009"]["Sales Class"], "B")
        self.assertEqual(item_map["SKU-1009"]["Hits Class"], "A")
        self.assertEqual(item_map["SKU-1009"]["Combined Weighted Score"], 3.35)
        self.assertEqual(item_map["SKU-1009"]["Final Inventory Class"], "A")

        # SKU-1010: Sales B (3), Hits A (4) -> Combined 3.35 -> Final A
        self.assertEqual(item_map["SKU-1010"]["Sales Class"], "B")
        self.assertEqual(item_map["SKU-1010"]["Hits Class"], "A")
        self.assertEqual(item_map["SKU-1010"]["Combined Weighted Score"], 3.35)
        self.assertEqual(item_map["SKU-1010"]["Final Inventory Class"], "A")

        # SKU-1016: Sales C (2), Hits D (1) -> Combined 1.65 -> Final D
        self.assertEqual(item_map["SKU-1016"]["Sales Class"], "C")
        self.assertEqual(item_map["SKU-1016"]["Hits Class"], "D")
        self.assertEqual(item_map["SKU-1016"]["Combined Weighted Score"], 1.65)
        self.assertEqual(item_map["SKU-1016"]["Final Inventory Class"], "D")

        # SKU-1014: Sales D (1), Hits B (3) -> Combined 1.70 -> Final D
        self.assertEqual(item_map["SKU-1014"]["Sales Class"], "D")
        self.assertEqual(item_map["SKU-1014"]["Hits Class"], "B")
        self.assertEqual(item_map["SKU-1014"]["Combined Weighted Score"], 1.70)
        self.assertEqual(item_map["SKU-1014"]["Final Inventory Class"], "D")

        # 4. Verification of Portfolio Summary Table
        summary = res.portfolio_summary_table.set_index("Final Inventory Class")
        self.assertEqual(summary.loc["A", "Item Count"], 5)
        self.assertEqual(summary.loc["A", "Total Sales ($)"], 15900944.0)

        self.assertEqual(summary.loc["B", "Item Count"], 3)
        self.assertEqual(summary.loc["B", "Total Sales ($)"], 8979561.0)

        self.assertEqual(summary.loc["C", "Item Count"], 4)
        self.assertEqual(summary.loc["C", "Total Sales ($)"], 4653450.0)

        self.assertEqual(summary.loc["D", "Item Count"], 8)
        self.assertEqual(summary.loc["D", "Total Sales ($)"], 5713857.0)

        self.assertEqual(summary.loc["Total", "Item Count"], 20)
        self.assertEqual(summary.loc["Total", "Total Sales ($)"], 35247812.0)

        # 5. Verification of Top 5 Items by Sales ($) per Final Inventory Class
        top_a = res.top_5_items_by_class["A"]["Item Number"].tolist()
        self.assertEqual(top_a, ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1009", "SKU-1010"])

        top_b = res.top_5_items_by_class["B"]["Item Number"].tolist()
        self.assertEqual(top_b, ["SKU-1004", "SKU-1005", "SKU-1006"])

        top_c = res.top_5_items_by_class["C"]["Item Number"].tolist()
        self.assertEqual(top_c, ["SKU-1007", "SKU-1008", "SKU-1011", "SKU-1013"])

        top_d = res.top_5_items_by_class["D"]["Item Number"].tolist()
        self.assertEqual(top_d, ["SKU-1016", "SKU-1018", "SKU-1012", "SKU-1015", "SKU-1019"])

    def test_multicriteria_validation_errors(self):
        import pandas as pd
        from inventory_tools import classify_inventory_multicriteria

        # Missing sales column
        df_no_sales = pd.DataFrame({"Item": ["A"], "Hits": [100]})
        with self.assertRaises(ValueError):
            classify_inventory_multicriteria(df_no_sales)

        # Missing hits column
        df_no_hits = pd.DataFrame({"Item": ["A"], "Sales": [500.0]})
        with self.assertRaises(ValueError):
            classify_inventory_multicriteria(df_no_hits)

        # Zero total sales
        df_zero_sales = pd.DataFrame({"Item": ["A"], "Sales": [0.0], "Hits": [100]})
        with self.assertRaises(ValueError):
            classify_inventory_multicriteria(df_zero_sales)


class TestUnitConsistency(unittest.TestCase):
    """Tests automatic time unit conversions for demand, holding cost, review period, and lead time."""

    def test_convert_rate_to_annual_utility(self):
        # Month -> Year
        val, msg = it.convert_rate_to_annual(100.0, "month")
        self.assertEqual(val, 1200.0)
        self.assertIn("12 months/year", msg)

        # Week -> Year
        val, msg = it.convert_rate_to_annual(50.0, "week")
        self.assertEqual(val, 2600.0)
        self.assertIn("52 weeks/year", msg)

        # Day -> Year (250 work days)
        val, msg = it.convert_rate_to_annual(10.0, "day", work_days=250)
        self.assertEqual(val, 2500.0)
        self.assertIn("250 work days/year", msg)

        # Quarter -> Year
        val, msg = it.convert_rate_to_annual(300.0, "quarter")
        self.assertEqual(val, 1200.0)
        self.assertIn("4 quarters/year", msg)

        # Year -> Year (no conversion)
        val, msg = it.convert_rate_to_annual(5000.0, "year")
        self.assertEqual(val, 5000.0)
        self.assertIsNone(msg)

    def test_convert_duration_to_days_utility(self):
        # Weeks -> Days
        val, msg = it.convert_duration_to_days(2.0, "week", days_per_week=7.0)
        self.assertEqual(val, 14.0)
        self.assertIn("7.0 days/week", msg)

        # Days -> Days (no conversion)
        val, msg = it.convert_duration_to_days(5.0, "day")
        self.assertEqual(val, 5.0)
        self.assertIsNone(msg)

        # Months -> Days
        val, msg = it.convert_duration_to_days(1.5, "month")
        self.assertEqual(val, 45.0)
        self.assertIn("30.0 days/month", msg)

    def test_eoq_monthly_demand_auto_conversion(self):
        # Monthly demand = 1,000 -> Annual D = 12,000
        # S = $50, H = $2.00
        # Q* = sqrt(2 * 12000 * 50 / 2) = sqrt(600,000) = 774.5966...
        res = it.calculate_eoq(demand=1000, demand_unit="month", order_cost=50, holding_cost=2.0)
        self.assertEqual(res.D, 12000.0)
        self.assertEqual(res.S, 50.0)
        self.assertEqual(res.H, 2.0)
        self.assertAlmostEqual(res.Q_star, 774.60, places=2)
        self.assertEqual(res.eoq_rounded, 775)
        self.assertTrue(res.verification["holding_equals_ordering_at_eoq"])

    def test_eoq_weekly_demand_auto_conversion(self):
        # Weekly demand = 250 -> Annual D = 13,000
        # S = $60, H = $3.00
        # Q* = sqrt(2 * 13000 * 60 / 3) = sqrt(520,000) = 721.11...
        res = it.calculate_eoq(demand=250, demand_unit="week", order_cost=60, holding_cost=3.0)
        self.assertEqual(res.D, 13000.0)
        self.assertAlmostEqual(res.Q_star, 721.11, places=2)

    def test_eoq_monthly_holding_cost_auto_conversion(self):
        # D = 10,000, S = $50, monthly holding cost H = $0.25 -> Annual H = $3.00
        # Q* = sqrt(2 * 10000 * 50 / 3) = sqrt(333,333.33) = 577.35...
        res = it.calculate_eoq(demand=10000, order_cost=50, holding_cost=0.25, holding_cost_unit="month")
        self.assertEqual(res.H, 3.0)
        self.assertAlmostEqual(res.Q_star, 577.35, places=2)

    def test_eoq_lead_time_weeks_converted_to_days(self):
        # Lead time = 2 weeks -> 14 days
        # D = 10,000, S = $50, H = $2, work_days = 250 -> daily demand = 40
        # ROP = 40 * 14 = 560
        res = it.calculate_eoq(demand=10000, order_cost=50, holding_cost=2.0, lead_time_days=2.0, lead_time_unit="week", work_days=250)
        self.assertAlmostEqual(res.reorder_point, 560.0, places=2)

    def test_quantity_discount_monthly_demand_auto_conversion(self):
        # Monthly demand = 500 -> Annual D = 6,000
        schedule = [(1, 499, 10.0), (500, 999, 9.5), (1000, None, 9.0)]
        res = it.calculate_quantity_discount(
            demand=500,
            demand_unit="month",
            order_cost=49,
            discount_schedule=schedule,
            holding_rate=0.20,
        )
        self.assertEqual(res.D, 6000.0)
        self.assertEqual(res.S, 49.0)
        self.assertTrue(res.verification["is_global_minimum"])

    def test_rop_lead_time_weeks_converted_to_days(self):
        # Daily demand = 50, lead time = 2 weeks -> L = 14 days
        # sigma_d = 10, sigma_L = 0 (Case 2: Deterministic lead time)
        # SS = z * sigma_d * sqrt(L) = 1.64485 * 10 * sqrt(14) = 61.54
        # ROP = d * L + SS = 50 * 14 + 61.54 = 761.54
        res = it.calculate_rop(
            daily_demand=50,
            lead_time=2.0,
            lead_time_unit="week",
            sigma_d=10,
            service_level=0.95,
        )
        self.assertIn("Case 2", res.case_name)
        self.assertEqual(res.lead_time_days_mean, 14.0)
        expected_ss = 1.6448536269514722 * 10 * math.sqrt(14)
        self.assertAlmostEqual(res.SS, expected_ss, places=2)
        self.assertAlmostEqual(res.ROP, (50 * 14) + expected_ss, places=2)

    def test_periodic_review_review_period_in_weeks_converted_to_days(self):
        # Review period T = 2 weeks -> 14 days
        # Lead time L = 4 days
        # Protection period T + L = 18 days
        # d = 25, sigma_d = 5, CSL = 95% -> z = 1.64485
        # Expected demand = 25 * 18 = 450
        # SS = z * sigma_d * sqrt(T + L) = 1.64485 * 5 * sqrt(18) = 34.89
        # S_T = 450 + 34.89 = 484.89
        res = it.calculate_periodic_review(
            review_period=2.0,
            review_period_unit="week",
            lead_time=4.0,
            lead_time_unit="day",
            demand_mean=25,
            demand_std=5,
            service_level=0.95,
            inventory_position=120,
        )
        self.assertEqual(res.protection_period, 18.0)
        self.assertAlmostEqual(res.S_T, 484.89, places=2)
        self.assertAlmostEqual(res.Q, 484.89 - 120.0, places=2)


class TestEOQAndQuantityDiscountCourseFormulas(unittest.TestCase):
    """Tests course formulas for EOQ (H = iC, Q* = sqrt(2DS/H)) and Quantity Discounts."""

    def test_eoq_holding_cost_percentage_H_equals_iC(self):
        # D = 8,000, S = $40, C = $25, i = 20% (entered as 20 or 0.20)
        # H = i * C = 0.20 * 25 = $5.00
        # Q* = sqrt(2 * 8000 * 40 / 5) = sqrt(128,000) = 357.77
        res1 = it.calculate_eoq(demand=8000, order_cost=40, unit_cost=25.0, holding_rate=20.0)
        res2 = it.calculate_eoq(demand=8000, order_cost=40, unit_cost=25.0, holding_rate=0.20)

        self.assertEqual(res1.H, 5.0)
        self.assertEqual(res2.H, 5.0)
        self.assertEqual(res1.i, 0.20)
        self.assertEqual(res1.C, 25.0)
        self.assertAlmostEqual(res1.Q_star, 357.77, places=2)
        self.assertAlmostEqual(res2.Q_star, 357.77, places=2)
        # Check TAC = Total Acquisition Cost including purchase cost
        # TAC = DC + (D/Q)*S + (Q/2)*H
        # DC = 8000 * 25 = 200,000
        # Holding = (357.77 / 2) * 5 = 894.43
        # Ordering = (8000 / 357.77) * 40 = 894.43
        # TAC = 200,000 + 1788.85 = 201,788.85
        self.assertAlmostEqual(res1.TAC, 201788.85, places=1)

    def test_quantity_discount_tier_feasibility_and_boundary_evaluation(self):
        # D = 5,000, S = $49, i = 20%
        # Tier 0: 1 - 499 @ $12.00 -> H0 = 0.20 * 12 = 2.40 -> Q0* = sqrt(2*5000*49/2.4) = 451.85 (FEASIBLE in [1, 499])
        # Tier 1: 500 - 999 @ $11.00 -> H1 = 0.20 * 11 = 2.20 -> Q1* = sqrt(2*5000*49/2.2) = 471.94 (INFEASIBLE < 500 -> Evaluate boundary min_q = 500)
        # Tier 2: 1000+ @ $10.00 -> H2 = 0.20 * 10 = 2.00 -> Q2* = sqrt(2*5000*49/2.0) = 494.97 (INFEASIBLE < 1000 -> Evaluate boundary min_q = 1000)
        schedule = [
            (1, 499, 12.0),
            (500, 999, 11.0),
            (1000, None, 10.0),
        ]
        res = it.calculate_quantity_discount(
            demand=5000,
            order_cost=49,
            holding_rate=0.20,
            discount_schedule=schedule,
        )

        tier0 = res.tiers[0]
        self.assertTrue(tier0.is_eoq_feasible)
        self.assertFalse(tier0.is_breakpoint_used)
        self.assertAlmostEqual(tier0.raw_eoq, 451.85, places=2)
        self.assertAlmostEqual(tier0.feasible_qty, 451.85, places=2)

        tier1 = res.tiers[1]
        self.assertFalse(tier1.is_eoq_feasible)
        self.assertTrue(tier1.is_breakpoint_used)
        self.assertAlmostEqual(tier1.raw_eoq, 471.94, places=2)
        self.assertEqual(tier1.feasible_qty, 500.0)

        tier2 = res.tiers[2]
        self.assertFalse(tier2.is_eoq_feasible)
        self.assertTrue(tier2.is_breakpoint_used)
        self.assertAlmostEqual(tier2.raw_eoq, 494.97, places=2)
        self.assertEqual(tier2.feasible_qty, 1000.0)

        # Total Acquisition Cost formula check: TAC = DC + (D/Q)*S + (Q/2)*H
        # Tier 0 TAC: 5000*12 + (5000/451.85)*49 + (451.85/2)*2.4 = 60000 + 542.21 + 542.22 = 61,084.43
        self.assertAlmostEqual(tier0.total_acquisition_cost, 61084.43, places=1)
        # Tier 1 TAC: 5000*11 + (5000/500)*49 + (500/2)*2.2 = 55000 + 490 + 550 = 56,040.00
        self.assertAlmostEqual(tier1.total_acquisition_cost, 56040.00, places=2)
        # Tier 2 TAC: 5000*10 + (5000/1000)*49 + (1000/2)*2.0 = 50000 + 245 + 1000 = 51,245.00
        self.assertAlmostEqual(tier2.total_acquisition_cost, 51245.00, places=2)

        # Minimum TAC is Tier 2 at Q = 1000
        self.assertEqual(res.optimal_tier_index, 2)
        self.assertEqual(res.Q_star, 1000.0)
        self.assertAlmostEqual(res.TAC, 51245.00, places=2)


class TestSafetyStockAndROPCourseFormulas(unittest.TestCase):
    """Tests course formulas for Safety Stock & ROP."""

    def test_deterministic_lead_time_course_formula(self):
        # Case 2: Constant L, variable d
        # SS = z * sigma_d * sqrt(L)
        # ROP = d_bar * L + SS
        d = 80.0
        L = 5.0
        sigma_d = 15.0
        csl = 0.95
        z = NormalDist(mu=0.0, sigma=1.0).inv_cdf(csl)  # ~ 1.64485
        expected_ss = z * sigma_d * math.sqrt(L)
        expected_rop = (d * L) + expected_ss

        res = it.calculate_rop(daily_demand=d, lead_time=L, sigma_d=sigma_d, service_level=csl)
        self.assertAlmostEqual(res.SS, expected_ss, places=2)
        self.assertAlmostEqual(res.ROP, expected_rop, places=2)
        self.assertEqual(res.d_bar, d)
        self.assertEqual(res.L_bar, L)

    def test_variable_lead_time_course_formula(self):
        # Case 4: Variable d and Variable L
        # SS = z * sqrt(L_bar * sigma_d^2 + d_bar^2 * sigma_L^2)
        # ROP = d_bar * L_bar + SS
        d = 60.0
        L = 4.0
        sigma_d = 12.0
        sigma_L = 1.5
        csl = 0.98
        z = NormalDist(mu=0.0, sigma=1.0).inv_cdf(csl)  # ~ 2.05375
        sigma_lt = math.sqrt(L * (sigma_d ** 2) + (d ** 2) * (sigma_L ** 2))
        expected_ss = z * sigma_lt
        expected_rop = (d * L) + expected_ss

        res = it.calculate_rop(
            daily_demand=d,
            lead_time=L,
            sigma_d=sigma_d,
            sigma_L=sigma_L,
            service_level=csl,
        )
        self.assertAlmostEqual(res.intermediate_calculations["std_lead_time_demand_sigma_LT"], sigma_lt, places=4)
        self.assertAlmostEqual(res.SS, expected_ss, places=2)
        self.assertAlmostEqual(res.ROP, expected_rop, places=2)


class TestPeriodicReviewAndNewsvendorCourseFormulas(unittest.TestCase):
    """Tests course formulas for Periodic Review and Newsvendor."""

    def test_periodic_review_course_formulas(self):
        # Protection Period = T + L
        # S_T = d_bar * (T + L) + z * sigma_d * sqrt(T + L)
        # Q = max(0, S_T - IP)
        T = 14.0
        L = 4.0
        d = 25.0
        sigma_d = 5.0
        csl = 0.98
        z = NormalDist(mu=0.0, sigma=1.0).inv_cdf(csl)
        ip = 100.0

        protection_period = T + L
        expected_ss = z * sigma_d * math.sqrt(protection_period)
        expected_s_t = (d * protection_period) + expected_ss
        expected_q = max(0.0, expected_s_t - ip)

        res = it.calculate_periodic_review(
            review_period=T,
            lead_time=L,
            demand_mean=d,
            demand_std=sigma_d,
            service_level=csl,
            inventory_position=ip,
        )
        self.assertEqual(res.protection_period, protection_period)
        self.assertEqual(res.d_bar, d)
        self.assertEqual(res.IP, ip)
        self.assertAlmostEqual(res.S_T, expected_s_t, places=2)
        self.assertAlmostEqual(res.Q, expected_q, places=2)

    def test_newsvendor_course_formulas(self):
        # Cu = p - c
        # Co = c - v
        # CR = Cu / (Cu + Co)
        # Q* = mu + z * sigma
        p = 150.0
        c = 60.0
        v = 25.0
        mu = 2000.0
        sigma = 450.0

        cu = p - c  # 90.0
        co = c - v  # 35.0
        cr = cu / (cu + co)  # 90 / 125 = 0.72
        z = NormalDist(mu=0.0, sigma=1.0).inv_cdf(cr)
        expected_q = mu + (z * sigma)

        res = it.calculate_newsvendor(
            selling_price=p,
            unit_cost=c,
            salvage_value=v,
            demand_mean=mu,
            demand_std=sigma,
        )
        self.assertEqual(res.p, p)
        self.assertEqual(res.c, c)
        self.assertEqual(res.v, v)
        self.assertEqual(res.Cu, cu)
        self.assertEqual(res.Co, co)
        self.assertAlmostEqual(res.CR, cr, places=4)
        self.assertAlmostEqual(res.Q_star, expected_q, places=2)

    def test_newsvendor_with_goodwill_penalty(self):
        # Cu = p - c + g
        # Co = c - v
        # CR = Cu / (Cu + Co)
        p = 100.0
        c = 40.0
        v = 15.0
        g = 10.0
        mu = 1000.0
        sigma = 200.0

        cu = (p - c) + g  # 60 + 10 = 70.0
        co = c - v  # 25.0
        cr = cu / (cu + co)  # 70 / 95 = 0.73684...

        res = it.calculate_newsvendor(
            selling_price=p,
            unit_cost=c,
            salvage_value=v,
            goodwill_penalty=g,
            demand_mean=mu,
            demand_std=sigma,
        )
        self.assertEqual(res.Cu, 70.0)
        self.assertEqual(res.Co, 25.0)
        self.assertAlmostEqual(res.CR, cr, places=4)


class TestAgentUnitConsistencyEndToEnd(unittest.TestCase):
    """Tests the agent's end-to-end handling of unit consistency across natural-language queries."""

    def setUp(self):
        from inventory_agent import InventoryDecisionAgent
        self.agent = InventoryDecisionAgent()

    def test_nl_monthly_demand_to_annual_eoq(self):
        query = "Monthly demand is 1,000 units. Ordering cost is $50, and holding cost is $2 per unit/year."
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_EOQ)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.raw_result.D, 12000.0)
        self.assertAlmostEqual(decision.raw_result.Q_star, 774.60, places=2)

    def test_nl_rop_lead_time_in_weeks(self):
        query = "Daily demand is 50 units with standard deviation of 10. Supplier lead time is 2 weeks. Service level target is 95%."
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_SAFETY_STOCK_ROP)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.raw_result.lead_time_days_mean, 14.0)
        self.assertAlmostEqual(decision.raw_result.ROP, 761.54, places=1)

    def test_nl_periodic_review_review_period_in_weeks(self):
        query = (
            "We review inventory every 2 weeks. Supplier lead time is 4 days. "
            "Average daily demand is 25 units with a standard deviation of 5. "
            "Our service level target is 95%. Current inventory position is 120."
        )
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_FIXED_PERIOD_REVIEW)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.raw_result.protection_period, 18.0)
        self.assertAlmostEqual(decision.raw_result.S_T, 484.89, places=2)
        self.assertAlmostEqual(decision.raw_result.Q, 364.89, places=2)

    def test_eoq_demand_time_unit_extraction_and_conversion_test_6(self):
        query = "Demand is 1,000 units per month. Ordering cost is $50, and holding cost is $2 per unit/year."
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_EOQ)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.raw_result.D, 12000.0)
        self.assertEqual(decision.raw_result.annual_demand_D, 12000.0)
        self.assertAlmostEqual(decision.raw_result.Q_star, 774.60, places=2)
        self.assertTrue(any("Converted monthly demand to annual demand (x12)" in c for c in decision.raw_result.unit_conversions_applied))

    def test_safety_stock_rop_combined_variability_test_7(self):
        query = "Daily demand averages 40 units with a standard deviation of 8. Average lead time is 3 weeks with a standard deviation of 0.5 weeks. Target cycle service level is 95%."
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_SAFETY_STOCK_ROP)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.raw_result.L, 21.0)
        self.assertEqual(decision.raw_result.sigma_L, 3.5)
        self.assertAlmostEqual(decision.raw_result.SS, 238.04, places=2)
        self.assertAlmostEqual(decision.raw_result.safety_stock_SS, 238.04, places=2)
        self.assertAlmostEqual(decision.raw_result.ROP, 1078.04, places=2)
        self.assertAlmostEqual(decision.raw_result.reorder_point_ROP, 1078.04, places=2)

    def test_quantity_discount_model_routing_and_calculation_test_8(self):
        query = "The supplier charges $12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for orders of 1,000 units or more. Annual demand is 5,000 units, ordering cost is $60, and annual holding cost rate is 20% of unit cost."
        decision = self.agent.solve(query)
        self.assertEqual(decision.model_identified, self.agent.MODEL_QUANTITY_DISCOUNT)
        self.assertNotEqual(decision.model_identified, self.agent.MODEL_EOQ)
        self.assertTrue(decision.verification_passed)
        self.assertEqual(decision.extracted_inputs.get("discount_schedule_tier_count"), 3)
        self.assertEqual(decision.raw_result.discount_schedule_tier_count, 3)
        self.assertEqual(decision.extracted_inputs.get("discount_schedule"), [(1, 499, 12.0), (500, 999, 11.5), (1000, None, 11.0)])
        self.assertEqual(decision.raw_result.best_tier, 2)
        self.assertAlmostEqual(decision.raw_result.optimal_order_quantity, 1000.00, places=2)
        self.assertAlmostEqual(decision.raw_result.total_annual_cost, 56400.00, places=2)
        self.assertAlmostEqual(decision.raw_result.optimal_unit_price, 11.00, places=2)

    def test_quantity_discount_open_ended_top_tier_phrasing_variations(self):
        variations = [
            "The supplier charges $12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for 1,000+ units. Annual demand is 5,000 units, ordering cost is $60, and annual holding cost rate is 20% of unit cost.",
            "The supplier charges $12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for orders >= 1,000. Annual demand is 5,000 units, ordering cost is $60, and annual holding cost rate is 20% of unit cost.",
            "The supplier charges $12 per unit for orders below 500 units, $11.50 for 500-999 units, and $11 for at least 1,000 units. Annual demand is 5,000 units, ordering cost is $60, and annual holding cost rate is 20% of unit cost.",
            "The supplier charges $12 per unit for orders below 500, $11.50 for 500-999, and $11 for 1,000 or more. Annual demand is 5,000 units, ordering cost is $60, and annual holding cost rate is 20% of unit cost.",
        ]
        for query in variations:
            decision = self.agent.solve(query)
            self.assertEqual(decision.model_identified, self.agent.MODEL_QUANTITY_DISCOUNT)
            self.assertEqual(decision.extracted_inputs.get("discount_schedule_tier_count"), 3)
            self.assertEqual(decision.raw_result.discount_schedule_tier_count, 3)
            self.assertEqual(decision.raw_result.best_tier, 2)
            self.assertAlmostEqual(decision.raw_result.optimal_order_quantity, 1000.00, places=2)
            self.assertAlmostEqual(decision.raw_result.total_annual_cost, 56400.00, places=2)


if __name__ == "__main__":
    unittest.main()
