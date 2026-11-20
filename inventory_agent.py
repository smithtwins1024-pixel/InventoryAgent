"""
Supply Chain Analytics: Natural-Language Inventory Decision Agent
==================================================================
Implements the 7-stage Decision Pipeline:
  Customer Question
       ↓
  Identify Model
       ↓
  Extract Inputs (Multi-pass semantic & synonym extractor)
       ↓
  Select Python Tool
       ↓
  Calculate
       ↓
  Verify
       ↓
  Explain

All numerical calculations are executed deterministically by `inventory_tools.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
import sys
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure the agent's directory is always first in sys.path
_agent_dir = str(Path(__file__).parent.resolve())
if _agent_dir not in sys.path:
    sys.path.insert(0, _agent_dir)

import inventory_tools as it
if not hasattr(it, "classify_inventory_multicriteria"):
    it = importlib.reload(it)

try:
    from inventory_tools import classify_inventory_multicriteria, ClassificationResult
except Exception:
    pass


# ============================================================================
# PIPELINE RESULT DATA STRUCTURE
# ============================================================================

@dataclass
class AgentDecision:
    question: str
    model_identified: str
    model_name: str
    extracted_inputs: Dict[str, Any]
    missing_inputs: List[str] = field(default_factory=list)
    raw_result: Any = None
    structured_output: Dict[str, Any] = field(default_factory=dict)
    verification_passed: bool = False
    verification_details: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "model_identified": self.model_identified,
            "model_name": self.model_name,
            "extracted_inputs": self.extracted_inputs,
            "missing_inputs": self.missing_inputs,
            "verification_passed": self.verification_passed,
            "verification_details": self.verification_details,
            "explanation": self.explanation,
            "error": self.error,
            "structured_output": self.structured_output,
        }


# ============================================================================
# ENHANCED NATURAL LANGUAGE INPUT EXTRACTOR
# ============================================================================

class InventoryInputExtractor:
    """
    Multi-pass semantic extractor recognizing standard supply chain terminology,
    synonyms, unit labels, currencies, and schedule tables.
    """

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalizes numbers with commas (e.g. 10,000 -> 10000) and standardizes dashes."""
        text = re.sub(r'(\d+),(\d+)', r'\1\2', text)
        text = text.replace("–", "-").replace("—", "-")
        return text

    @classmethod
    def extract_first_match(cls, patterns: List[str], text: str) -> Optional[float]:
        """Runs an ordered list of regex patterns and returns the first valid float matched."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    return float(match.group(1))
                except (ValueError, IndexError):
                    continue
        return None

    @classmethod
    def extract_percentage(cls, patterns: List[str], text: str) -> Optional[float]:
        """Extracts percentages and returns a decimal fraction between 0.0 and 1.0."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    val = float(match.group(1))
                    if val > 1.0:
                        return val / 100.0
                    return val
                except (ValueError, IndexError):
                    continue
        return None

    @classmethod
    def extract_discount_schedule(cls, text: str) -> List[Tuple[int, Optional[int], float]]:
        """
        Parses tiered price breaks:
          - "charges $12 per unit for orders below 500, $11.50 for 500-999, and $11 for 1,000 or more"
          - "<500: $12, 500-999: $11.50, >=1000: $11"
          - "1-499: $10, 500-999: $9.50, 1000+: $9.00"
          - "1 to 499 units at $12, 500 to 999 at $11, 1000 and above at $10"
        """
        text = cls.clean_text(text)
        tiers: List[Tuple[int, Optional[int], float]] = []

        # 1. Price before 'below/under/less than/< Y'
        # e.g., 'charges $12 per unit for orders below 500'
        for m in re.finditer(
            r'(?:charges?|costs?|price\s*is)?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*(?:for\s+)?(?:orders?\s+)?(?:below|under|less\s+than|<)\s*([0-9]+)',
            text,
            re.IGNORECASE,
        ):
            p = float(m.group(1))
            hi = int(m.group(2)) - 1
            tiers.append((1, hi, p))

        # 2. 'below/under/less than/< Y at/cost/: $price'
        # e.g., '<500: $12' or 'orders below 500: $12' or 'below 500 at $12'
        for m in re.finditer(
            r'(?:orders?\s+)?(?:below|under|less\s+than|<)\s*([0-9]+)\s*(?:units?)?\s*(?:at|@|:|\$|\s+for|\s+costs?|\s+charges?)\s*(?:cost\s+)?\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE,
        ):
            hi = int(m.group(1)) - 1
            p = float(m.group(2))
            if not any(t[1] == hi and t[2] == p for t in tiers):
                tiers.append((1, hi, p))

        # 3. Price before 'min-max'
        # e.g., '$11.50 for 500-999' or '$11.50 for orders of 500 to 999'
        for m in re.finditer(
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*for\s*(?:orders?\s+(?:of\s+)?)?([0-9]+)\s*(?:-|to)\s*([0-9]+)',
            text,
            re.IGNORECASE,
        ):
            p = float(m.group(1))
            lo = int(m.group(2))
            hi = int(m.group(3))
            if not any(t[0] == lo and t[1] == hi for t in tiers):
                tiers.append((lo, hi, p))

        # 4. 'min-max at/:/$ $price'
        # e.g., '500-999: $11.50' or '500 to 999 units at $11'
        for m in re.finditer(
            r'([0-9]+)\s*(?:-|to)\s*([0-9]+)\s*(?:units?)?\s*(?:at|@|:|\$)\s*\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE,
        ):
            lo = int(m.group(1))
            hi = int(m.group(2))
            p = float(m.group(3))
            if not any(t[0] == lo and t[1] == hi for t in tiers):
                tiers.append((lo, hi, p))

        # 5. Price before open-ended tier:
        # 5a. '$11 for orders of 1,000 units or more', '$11 for 1000 units and above', '$11 for 1000 or more'
        for m in re.finditer(
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*for\s*(?:orders?\s+(?:of\s+)?)?([0-9]+)\s*(?:units?|items?|pieces?)?\s*(?:or\s+more|and\s+above|and\s+over|\+|or\s+over)',
            text,
            re.IGNORECASE,
        ):
            p = float(m.group(1))
            lo = int(m.group(2))
            if not any(t[0] == lo and t[1] is None for t in tiers):
                tiers.append((lo, None, p))

        # 5b. '$11 for 1,000+ units' or '$11 for orders of 1,000+ units'
        for m in re.finditer(
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*for\s*(?:orders?\s+(?:of\s+)?)?([0-9]+)\s*\+\s*(?:units?|items?|pieces?)?',
            text,
            re.IGNORECASE,
        ):
            p = float(m.group(1))
            lo = int(m.group(2))
            if not any(t[0] == lo and t[1] is None for t in tiers):
                tiers.append((lo, None, p))

        # 5c. '$11 for orders >= 1,000' or '$11 for at least 1,000 units' or '$11 for minimum 1,000 units'
        for m in re.finditer(
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s*unit|each)?\s*for\s*(?:orders?\s+(?:of\s+)?)?(?:>=|>|at\s+least|minimum\s+of|minimum|min\.?\s+of|min\.?)\s*([0-9]+)\s*(?:units?|items?|pieces?)?',
            text,
            re.IGNORECASE,
        ):
            p = float(m.group(1))
            lo = int(m.group(2))
            if not any(t[0] == lo and t[1] is None for t in tiers):
                tiers.append((lo, None, p))

        # 6. Open-ended tier before price:
        # 6a. '>= 1000 units at $11' or 'at least 1000 units at $11' or 'orders >= 1000: $11'
        for m in re.finditer(
            r'(?:orders?\s+(?:of\s+)?)?(?:>=|>|at\s+least|minimum\s+of|minimum|min\.?\s+of|min\.?)\s*([0-9]+)\s*(?:units?|items?|pieces?)?\s*(?:at|@|:|\$|\s+for|\s+costs?|\s+charges?)\s*(?:cost\s+)?\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE,
        ):
            lo = int(m.group(1))
            p = float(m.group(2))
            if not any(t[0] == lo and t[1] is None for t in tiers):
                tiers.append((lo, None, p))

        # 6b. '1000+ units at $11' or '1000 or more units at $11' or '1000 units or more: $11'
        for m in re.finditer(
            r'(?:orders?\s+(?:of\s+)?)?([0-9]+)\s*(?:units?|items?|pieces?)?\s*(?:\+|and\s+above|or\s+more|and\s+over|\s*plus)\s*(?:units?|items?|pieces?)?\s*(?:at|@|:|\$|\s+for|\s+costs?|\s+charges?)\s*(?:cost\s+)?\$?([0-9]+(?:\.[0-9]+)?)',
            text,
            re.IGNORECASE,
        ):
            lo = int(m.group(1))
            p = float(m.group(2))
            if not any(t[0] == lo and t[1] is None for t in tiers):
                tiers.append((lo, None, p))

        # Deduplicate and sort
        unique_tiers = []
        seen = set()
        for t in tiers:
            key = (t[0], t[1])
            if key not in seen:
                seen.add(key)
                unique_tiers.append(t)

        unique_tiers.sort(key=lambda t: t[0])
        return unique_tiers

    # ------------------------------------------------------------------------
    # 1. EOQ INPUT EXTRACTION
    # ------------------------------------------------------------------------
    @classmethod
    def extract_eoq(cls, text: str) -> Dict[str, Any]:
        """
        Extracts EOQ parameters recognizing ordering cost synonyms:
          - ordering cost
          - order cost
          - setup cost
          - setup expense
          - cost per order
          - cost of placing an order
          - ordering/setup cost
          - ordering / setup cost
        Automatically detects and converts time units (e.g. monthly demand to annual, weeks to days).
        """
        inputs: Dict[str, Any] = {}

        # 1. Demand D (Detect unit: monthly, weekly, daily, annual)
        monthly_demand = cls.extract_first_match([
            r'(?:monthly\s+demand|demand\s+per\s+month|monthly\s+sales|demand\s+each\s+month)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+month|/\s*month|monthly|a\s+month|each\s+month|per\s+mo|/\s*mo)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+month|/\s*month|monthly|a\s+month|each\s+month|per\s+mo|/\s*mo)',
        ], text)
        weekly_demand = cls.extract_first_match([
            r'(?:weekly\s+demand|demand\s+per\s+week|weekly\s+sales|demand\s+each\s+week)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+week|/\s*week|weekly|a\s+week|each\s+week|per\s+wk|/\s*wk)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+week|/\s*week|weekly|a\s+week|each\s+week|per\s+wk|/\s*wk)',
        ], text)
        daily_demand = cls.extract_first_match([
            r'(?:daily\s+demand|demand\s+per\s+day|daily\s+sales|demand\s+each\s+day)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+day|/\s*day|daily|a\s+day|each\s+day)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+day|/\s*day|daily|a\s+day|each\s+day)',
        ], text)
        annual_demand = cls.extract_first_match([
            r'(?:annual\s+demand|yearly\s+demand|annual\s+sales|demand\s+per\s+year)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+year|/\s*year|annually|yearly|a\s+year|per\s+yr|/\s*yr)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+year|/\s*year|annually|yearly|a\s+year)',
            r'(?:we\s+sell|sells?|selling)\s+([0-9]+(?:\.[0-9]+)?)\s*(?:units)?\s*(?:per\s+year|annually|a\s+year)?',
            r'demands?\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)

        if monthly_demand is not None:
            inputs["original_demand"] = monthly_demand
            inputs["demand_unit"] = "month"
            inputs["demand"] = monthly_demand * 12.0
            inputs["annual_demand_D"] = monthly_demand * 12.0
        elif weekly_demand is not None:
            inputs["original_demand"] = weekly_demand
            inputs["demand_unit"] = "week"
            inputs["demand"] = weekly_demand * 52.0
            inputs["annual_demand_D"] = weekly_demand * 52.0
        elif daily_demand is not None:
            inputs["original_demand"] = daily_demand
            inputs["demand_unit"] = "day"
            inputs["demand"] = daily_demand * 250.0
            inputs["annual_demand_D"] = daily_demand * 250.0
        elif annual_demand is not None:
            inputs["demand"] = annual_demand
            inputs["demand_unit"] = "year"
            inputs["annual_demand_D"] = annual_demand

        # 2. Ordering / Setup Cost S (comprehensive synonyms)
        order_cost = cls.extract_first_match([
            r'(?:each\s+)?setup\s+costs?(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'(?:each\s+)?order\s+costs?(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'costs?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:to place|per order|per setup|to order|each time)',
            r'(?:ordering/setup cost|ordering\s*/\s*setup cost|ordering and setup cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'(?:cost of placing an order|cost to place an order|cost of an order|cost to place each order)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'(?:cost per order|expense per order|charge per order|fee per order)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'(?:setup costs?|setup expenses?|setup fees?|setup charges?)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'(?:ordering costs?|order costs?|order expenses?)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per order|per setup|to place each order|to place an order|to place|setup cost|setup expense|cost per order)',
        ], text)
        if order_cost is not None:
            inputs["order_cost"] = order_cost

        # 3. Holding Rate i (Percentage) - extract first so 20% is not misparsed as $2.00
        holding_rate = cls.extract_percentage([
            r'(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|annual\s+holding\s+cost\s+rate|annual\s+holding\s+rate|carrying\s+cost\s+rate|holding\s+rate|carrying\s+rate|holding\s+percentage|carrying\s+percentage|annual\s+holding\s+cost|holding\s+cost|carrying\s+cost)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|annual\s+holding\s+rate|of\s+purchase\s+price|of\s+unit\s+cost|of\s+unit\s+price|of\s+cost|of\s+price|holding\s+cost|carrying\s+cost)',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:annual\s+holding|carrying\s+rate)',
        ], text)
        if holding_rate is not None:
            inputs["holding_rate"] = holding_rate

        # 4. Holding Cost H (Fixed amount - only if holding_rate is not provided)
        if "holding_rate" not in inputs:
            monthly_h = cls.extract_first_match([
                r'(?:monthly holding cost|holding cost per month|carrying cost per month)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per unit per month|per unit/month|per month|/month)\s*(?:holding|carrying)?',
                r'(?:holding cost|carrying cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per unit per month|per unit/month|per month|/month)',
            ], text)
            weekly_h = cls.extract_first_match([
                r'(?:weekly holding cost|holding cost per week|carrying cost per week)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per unit per week|per unit/week|per week|/week)\s*(?:holding|carrying)?',
            ], text)
            annual_h = cls.extract_first_match([
                r'(?:annual holding cost|holding cost per year|carrying cost per year)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'(?:fixed holding cost|holding cost per unit|carrying cost per unit)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'(?:inventory holding cost|holding cost|carrying cost|storage cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per unit per year|per unit/year|per unit annually|holding cost|carrying cost)',
            ], text)

            if monthly_h is not None:
                inputs["original_holding_cost"] = monthly_h
                inputs["holding_cost_unit"] = "month"
                inputs["holding_cost"] = monthly_h * 12.0
            elif weekly_h is not None:
                inputs["original_holding_cost"] = weekly_h
                inputs["holding_cost_unit"] = "week"
                inputs["holding_cost"] = weekly_h * 52.0
            elif annual_h is not None:
                inputs["holding_cost"] = annual_h
                inputs["holding_cost_unit"] = "year"

        # 5. Unit purchase cost C
        unit_cost = cls.extract_first_match([
            r'(?:unit purchase price|unit purchase cost|unit cost|purchase price|cost per unit|unit price)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per unit|each)(?!\s*(?:per year|/year| annually|per month|/month))',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:purchase cost)',
        ], text)
        if unit_cost is not None:
            inputs["unit_cost"] = unit_cost

        # 6. Lead time (Convert weeks to days if in weeks)
        lt_wks = cls.extract_first_match([
            r'(?:lead time|delivery time|supplier takes)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*weeks?',
            r'([0-9]+(?:\.[0-9]+)?)\s*weeks?\s*(?:lead time|delivery time)',
        ], text)
        lt_days = cls.extract_first_match([
            r'(?:lead time|delivery time|supplier takes)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*days?',
            r'([0-9]+(?:\.[0-9]+)?)\s*days?\s*(?:lead time|delivery time)',
        ], text)

        if lt_wks is not None:
            inputs["lead_time_days"] = lt_wks * 7.0
            inputs["lead_time_unit"] = "week"
            inputs["original_lead_time"] = lt_wks
        elif lt_days is not None:
            inputs["lead_time_days"] = lt_days
            inputs["lead_time_unit"] = "day"

        return inputs

    # ------------------------------------------------------------------------
    # 2. SAFETY STOCK & ROP EXTRACTION
    # ------------------------------------------------------------------------
    @classmethod
    def extract_rop(cls, text: str) -> Dict[str, Any]:
        """
        Carefully distinguishes:
          - average demand (d)
          - demand standard deviation (sigma_d)
          - average lead time (L)
          - lead-time standard deviation (sigma_L)
          - service level (CSL)
        Parses lead time (L) and lead-time standard deviation (sigma_L) expressed
        in weeks, months, or days, and converts them to match daily demand (1 week = 7 days, 1 month = 30 days).
        """
        inputs: Dict[str, Any] = {}

        # 1. Lead time (L) and Lead-time standard deviation (sigma_L) combined or separate
        lt_val: Optional[float] = None
        lt_unit = "day"
        lt_std_val: Optional[float] = None
        lt_std_unit: Optional[str] = None

        # Check combined pattern first: e.g. "Average lead time is 3 weeks with a standard deviation of 0.5 weeks"
        comb_match = re.search(
            r'(?:average\s+lead\s+time|mean\s+lead\s+time|supplier\s+lead\s+time|lead\s+time)'
            r'(?:\s*(?:is|averages|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*(weeks?|months?|days?|wks?|mo|d)?'
            r'.*?(?:standard\s+deviation|std\s*dev|sigma|sigma_l|sigma\s*l)'
            r'(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*(weeks?|months?|days?|wks?|mo|d)?',
            text,
            re.IGNORECASE,
        )
        if comb_match:
            lt_val = float(comb_match.group(1))
            lt_unit = comb_match.group(2) or "day"
            lt_std_val = float(comb_match.group(3))
            lt_std_unit = comb_match.group(4) or lt_unit
        else:
            # Standalone lead-time standard deviation
            std_match = re.search(
                r'(?:lead\s*-?\s*time\s+(?:standard\s+deviation|std\s*dev|std|sigma)|standard\s+deviation\s+of\s+lead\s*-?\s*time|sigma_l|sigma\s+l)'
                r'(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*(weeks?|months?|days?|wks?|mo|d)?',
                text,
                re.IGNORECASE,
            )
            if std_match:
                lt_std_val = float(std_match.group(1))
                lt_std_unit = std_match.group(2)

            # Standalone lead time
            lt_match = re.search(
                r'(?:average\s+lead\s+time|mean\s+lead\s+time|supplier\s+lead\s+time|lead\s+time)'
                r'(?:\s*(?:is|averages|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*(weeks?|months?|days?|wks?|mo|d)?',
                text,
                re.IGNORECASE,
            )
            if lt_match:
                lt_val = float(lt_match.group(1))
                if lt_match.group(2):
                    lt_unit = lt_match.group(2)

        # Normalize lead time units and convert to days (1 week = 7 days, 1 month = 30 days)
        if lt_val is not None:
            u_clean = lt_unit.lower()
            inputs["original_lead_time"] = lt_val
            if u_clean in ["week", "weeks", "wk", "wks", "weekly"]:
                inputs["lead_time_unit"] = "week"
                l_days = lt_val * 7.0
            elif u_clean in ["month", "months", "mo", "monthly"]:
                inputs["lead_time_unit"] = "month"
                l_days = lt_val * 30.0
            else:
                inputs["lead_time_unit"] = "day"
                l_days = lt_val

            inputs["lead_time"] = l_days
            inputs["lead_time_mean"] = l_days
            inputs["lead_time_days_mean"] = l_days
            inputs["lead_time_days"] = l_days

        if lt_std_val is not None:
            su_clean = (lt_std_unit or lt_unit).lower()
            inputs["original_sigma_L"] = lt_std_val
            if su_clean in ["week", "weeks", "wk", "wks", "weekly"]:
                s_l_days = lt_std_val * 7.0
            elif su_clean in ["month", "months", "mo", "monthly"]:
                s_l_days = lt_std_val * 30.0
            else:
                s_l_days = lt_std_val

            inputs["sigma_L"] = s_l_days
            inputs["lead_time_days_std"] = s_l_days

        # 2. Demand standard deviation (sigma_d)
        demand_std = cls.extract_first_match([
            r'(?:demand\s+standard\s+deviation|standard\s+deviation\s+of\s+demand|demand\s+std\s*dev|demand\s+std|daily\s+demand\s+std|sigma\s*d|sigma_d)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'demand(?:\s+averages|\s+is)?\s+[0-9]+(?:\.[0-9]+)?(?:\s+units)?\s+with\s+(?:a\s+)?standard\s+deviation\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)',
            r'demand.*?(?:standard\s+deviation|std\s*dev|sigma)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'with\s+(?:a\s+)?standard\s+deviation\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)(?!\s*(?:days?|weeks?|months?))',
        ], text)
        if demand_std is not None:
            inputs["sigma_d"] = demand_std
            inputs["daily_demand_std"] = demand_std

        # 3. Average daily demand (d)
        demand = cls.extract_first_match([
            r'(?:average\s+daily\s+demand|mean\s+daily\s+demand|daily\s+demand\s+mean|daily\s+demand\s+averages|daily\s+demand\s+average|average\s+demand|mean\s+demand|demand\s+mean|daily\s+demand)(?:\s*(?:is|averages|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*units?\s*(?:per\s+day|daily|average\s+daily\s+demand)',
            r'demand\s*(?:averages|is|=)\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if demand is not None:
            inputs["daily_demand"] = demand
            inputs["daily_demand_mean"] = demand

        # 4. Service level (CSL)
        csl = cls.extract_percentage([
            r'(?:cycle\s+service\s+level|service\s+level|target\s+service\s+level|csl|in-stock\s+probability|in-stock\s+rate)(?:[^0-9%]*?)\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:cycle\s+service\s+level|service\s+level|csl|in-stock)',
        ], text)
        if csl is not None:
            inputs["service_level"] = csl

        # 5. Order quantity Q (optional, for fill rate)
        q = cls.extract_first_match([
            r'(?:order\s+quantity|batch\s+size|order\s+size|q)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if q is not None:
            inputs["order_quantity"] = q

        return inputs

    # ------------------------------------------------------------------------
    # 3. QUANTITY DISCOUNT EXTRACTION
    # ------------------------------------------------------------------------
    @classmethod
    def extract_quantity_discount(cls, text: str) -> Dict[str, Any]:
        """
        Distinguishes:
          - quantity break / discount tiers
          - unit price per bracket
          - ordering cost
          - fixed holding cost
          - percentage holding cost
          - unit purchase price
        Automatically converts time units (e.g. monthly demand to annual).
        """
        inputs: Dict[str, Any] = {}

        # 1. Demand D (Check monthly/weekly/annual)
        monthly_demand = cls.extract_first_match([
            r'(?:monthly\s+demand|demand\s+per\s+month|demand\s+each\s+month)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+month|/\s*month|monthly|a\s+month|each\s+month|per\s+mo|/\s*mo)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+month|/\s*month|monthly|a\s+month|each\s+month|per\s+mo|/\s*mo)',
        ], text)
        weekly_demand = cls.extract_first_match([
            r'(?:weekly\s+demand|demand\s+per\s+week|demand\s+each\s+week)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+week|/\s*week|weekly|a\s+week|each\s+week|per\s+wk|/\s*wk)',
            r'demands?\s*(?:is|of|:|=)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+week|/\s*week|weekly|a\s+week|each\s+week|per\s+wk|/\s*wk)',
        ], text)
        annual_demand = cls.extract_first_match([
            r'(?:annual\s+demand|yearly\s+demand|demand\s+per\s+year|demand\s+of|demand\s+is|demand:)\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units|items|pieces)?\s*(?:per\s+year|/\s*year|annual\s+demand|annually|yearly|a\s+year)',
            r'demands?\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)

        if monthly_demand is not None:
            inputs["original_demand"] = monthly_demand
            inputs["demand_unit"] = "month"
            inputs["demand"] = monthly_demand * 12.0
            inputs["annual_demand_D"] = monthly_demand * 12.0
        elif weekly_demand is not None:
            inputs["original_demand"] = weekly_demand
            inputs["demand_unit"] = "week"
            inputs["demand"] = weekly_demand * 52.0
            inputs["annual_demand_D"] = weekly_demand * 52.0
        elif annual_demand is not None:
            inputs["demand"] = annual_demand
            inputs["demand_unit"] = "year"
            inputs["annual_demand_D"] = annual_demand

        # 2. Ordering cost S (using full synonym list)
        order_cost = cls.extract_first_match([
            r'(?:ordering/setup\s+cost|ordering\s*/\s*setup\s+cost|ordering\s+and\s+setup\s+cost|cost\s+of\s+placing\s+an\s+order|cost\s+per\s+order|setup\s+cost|setup\s+expense|ordering\s+cost|order\s+cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+order|order\s+cost|setup\s+cost|to\s+place\s+each\s+order|to\s+place\s+an\s+order)',
        ], text)
        if order_cost is not None:
            inputs["order_cost"] = order_cost

        # 3. Percentage holding cost (i) - extract first so 20% is not misparsed as $2.00
        holding_rate = cls.extract_percentage([
            r'(?:percentage\s+holding\s+cost|holding\s+cost\s+rate|annual\s+holding\s+cost\s+rate|annual\s+holding\s+rate|carrying\s+cost\s+rate|holding\s+rate|carrying\s+rate|holding\s+percentage|carrying\s+percentage|annual\s+holding\s+cost|holding\s+cost|carrying\s+cost)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:holding|carrying|annual\s+holding\s+rate|of\s+purchase\s+price|of\s+unit\s+cost|of\s+unit\s+price|of\s+cost|of\s+price|holding\s+cost|carrying\s+cost)',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:annual\s+holding|carrying\s+rate)',
        ], text)
        if holding_rate is not None:
            inputs["holding_rate"] = holding_rate

        # 4. Fixed holding cost (H) - only if holding_rate is not specified
        if "holding_rate" not in inputs:
            monthly_h = cls.extract_first_match([
                r'(?:monthly\s+holding\s+cost|holding\s+cost\s+per\s+month)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:per\s+unit\s+per\s+month|per\s+unit/month|per\s+month|/month)\s*(?:holding|carrying)?',
            ], text)
            annual_h = cls.extract_first_match([
                r'(?:fixed\s+holding\s+cost|fixed\s+carrying\s+cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
                r'holding\s+cost(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)(?!\s*%)',
            ], text)

            if monthly_h is not None:
                inputs["holding_cost"] = monthly_h * 12.0
                inputs["holding_cost_unit"] = "month"
                inputs["original_holding_cost"] = monthly_h
            elif annual_h is not None:
                inputs["holding_cost"] = annual_h
                inputs["holding_cost_unit"] = "year"

        # 5. Discount Schedule / Quantity Breaks
        tiers = cls.extract_discount_schedule(text)
        if tiers:
            inputs["discount_schedule"] = tiers
            inputs["discount_schedule_tier_count"] = len(tiers)

        return inputs

    # ------------------------------------------------------------------------
    # 4. FIXED-PERIOD REVIEW EXTRACTION
    # ------------------------------------------------------------------------
    @classmethod
    def extract_periodic_review(cls, text: str) -> Dict[str, Any]:
        """
        Distinguishes:
          - demand (d, sigma_d)
          - review period (T)
          - lead time (L)
          - safety stock (SS)
          - inventory position
          - on-hand inventory
          - on-order inventory
          - backorders
        Automatically converts time units (weeks to days) for protection period (T + L) consistency.
        """
        inputs: Dict[str, Any] = {}

        # 1. Review Period T (Detect weeks vs days)
        t_wks = cls.extract_first_match([
            r'(?:review period|review interval|reviewed every|review every)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*weeks?',
            r'every\s*([0-9]+(?:\.[0-9]+)?)\s*weeks?',
        ], text)
        t_days = cls.extract_first_match([
            r'(?:review period|review interval|reviewed every|review every)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*days?',
            r'every\s*([0-9]+(?:\.[0-9]+)?)\s*days?',
        ], text)

        if t_wks is not None:
            inputs["review_period_unit"] = "week"
            inputs["original_review_period"] = t_wks
            inputs["review_period"] = t_wks * 7.0
            inputs["review_period_days"] = t_wks * 7.0
        elif t_days is not None:
            inputs["review_period_unit"] = "day"
            inputs["review_period"] = t_days
            inputs["review_period_days"] = t_days
        else:
            if "biweekly" in text.lower() or "two weeks" in text.lower():
                inputs["review_period"] = 14.0
                inputs["review_period_days"] = 14.0
                inputs["review_period_unit"] = "week"
            elif "weekly" in text.lower() or "one week" in text.lower() or "every week" in text.lower():
                inputs["review_period"] = 7.0
                inputs["review_period_days"] = 7.0
                inputs["review_period_unit"] = "week"
            elif "monthly" in text.lower() or "every month" in text.lower():
                inputs["review_period"] = 30.0
                inputs["review_period_days"] = 30.0
                inputs["review_period_unit"] = "month"

        # 2. Lead time L (Detect weeks vs days)
        l_wks = cls.extract_first_match([
            r'(?:lead time|delivery time|supplier lead time)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*weeks?',
            r'([0-9]+(?:\.[0-9]+)?)\s*weeks?\s*(?:lead time|delivery time)',
        ], text)
        l_days = cls.extract_first_match([
            r'(?:lead time|delivery time|supplier lead time)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)\s*days?',
            r'([0-9]+(?:\.[0-9]+)?)\s*days?\s*(?:lead time|delivery time)',
        ], text)

        if l_wks is not None:
            inputs["lead_time_unit"] = "week"
            inputs["original_lead_time"] = l_wks
            inputs["lead_time"] = l_wks * 7.0
            inputs["lead_time_days"] = l_wks * 7.0
        elif l_days is not None:
            inputs["lead_time_unit"] = "day"
            inputs["lead_time"] = l_days
            inputs["lead_time_days"] = l_days

        # 3. Average daily demand d
        demand_mean = cls.extract_first_match([
            r'(?:average daily demand|mean daily demand|daily demand mean|daily demand average|daily demand|average demand|mean demand|demand mean)(?:\s*(?:is|averages|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*units?\s*(?:per day|daily|average daily demand)',
            r'demand\s*(?:averages|is|=)\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if demand_mean is not None:
            inputs["demand_mean"] = demand_mean
            inputs["daily_demand_mean"] = demand_mean

        # 4. Demand standard deviation sigma_d
        demand_std = cls.extract_first_match([
            r'(?:standard deviation of demand|demand standard deviation|demand std dev|std dev of demand|daily std dev|sigma d)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'with\s+(?:a\s+)?(?:standard deviation|std dev|sigma)\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)',
            r'(?:standard deviation|std dev|sigma)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if demand_std is not None:
            inputs["demand_std"] = demand_std
            inputs["daily_demand_std"] = demand_std

        # 5. Service level
        csl = cls.extract_percentage([
            r'(?:service level|cycle service level|csl|target service level)(?:[^0-9%]*?)\s*([0-9]+(?:\.[0-9]+)?)\s*%',
            r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:service level|cycle service level|csl)',
        ], text)
        if csl is not None:
            inputs["service_level"] = csl

        # 6. Explicit Inventory Position
        inv_pos = cls.extract_first_match([
            r'(?:inventory position|net inventory position)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if inv_pos is not None:
            inputs["inventory_position"] = inv_pos

        # 7. On-hand inventory
        on_hand = cls.extract_first_match([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:on hand|in stock|on-hand)',
            r'(?:on-hand|on hand|in stock|current stock)(?:\s+inventory)?(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if on_hand is not None:
            inputs["on_hand"] = on_hand
            inputs["on_hand_inventory"] = on_hand

        # 8. On-order inventory
        on_order = cls.extract_first_match([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:on order|on-order|in pipeline|pipeline|in transit)',
            r'(?:on-order|on order|in transit|pipeline|in pipeline)(?:\s+inventory)?(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if on_order is not None:
            inputs["on_order"] = on_order
            inputs["on_order_inventory"] = on_order

        # 9. Backorders
        backorders = cls.extract_first_match([
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:units?)?\s*(?:backordered|back-ordered|backorders)',
            r'(?:backorders|backordered|backlog)(?:\s*(?:are|is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if backorders is not None:
            inputs["backorders"] = backorders

        # Calculate composite inventory position if on-hand / on-order / backorders are given
        if "inventory_position" not in inputs and ("on_hand" in inputs or "on_order" in inputs):
            oh = inputs.get("on_hand", 0.0)
            oo = inputs.get("on_order", 0.0)
            bo = inputs.get("backorders", 0.0)
            inputs["inventory_position"] = oh + oo - bo

        return inputs

    # ------------------------------------------------------------------------
    # 5. NEWSVENDOR EXTRACTION
    # ------------------------------------------------------------------------
    @classmethod
    def extract_newsvendor(cls, text: str) -> Dict[str, Any]:
        """
        Distinguishes:
          - selling price (p)
          - purchase cost (c)
          - salvage value (v / s)
          - underage / shortage cost (Cu = p - c)
          - overage / excess cost (Co = c - v)
          - demand distribution: mean demand (mu), demand std (sigma)
        """
        inputs: Dict[str, Any] = {}

        # 1. Selling price p
        selling_price = cls.extract_first_match([
            r'(?:selling price|retail price|retail value|revenue per unit)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'sells?\s+(?:.*?\s+)?for\s+\$?([0-9]+(?:\.[0-9]+)?)',
            r'price(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:selling price|retail price|retail)',
        ], text)
        if selling_price is not None:
            inputs["selling_price"] = selling_price
            inputs["p"] = selling_price

        # 2. Purchase cost c
        unit_cost = cls.extract_first_match([
            r'(?:purchase cost|wholesale cost|unit cost|cost per unit|wholesale price|procurement cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'costs?\s+(?:to purchase\s+|to make\s+|to acquire\s+)?\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:unit cost|purchase cost|wholesale cost|to buy|to purchase)',
        ], text)
        if unit_cost is not None:
            inputs["unit_cost"] = unit_cost
            inputs["c"] = unit_cost

        # 3. Salvage value v / s
        salvage_value = cls.extract_first_match([
            r'(?:salvage value|salvage price|liquidated for|liquidate for|scrap value|markdown price|residual value|salvage\s*v|v\s*=)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
            r'\$?([0-9]+(?:\.[0-9]+)?)\s*(?:salvage|salvage value|scrap value|salvage price)',
        ], text)
        if salvage_value is not None:
            inputs["salvage_value"] = salvage_value
            inputs["v"] = salvage_value

        # 4. Underage / Shortage cost Cu (or goodwill penalty g)
        cu = cls.extract_first_match([
            r'(?:underage cost|cost of underage|shortage cost|cost of shortage|unmet demand penalty)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if cu is not None:
            inputs["underage_cost"] = cu
            inputs["Cu"] = cu

        goodwill = cls.extract_first_match([
            r'(?:goodwill penalty|goodwill loss|loss of goodwill|shortage penalty|penalty cost)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if goodwill is not None:
            inputs["goodwill_penalty"] = goodwill
            inputs["g"] = goodwill

        # 5. Overage / Excess cost Co
        co = cls.extract_first_match([
            r'(?:overage cost|cost of overage|excess cost|cost of excess inventory)(?:\s*(?:is|of|:|=))?\s*\$?([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if co is not None:
            inputs["overage_cost"] = co
            inputs["Co"] = co

        # If Cu and Co are directly provided, synthesize consistent p and c
        if "underage_cost" in inputs and "overage_cost" in inputs:
            if "selling_price" not in inputs and "unit_cost" not in inputs:
                # Set c = Co, p = Cu + c, s = 0 so that:
                # Cu = p - c = (Cu + Co) - Co = Cu
                # Co = c - s = Co - 0 = Co
                inputs["unit_cost"] = inputs["overage_cost"]
                inputs["c"] = inputs["unit_cost"]
                inputs["selling_price"] = inputs["underage_cost"] + inputs["unit_cost"]
                inputs["p"] = inputs["selling_price"]
                inputs["salvage_value"] = 0.0
                inputs["v"] = 0.0

        # 6. Mean demand mu
        demand_mean = cls.extract_first_match([
            r'(?:mean demand|average demand|expected demand|demand mean|mean of|demand averages?)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'mean\s*(?:of|=|:)?\s*([0-9]+(?:\.[0-9]+)?)',
            r'([0-9]+(?:\.[0-9]+)?)\s*(?:mean demand|average units)',
        ], text)
        if demand_mean is not None:
            inputs["demand_mean"] = demand_mean
            inputs["mu"] = demand_mean

        # 7. Demand standard deviation sigma
        demand_std = cls.extract_first_match([
            r'(?:standard deviation of demand|demand standard deviation|demand std dev|std dev of demand|sigma)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
            r'with\s+(?:a\s+)?(?:standard deviation|std dev)\s+(?:of\s+)?([0-9]+(?:\.[0-9]+)?)',
            r'(?:standard deviation|std dev)(?:\s*(?:is|of|:|=))?\s*([0-9]+(?:\.[0-9]+)?)',
        ], text)
        if demand_std is not None:
            inputs["demand_std"] = demand_std
            inputs["sigma"] = demand_std

        return inputs


# ============================================================================
# INVENTORY DECISION AGENT
# ============================================================================

class InventoryDecisionAgent:
    """
    Supply Chain Analytics Inventory Decision Agent.
    Executes the 7-stage decision loop over any natural-language or structured question.
    """

    MODEL_EOQ = "EOQ"
    MODEL_QUANTITY_DISCOUNT = "QUANTITY_DISCOUNT"
    MODEL_SAFETY_STOCK_ROP = "SAFETY_STOCK_ROP"
    MODEL_FIXED_PERIOD_REVIEW = "FIXED_PERIOD_REVIEW"
    MODEL_NEWSVENDOR = "NEWSVENDOR"

    MODEL_NAMES = {
        MODEL_EOQ: "Economic Order Quantity (EOQ)",
        MODEL_QUANTITY_DISCOUNT: "Quantity Discount (All-Units)",
        MODEL_SAFETY_STOCK_ROP: "Safety Stock & Reorder Point (Continuous Review)",
        MODEL_FIXED_PERIOD_REVIEW: "Fixed-Period Review (Periodic Review / Order-Up-To)",
        MODEL_NEWSVENDOR: "Newsvendor (Single-Period Stochastic)",
    }

    def __init__(self):
        self.extractor = InventoryInputExtractor()

    # ------------------------------------------------------------------------
    # STAGE 1: IDENTIFY MODEL
    # ------------------------------------------------------------------------
    def identify_model(self, question: str) -> str:
        """
        Classifies the customer query into one of the 5 inventory models.
        Uses priority keyword scoring and structural pattern matching.
        """
        q_lower = question.lower()

        scores = {
            self.MODEL_EOQ: 0,
            self.MODEL_QUANTITY_DISCOUNT: 0,
            self.MODEL_SAFETY_STOCK_ROP: 0,
            self.MODEL_FIXED_PERIOD_REVIEW: 0,
            self.MODEL_NEWSVENDOR: 0,
        }

        # 1. Newsvendor indicators
        if any(k in q_lower for k in ["newsvendor", "newsboy", "single period", "single-period", "perishable"]):
            scores[self.MODEL_NEWSVENDOR] += 15
        if any(k in q_lower for k in ["salvage", "liquidate", "scrap value", "unsold", "spoilage", "critical fractile", "critical ratio", "underage", "overage"]):
            scores[self.MODEL_NEWSVENDOR] += 8
        if "seasonal" in q_lower and any(w in q_lower for w in ["demand", "order", "coat", "jacket", "item"]):
            scores[self.MODEL_NEWSVENDOR] += 6

        # 2. Quantity Discount indicators
        if any(k in q_lower for k in [
            "quantity discount", "discount schedule", "price break", "price bracket",
            "all-units", "all units discount", "volume discount", "tiered", "price tier"
        ]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 25
        if "tier" in q_lower or ("discount" in q_lower and any(w in q_lower for w in ["price", "cost", "bracket", "schedule"])):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 15
        if any(w in q_lower for w in ["below", "under", "or more", "and above", "and over"]) and ("$" in q_lower or "charges" in q_lower or "price" in q_lower or "per unit" in q_lower):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 20
        if "charges" in q_lower and ("$" in q_lower or "per unit" in q_lower):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 15
        if "-" in q_lower and ("$" in q_lower or "at" in q_lower) and any(w in q_lower for w in ["order", "bracket", "tier", "price"]):
            scores[self.MODEL_QUANTITY_DISCOUNT] += 15
        try:
            tiers_found = self.extractor.extract_discount_schedule(self.extractor.clean_text(question))
            if len(tiers_found) >= 2:
                scores[self.MODEL_QUANTITY_DISCOUNT] += 35
        except Exception:
            pass

        # 3. Fixed-Period Review indicators
        if any(k in q_lower for k in ["fixed period", "fixed-period", "periodic review", "order-up-to", "order up to", "target level", "review interval", "(t, s)", "t,s", "inventory position"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 15
        if any(k in q_lower for k in ["review every", "reviewed every", "review period", "every 7 days", "every 14 days", "every week", "every 2 weeks", "every two weeks", "every month", "weekly review", "biweekly review", "review inventory"]):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 9
        if "on hand" in q_lower and ("on order" in q_lower or "backorder" in q_lower):
            scores[self.MODEL_FIXED_PERIOD_REVIEW] += 7

        # 4. Safety Stock & Reorder Point (Continuous Review)
        if any(k in q_lower for k in ["safety stock", "reorder point", "rop", "(s, q)", "s,q", "continuous review", "lead time demand"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 15
        if any(k in q_lower for k in ["service level", "cycle service level", "fill rate", "stockout probability"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 6
        if any(k in q_lower for k in ["std dev", "standard deviation", "sigma"]):
            scores[self.MODEL_SAFETY_STOCK_ROP] += 2

        # 5. EOQ indicators
        if any(k in q_lower for k in ["eoq", "economic order quantity", "optimal batch size", "optimal order quantity", "order frequency"]):
            scores[self.MODEL_EOQ] += 15
        if any(k in q_lower for k in ["ordering cost", "setup cost", "cost per order", "setup expense", "cost of placing an order", "ordering/setup cost"]):
            scores[self.MODEL_EOQ] += 5
        if any(k in q_lower for k in ["holding cost", "carrying cost", "annual demand"]):
            scores[self.MODEL_EOQ] += 3

        # Disambiguation heuristics
        # Route any prompt containing price breaks, volume discounts, or tiered unit prices
        # (such as "charges $X per unit for orders below Y, $Z for ..." or "or more") to the Quantity Discount model - NEVER basic EOQ!
        has_multiple_tiers = False
        try:
            has_multiple_tiers = len(self.extractor.extract_discount_schedule(self.extractor.clean_text(question))) >= 2
        except Exception:
            pass

        qd_terms = [
            "charges", "price break", "volume discount", "tiered", "price tier", "quantity discount",
            "discount schedule", "price bracket", "all-units", "all units",
            "orders below", "orders under", "below 500", "under 500", "<500", "< 500",
            "or more", "and above", "and over", ">=1000", ">= 1000",
        ]
        has_qd_terms = any(term in q_lower for term in qd_terms) and ("$" in q_lower or "%" in q_lower or "tier" in q_lower or "schedule" in q_lower)

        if has_multiple_tiers or (has_qd_terms and scores[self.MODEL_QUANTITY_DISCOUNT] > 0):
            return self.MODEL_QUANTITY_DISCOUNT

        if scores[self.MODEL_FIXED_PERIOD_REVIEW] >= 8:
            return self.MODEL_FIXED_PERIOD_REVIEW

        if scores[self.MODEL_QUANTITY_DISCOUNT] >= 7:
            return self.MODEL_QUANTITY_DISCOUNT

        if scores[self.MODEL_NEWSVENDOR] >= 7:
            return self.MODEL_NEWSVENDOR

        best_model = max(scores, key=lambda k: scores[k])
        if scores[best_model] > 0:
            return best_model

        return self.MODEL_EOQ

    # ------------------------------------------------------------------------
    # STAGE 2: EXTRACT INPUTS
    # ------------------------------------------------------------------------
    def extract_inputs(self, model: str, question: str, explicit_inputs: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Examines the entire question using specialized extractors before declaring information missing.
        """
        clean_q = self.extractor.clean_text(question)
        inputs: Dict[str, Any] = {}
        missing: List[str] = []

        if model == self.MODEL_EOQ:
            extracted = self.extractor.extract_eoq(clean_q)
            inputs.update(extracted)
            if explicit_inputs:
                inputs.update(explicit_inputs)

            if inputs.get("demand") is None:
                missing.append("annual demand (D)")
            if inputs.get("order_cost") is None:
                missing.append("ordering cost (S) [e.g. setup cost, setup expense, cost of placing an order]")
            if inputs.get("holding_cost") is None and inputs.get("holding_rate") is None:
                missing.append("holding cost (H) or holding rate (i)")

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            extracted = self.extractor.extract_quantity_discount(clean_q)
            inputs.update(extracted)
            if explicit_inputs:
                inputs.update(explicit_inputs)

            if inputs.get("demand") is None:
                missing.append("annual demand (D)")
            if inputs.get("order_cost") is None:
                missing.append("ordering cost (S)")
            if inputs.get("holding_rate") is None and inputs.get("holding_cost") is None:
                missing.append("holding rate (i) or fixed holding cost (H)")
            if not inputs.get("discount_schedule"):
                missing.append("discount schedule / quantity breaks")

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            extracted = self.extractor.extract_rop(clean_q)
            inputs.update(extracted)
            if explicit_inputs:
                inputs.update(explicit_inputs)

            if inputs.get("daily_demand") is None and inputs.get("daily_demand_mean") is None:
                missing.append("average daily demand (d)")
            if inputs.get("lead_time") is None and inputs.get("lead_time_days_mean") is None:
                missing.append("average lead time in days (L)")

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            extracted = self.extractor.extract_periodic_review(clean_q)
            inputs.update(extracted)
            if explicit_inputs:
                inputs.update(explicit_inputs)

            if inputs.get("review_period") is None and inputs.get("review_period_days") is None:
                missing.append("review period (T)")
            if inputs.get("lead_time") is None and inputs.get("lead_time_days") is None:
                missing.append("lead time (L)")
            if inputs.get("demand_mean") is None and inputs.get("daily_demand_mean") is None:
                missing.append("average demand (d)")

        elif model == self.MODEL_NEWSVENDOR:
            extracted = self.extractor.extract_newsvendor(clean_q)
            inputs.update(extracted)
            if explicit_inputs:
                inputs.update(explicit_inputs)

            if inputs.get("selling_price") is None:
                missing.append("selling price per unit (p)")
            if inputs.get("unit_cost") is None:
                missing.append("unit purchase cost (c)")
            if inputs.get("demand_mean") is None:
                missing.append("mean expected demand (mu)")
            if inputs.get("demand_std") is None:
                missing.append("demand standard deviation (sigma)")

        return inputs, missing

    # ------------------------------------------------------------------------
    # STAGE 3, 4, 5: SELECT TOOL, CALCULATE, & VERIFY
    # ------------------------------------------------------------------------
    def execute_calculation(self, model: str, inputs: Dict[str, Any]) -> Tuple[Any, bool, Dict[str, Any]]:
        """
        Invokes the deterministic tools in `inventory_tools.py`.
        """
        if model == self.MODEL_EOQ:
            demand_val = float(inputs.get("original_demand", inputs["demand"])) if inputs.get("demand_unit") and inputs.get("demand_unit") != "year" and "original_demand" in inputs else float(inputs["demand"])
            holding_cost_val = float(inputs["original_holding_cost"]) if inputs.get("holding_cost_unit") and inputs.get("holding_cost_unit") != "year" and "original_holding_cost" in inputs else (float(inputs["holding_cost"]) if inputs.get("holding_cost") is not None else None)
            lead_time_val = float(inputs["original_lead_time"]) if inputs.get("lead_time_unit") and inputs.get("lead_time_unit") != "day" and "original_lead_time" in inputs else float(inputs.get("lead_time_days", 0.0))

            res = it.calculate_eoq(
                demand=demand_val,
                order_cost=float(inputs["order_cost"]),
                holding_cost=holding_cost_val,
                holding_rate=float(inputs["holding_rate"]) if inputs.get("holding_rate") is not None else None,
                unit_cost=float(inputs.get("unit_cost", 0.0)),
                work_days=int(inputs.get("work_days", 250)),
                lead_time_days=lead_time_val,
                demand_unit=str(inputs.get("demand_unit", "year")),
                holding_cost_unit=str(inputs.get("holding_cost_unit", "year")),
                lead_time_unit=str(inputs.get("lead_time_unit", "day")),
                days_per_week=float(inputs.get("days_per_week", 7.0)),
            )
            v_passed = res.verification.get("holding_equals_ordering_at_eoq", False)
            return res, v_passed, res.verification

        elif model == self.MODEL_QUANTITY_DISCOUNT:
            demand_val = float(inputs.get("original_demand", inputs["demand"])) if inputs.get("demand_unit") and inputs.get("demand_unit") != "year" and "original_demand" in inputs else float(inputs["demand"])
            holding_cost_val = float(inputs["original_holding_cost"]) if inputs.get("holding_cost_unit") and inputs.get("holding_cost_unit") != "year" and "original_holding_cost" in inputs else (float(inputs["holding_cost"]) if inputs.get("holding_cost") is not None else None)

            res = it.calculate_quantity_discount(
                demand=demand_val,
                order_cost=float(inputs["order_cost"]),
                holding_rate=float(inputs["holding_rate"]) if inputs.get("holding_rate") is not None else None,
                holding_cost=holding_cost_val,
                discount_schedule=inputs["discount_schedule"],
                demand_unit=str(inputs.get("demand_unit", "year")),
                holding_cost_unit=str(inputs.get("holding_cost_unit", "year")),
            )
            v_passed = res.verification.get("is_global_minimum", False)
            return res, v_passed, res.verification

        elif model == self.MODEL_SAFETY_STOCK_ROP:
            d_val = float(inputs.get("original_daily_demand", inputs.get("daily_demand") or inputs.get("daily_demand_mean")))  # type: ignore
            L_val = float(inputs.get("original_lead_time", inputs.get("lead_time") or inputs.get("lead_time_days_mean") or inputs.get("lead_time_days")))  # type: ignore
            s_d = float(inputs.get("sigma_d") or inputs.get("daily_demand_std") or 0.0)
            s_L = float(inputs.get("original_sigma_L", inputs.get("sigma_L") or inputs.get("lead_time_days_std") or 0.0))
            csl = float(inputs.get("service_level", 0.95))
            q = float(inputs["order_quantity"]) if inputs.get("order_quantity") else None

            res = it.calculate_rop(
                daily_demand=d_val,
                lead_time=L_val,
                sigma_d=s_d,
                sigma_L=s_L,
                service_level=csl,
                order_quantity=q,
                demand_time_unit=str(inputs.get("demand_time_unit", "day")),
                lead_time_unit=str(inputs.get("lead_time_unit", "day")),
                days_per_week=float(inputs.get("days_per_week", 7.0)),
            )
            v_passed = res.verification.get("rop_greater_than_or_equal_dltd", False) and res.verification.get("ss_non_negative", False)
            return res, v_passed, res.verification

        elif model == self.MODEL_FIXED_PERIOD_REVIEW:
            T_val = float(inputs.get("original_review_period", inputs.get("review_period") or inputs.get("review_period_days")))  # type: ignore
            L_val = float(inputs.get("original_lead_time", inputs.get("lead_time") or inputs.get("lead_time_days")))  # type: ignore
            d = float(inputs.get("demand_mean") or inputs.get("daily_demand_mean"))  # type: ignore
            s_d = float(inputs.get("demand_std") or inputs.get("daily_demand_std") or 0.0)
            csl = float(inputs.get("service_level", 0.95))
            inv_pos = float(inputs["inventory_position"]) if inputs.get("inventory_position") is not None else None
            oh = float(inputs.get("on_hand", 0.0))
            oo = float(inputs.get("on_order", 0.0))

            res = it.calculate_periodic_review(
                review_period=T_val,
                lead_time=L_val,
                demand_mean=d,
                demand_std=s_d,
                service_level=csl,
                inventory_position=inv_pos,
                on_hand=oh,
                on_order=oo,
                review_period_unit=str(inputs.get("review_period_unit", "day")),
                lead_time_unit=str(inputs.get("lead_time_unit", "day")),
                demand_time_unit=str(inputs.get("demand_time_unit", "day")),
                days_per_week=float(inputs.get("days_per_week", 7.0)),
            )
            v_passed = res.verification.get("target_inventory_covers_expected_demand", False) and res.verification.get("order_quantity_non_negative", False)
            return res, v_passed, res.verification

        elif model == self.MODEL_NEWSVENDOR:
            res = it.calculate_newsvendor(
                selling_price=float(inputs.get("selling_price") if inputs.get("selling_price") is not None else inputs["p"]),
                unit_cost=float(inputs.get("unit_cost") if inputs.get("unit_cost") is not None else inputs["c"]),
                salvage_value=float(inputs.get("salvage_value") if inputs.get("salvage_value") is not None else (inputs.get("v") or 0.0)),
                goodwill_penalty=float(inputs.get("goodwill_penalty") if inputs.get("goodwill_penalty") is not None else (inputs.get("g") or 0.0)),
                demand_mean=float(inputs.get("demand_mean") if inputs.get("demand_mean") is not None else inputs["mu"]),
                demand_std=float(inputs.get("demand_std") if inputs.get("demand_std") is not None else inputs["sigma"]),
            )
            v_passed = res.verification.get("critical_ratio_valid_bounds", False) and res.verification.get("evpi_non_negative", False)
            return res, v_passed, res.verification

        raise ValueError(f"Unknown model identifier: {model}")

    # ------------------------------------------------------------------------
    # STAGE 6: EXPLAIN (8 Standard Student Verification Components)
    # ------------------------------------------------------------------------
    def generate_explanation(self, model: str, raw_res: Any, inputs: Dict[str, Any], verification_details: Dict[str, Any]) -> str:
        """
        Formats the 8 verification components directly produced by the deterministic Python tool:
          1. Inputs Extracted
          2. Variable Definitions
          3. Formula
          4. Formula with Substituted Values
          5. Intermediate Calculations
          6. Final Answer
          7. Verification Checks
          8. Assumptions

        All numerical steps originate strictly from the Python tool (never reconstructed in LLM).
        """
        lines = []
        model_header = getattr(raw_res, "model", model)
        case_header = f" - {getattr(raw_res, 'case_name')}" if hasattr(raw_res, "case_name") and getattr(raw_res, "case_name") else ""
        lines.append(f"## 📦 Supply Chain Analytics Decision: {model_header}{case_header}\n")

        # 1. Inputs Extracted
        lines.append("### 1. Inputs Extracted")
        extracted_dict = getattr(raw_res, "inputs_extracted", inputs)
        for k, v in extracted_dict.items():
            lines.append(f"- **`{k}`**: `{v}`")
        lines.append("")

        # 2. Variable Definitions
        lines.append("### 2. Variable Definitions")
        var_defs = getattr(raw_res, "variable_definitions", {})
        for sym, desc in var_defs.items():
            lines.append(f"- **`{sym}`**: {desc}")
        lines.append("")

        # 3. Formula
        lines.append("### 3. Formula")
        formulas = getattr(raw_res, "formula", {})
        if isinstance(formulas, dict):
            for fname, fstr in formulas.items():
                lines.append(f"- **{fname}**: `{fstr}`")
        elif isinstance(formulas, list):
            for fstr in formulas:
                lines.append(f"- `{fstr}`")
        else:
            lines.append(f"`{formulas}`")
        lines.append("")

        # 4. Formula with Substituted Values
        lines.append("### 4. Formula with Substituted Values")
        substituted = getattr(raw_res, "formula_with_substituted_values", {})
        if isinstance(substituted, dict):
            for sname, sstr in substituted.items():
                lines.append(f"- **{sname}**: `{sstr}`")
        elif isinstance(substituted, list):
            for sstr in substituted:
                lines.append(f"- `{sstr}`")
        else:
            lines.append(f"`{substituted}`")
        lines.append("")

        # 5. Intermediate Calculations
        lines.append("### 5. Intermediate Calculations")
        intermediates = getattr(raw_res, "intermediate_calculations", {})
        for ik, iv in intermediates.items():
            if isinstance(iv, list):
                lines.append(f"- **`{ik}`**:")
                for item in iv:
                    lines.append(f"  - `{item}`")
            else:
                lines.append(f"- **`{ik}`**: `{iv}`")
        lines.append("")

        # 6. Final Answer
        lines.append("### 6. Final Answer")
        final_ans = getattr(raw_res, "final_answer", {})
        for ak, av in final_ans.items():
            lines.append(f"- **`{ak}`**: `{av}`")
        lines.append("")

        # 7. Verification Checks
        lines.append("### 7. Verification Checks")
        verif = getattr(raw_res, "verification_checks", verification_details)
        for vk, vv in verif.items():
            lines.append(f"- **`{vk}`**: `{vv}`")
        lines.append("")

        # 8. Assumptions
        lines.append("### 8. Assumptions")
        assumptions_list = getattr(raw_res, "assumptions", [])
        for a in assumptions_list:
            lines.append(f"- {a}")

        return "\n".join(lines)

    # ------------------------------------------------------------------------
    # FULL END-TO-END PIPELINE ORCHESTRATOR
    # ------------------------------------------------------------------------
    def solve(self, question: str, explicit_model: Optional[str] = None, explicit_inputs: Optional[Dict[str, Any]] = None) -> AgentDecision:
        """
        Executes the full pipeline:
          Customer Question → Identify Model → Extract Inputs → Select Python Tool → Calculate → Verify → Explain
        """
        # Step 1: Identify Model
        model = explicit_model or self.identify_model(question)
        model_name = self.MODEL_NAMES.get(model, model)

        # Step 2: Extract Inputs
        extracted_inputs, missing = self.extract_inputs(model, question, explicit_inputs)

        # If essential inputs are missing, halt with diagnostic guidance
        if missing:
            err_msg = f"Insufficient parameters extracted for model '{model_name}'. Missing: {', '.join(missing)}."
            explanation = (
                f"### ⚠️ Clarification Needed: {model_name}\n\n"
                f"To calculate the optimal inventory policy, please specify the following required inputs:\n"
                + "\n".join([f"- **{m}**" for m in missing]) + "\n\n"
                f"**Extracted so far:** `{extracted_inputs}`"
            )
            return AgentDecision(
                question=question,
                model_identified=model,
                model_name=model_name,
                extracted_inputs=extracted_inputs,
                missing_inputs=missing,
                explanation=explanation,
                error=err_msg,
            )

        # Step 3, 4, 5: Select Tool, Calculate, & Verify
        try:
            raw_res, v_passed, v_details = self.execute_calculation(model, extracted_inputs)
        except Exception as e:
            err_msg = f"Calculation error in {model_name}: {str(e)}"
            return AgentDecision(
                question=question,
                model_identified=model,
                model_name=model_name,
                extracted_inputs=extracted_inputs,
                explanation=f"### ❌ Calculation Error\n\n{err_msg}",
                error=err_msg,
            )

        # Step 6: Explain
        explanation = self.generate_explanation(model, raw_res, extracted_inputs, v_details)

        return AgentDecision(
            question=question,
            model_identified=model,
            model_name=model_name,
            extracted_inputs=extracted_inputs,
            missing_inputs=[],
            raw_result=raw_res,
            structured_output=raw_res.to_dict(),
            verification_passed=v_passed,
            verification_details=v_details,
            explanation=explanation,
        )
