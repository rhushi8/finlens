import logging
from datetime import datetime

from finance_ai.schemas.tools import CalculatorResponse

logger = logging.getLogger(__name__)


def calculate_financial_metric(operation: str, params: dict) -> CalculatorResponse:
    try:
        if operation == "pct_change":
            old_value = float(params.get("old_value", 0))
            new_value = float(params.get("new_value", 0))

            if old_value == 0:
                return CalculatorResponse(
                    operation=operation,
                    result=0,
                    formatted_result="N/A",
                    retrieved_at=datetime.now(),
                    error="Old value is zero",
                )

            pct = ((new_value - old_value) / abs(old_value)) * 100
            return CalculatorResponse(
                operation=operation,
                result=round(pct, 2),
                formatted_result=f"{pct:+.2f}%",
                details={"old_value": old_value, "new_value": new_value},
                retrieved_at=datetime.now(),
            )

        elif operation == "pe_multiple":
            earnings_per_share = float(params.get("earnings_per_share", 0))
            target_pe = float(params.get("target_pe", 0))

            if earnings_per_share == 0:
                return CalculatorResponse(
                    operation=operation,
                    result=0,
                    formatted_result="N/A",
                    retrieved_at=datetime.now(),
                    error="EPS is zero",
                )

            target_price = earnings_per_share * target_pe
            return CalculatorResponse(
                operation=operation,
                result=round(target_price, 2),
                formatted_result=f"${target_price:.2f}",
                details={"eps": earnings_per_share, "target_pe": target_pe},
                retrieved_at=datetime.now(),
            )

        elif operation == "dividend_income":
            shares_owned = float(params.get("shares_owned", 0))
            annual_dividend = float(params.get("annual_dividend_per_share", 0))

            income = shares_owned * annual_dividend
            return CalculatorResponse(
                operation=operation,
                result=round(income, 2),
                formatted_result=f"${income:.2f}",
                details={"shares": shares_owned, "dividend_per_share": annual_dividend},
                retrieved_at=datetime.now(),
            )
        elif operation == "price_target":
            current_price = float(params.get("current_price", 0))
            target_price = float(params.get("target_price", 0))

            if current_price == 0:
                return CalculatorResponse(
                    operation=operation,
                    result=0,
                    formatted_result="N/A",
                    retrieved_at=datetime.now(),
                    error="Current price is zero",
                )

            pct_upside = ((target_price - current_price) / abs(current_price)) * 100
            absolute_change = target_price - current_price
            return CalculatorResponse(
                operation=operation,
                result=round(pct_upside, 2),
                formatted_result=f"{pct_upside:+.2f}%",
                details={
                    "current_price": current_price,
                    "target_price": target_price,
                    "absolute_change": round(absolute_change, 2),
                },
                retrieved_at=datetime.now(),
            )

        else:
            return CalculatorResponse(
                operation=operation,
                result=0,
                formatted_result="N/A",
                retrieved_at=datetime.now(),
                error=f"Unknown operation: {operation}",
            )

    except Exception as e:
        logger.error(f"Error in calculator for {operation}: {e}")
        return CalculatorResponse(
            operation=operation,
            result=0,
            formatted_result="N/A",
            retrieved_at=datetime.now(),
            error=str(e),
        )
