"""Safe arithmetic tool. Parses expressions with ast — no unrestricted eval."""

from __future__ import annotations

import ast
import math
import operator

from langchain_core.tools import tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pow": math.pow,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalculatorError(ValueError):
    pass


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise CalculatorError("Division by zero.")
        return float(_BIN_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return float(_UNARY_OPS[type(node.op)](_eval_node(node.operand)))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return float(_CONSTANTS[node.id])
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        name = node.func.id
        if name not in _FUNCTIONS:
            raise CalculatorError(f"Function `{name}` is not allowed.")
        args = [_eval_node(arg) for arg in node.args]
        return float(_FUNCTIONS[name](*args))
    raise CalculatorError("Only numbers and basic math operators are allowed.")


def safe_calculate(expression: str) -> float:
    cleaned = expression.strip().replace("^", "**")
    if not cleaned:
        raise CalculatorError("Expression is empty.")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError("Could not parse the expression.") from exc
    return _eval_node(tree)


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic math expression such as 1234 * 0.15 or sqrt(16) + 2.

    Use this for arithmetic instead of guessing. Allowed: + - * / // % ** and
    functions sqrt, abs, round, log, log10, exp, sin, cos, tan, pow, min, max.
    """
    try:
        result = safe_calculate(expression)
    except CalculatorError as exc:
        return f"Calculator error: {exc}"
    if result.is_integer():
        return str(int(result))
    return f"{result:.10g}"
