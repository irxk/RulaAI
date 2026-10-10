"""Safe arithmetic evaluator — AST-only, no exec/eval."""
import ast
import math
import operator
import re

from .base import Tool

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "int": int,
    "float": float,
    "max": max,
    "min": min,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "pow": pow,
}

_MAX_POW = 10000
_MAX_FACTORIAL = 170


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp):
        fn = _UNARY.get(type(node.op))
        if fn is None:
            raise ValueError("使えない演算子です")
        return fn(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        fn = _BINARY.get(type(node.op))
        if fn is None:
            raise ValueError("使えない演算子です")
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > _MAX_POW:
                raise ValueError("指数が大きすぎます")
        return fn(left, right)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _FUNCTIONS.get(node.func.id.lower())
        if fn is None:
            raise ValueError(f"不明な関数: {node.func.id}")
        args = [_eval_node(a) for a in node.args]
        if node.func.id.lower() == "factorial" and args and args[0] > _MAX_FACTORIAL:
            raise ValueError("階乗が大きすぎます")
        return fn(*args)
    if isinstance(node, ast.Name):
        val = _CONSTANTS.get(node.id.lower())
        if val is not None:
            return val
        raise ValueError(f"不明な名前: {node.id}")
    raise ValueError("使えない構文です")


_FULLWIDTH = str.maketrans(
    "０１２３４５６７８９＋－＊／（）．＝",
    "0123456789+-*/().=")


def _normalize(expr: str) -> str:
    expr = expr.translate(_FULLWIDTH)
    expr = expr.replace("×", "*").replace("÷", "/").replace("−", "-")
    expr = expr.replace("^", "**")
    expr = re.sub(r"(\d+\.?\d*)%(?!\s*\d)", r"(\1/100)", expr)
    return expr.strip().rstrip("=").strip()


def _fmt(value) -> str:
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return str(value)
        if value == int(value) and abs(value) < 1e15:
            return str(int(value))
        return f"{value:g}"
    return str(value)


# -- straight-line programs ---------------------------------------------------
# A word problem's calculation as the model writes it out, variables and all:
#   仕事量 = 1/12 + 1/6
#   答え = 1 / 仕事量
# Evaluated here, on the syntax tree, in exact fractions: no loops, no
# attributes, no imports, only the functions below. Nothing is executed.
from fractions import Fraction  # noqa: E402

_PROGRAM_FUNCS = {
    "comb": math.comb, "perm": math.perm, "factorial": math.factorial, "gcd": math.gcd,
    "lcm": math.lcm, "sqrt": math.sqrt, "abs": abs, "round": round, "min": min, "max": max,
    "floor": math.floor, "ceil": math.ceil, "sum": sum, "int": int, "float": float,
}
_INT_ARGS = {"comb", "perm", "factorial", "gcd", "lcm"}
_PROGRAM_BINARY = {**_BINARY}
_ANSWER_NAMES = ("答え", "answer", "ans", "result", "解")
_MAX_STATEMENTS, _MAX_INT_ARG, _MAX_DIGITS = 40, 10000, 200


def _to_number(value):
    if isinstance(value, bool):
        raise ValueError("真偽値は使えません")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise ValueError("計算できない値です")
        return Fraction(str(value))
    if isinstance(value, Fraction):
        if abs(value.numerator).bit_length() > _MAX_DIGITS * 4:
            raise ValueError("値が大きすぎます")
        return value
    raise ValueError("数ではありません")


def _as_int(value) -> int:
    if isinstance(value, Fraction) and value.denominator == 1 and abs(value) <= _MAX_INT_ARG:
        return int(value)
    raise ValueError("整数が必要です")


def _program_node(node, env: dict):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return _to_number(node.value)
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id.lower() in _CONSTANTS:
            return _to_number(_CONSTANTS[node.id.lower()])
        raise ValueError(f"不明な名前: {node.id}")
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_program_node(node.operand, env))
    if isinstance(node, ast.BinOp) and type(node.op) in _PROGRAM_BINARY:
        left, right = _program_node(node.left, env), _program_node(node.right, env)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, Fraction) and right.denominator == 1 and abs(right) <= 1000 \
                    and abs(left) < 10 ** 6:
                return _to_number(left ** int(right))
            return _to_number(float(left) ** float(right))
        if isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)) and right == 0:
            raise ZeroDivisionError
        return _to_number(_PROGRAM_BINARY[type(node.op)](left, right))
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_program_node(e, env) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id.lower()
        fn = _PROGRAM_FUNCS.get(name)
        if fn is None:
            raise ValueError(f"使えない関数: {node.func.id}")
        args = [_program_node(a, env) for a in node.args]
        if name in _INT_ARGS:
            args = [_as_int(a) for a in args]
            if name == "factorial" and args and args[0] > _MAX_FACTORIAL:
                raise ValueError("階乗が大きすぎます")
        elif name in ("sqrt", "floor", "ceil", "round", "int", "float"):
            args = [float(a) if isinstance(a, Fraction) else a for a in args]
        return _to_number(fn(*args))
    raise ValueError("使えない構文です")


def run_program(text: str):
    """The answer of a straight-line calculation, as an exact Fraction.

    Statements are `name = expression` or a bare expression; the answer is the
    variable 答え (or answer/ans/result), else the last value. Raises ValueError
    (or ZeroDivisionError) on anything else.
    """
    tree = ast.parse(_normalize_program(text), mode="exec")
    if not tree.body or len(tree.body) > _MAX_STATEMENTS:
        raise ValueError("式の数が範囲外です")
    env, last = {}, None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 \
                and isinstance(stmt.targets[0], ast.Name):
            last = env[stmt.targets[0].id] = _program_node(stmt.value, env)
        elif isinstance(stmt, ast.Expr):
            last = _program_node(stmt.value, env)
        else:
            raise ValueError("使えない文です")
    for name in _ANSWER_NAMES:
        if name in env:
            last = env[name]
            break
    if not isinstance(last, Fraction):
        raise ValueError("答えが数ではありません")
    return last


def _normalize_program(text: str) -> str:
    """The calculation lines of what the model wrote, as Python expressions.

    It does not always write bare code: seen were "calc: 3000*(1-0.15)" (a
    variable annotation to Python, so the whole check was skipped), backticks,
    "-> 2550" results and a closing sentence. Prose lines (hiragana, no "=")
    are dropped; an assignment may still have a Japanese name (答え).
    """
    lines = []
    for line in text.translate(_FULLWIDTH).splitlines():
        line = line.split("#", 1)[0].strip().strip("`").strip()
        line = re.sub(r"^(?:式|calc|計算)\s*[:：]\s*", "", line)
        line = re.split(r"\s*(?:->|→|=>)\s*", line)[0]
        if not line:
            continue
        if not re.match(r"^[^\s=]+\s*=[^=]", line) and re.search(r"[぀-ゟ]", line):
            continue  # a sentence, not a calculation
        line = line.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
        line = re.sub(r"(\d+\.?\d*)\s*[%％](?!\s*\d)", r"(\1/100)", line)
        lines.append(line)
    return "\n".join(lines)


def calc(expression: str) -> str:
    expression = expression.strip()
    if not expression:
        return "計算式が空です。"
    normalized = _normalize(expression)
    if not normalized:
        return "計算式が空です。"
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return f"式を解析できませんでした: {expression}"
    try:
        result = _eval_node(tree)
        return f"{expression} = {_fmt(result)}"
    except ZeroDivisionError:
        return "0で割ることはできません。"
    except (ValueError, TypeError, OverflowError) as e:
        return f"計算できませんでした: {e}"


CALC_TOOL = Tool(
    name="calc",
    description="数式を計算する。暗算せず必ずこれを使う。",
    usage="calc: 式 (例: 123*456+789, sqrt(2), 3.14*5**2)",
    run=calc,
)
