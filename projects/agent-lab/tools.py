from langchain_core.tools import tool


@tool
def multiply(a: float, b: float) -> str:
    """两数相乘，返回乘积。"""
    return str(a * b)
