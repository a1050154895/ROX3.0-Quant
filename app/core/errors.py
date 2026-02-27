"""
统一 API 错误响应格式，便于前端与监控解析
"""
from typing import Optional, Any
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

# 错误代码映射
ERROR_CODES = {
    "VALIDATION_ERROR": "请求参数校验失败",
    "AUTH_ERROR": "认证失败",
    "PERMISSION_ERROR": "权限不足",
    "NOT_FOUND": "资源不存在",
    "DUPLICATE_RESOURCE": "资源已存在",
    "RATE_LIMIT_EXCEEDED": "请求过于频繁",
    "INTERNAL_ERROR": "服务器内部错误",
    "NETWORK_ERROR": "网络错误",
    "DATA_ERROR": "数据错误",
}

def error_response(
    error: str,
    code: Optional[str] = None,
    status_code: int = 400,
    details: Optional[Any] = None,
) -> JSONResponse:
    """统一错误体：{ "error": str, "code"?: str, "details"?: any }"""
    body = {"error": error}
    if code:
        body["code"] = code
    if details is not None:
        body["details"] = details
    return JSONResponse(status_code=status_code, content=body)


def register_exception_handlers(app):
    """注册全局异常处理器"""
    from fastapi import HTTPException

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        # 为常见的 HTTP 错误提供更友好的错误信息
        error_map = {
            400: ("请求参数错误", "VALIDATION_ERROR"),
            401: ("认证失败，请重新登录", "AUTH_ERROR"),
            403: ("权限不足，无法访问该资源", "PERMISSION_ERROR"),
            404: ("请求的资源不存在", "NOT_FOUND"),
            409: ("资源已存在", "DUPLICATE_RESOURCE"),
            429: ("请求过于频繁，请稍后再试", "RATE_LIMIT_EXCEEDED"),
        }
        
        error_msg, error_code = error_map.get(exc.status_code, (exc.detail if isinstance(exc.detail, str) else str(exc.detail), f"HTTP_{exc.status_code}"))
        
        return error_response(
            error=error_msg,
            code=error_code,
            status_code=exc.status_code,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        # 优化参数校验错误信息
        error_messages = []
        for error in exc.errors():
            field = error.get("loc", [])[-1] if error.get("loc") else "未知字段"
            msg = error.get("msg", "参数错误")
            error_messages.append(f"{field}: {msg}")
        
        return error_response(
            error="请求参数校验失败",
            code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={"messages": error_messages, "original": exc.errors()},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        import logging
        logging.getLogger("rox-backend").exception("Unhandled exception: %s", exc)
        
        # 为常见异常类型提供更友好的错误信息
        exception_map = {
            "ValueError": ("数据值错误", 400),
            "TypeError": ("参数类型错误", 400),
            "KeyError": ("缺少必要参数", 400),
        }
        
        exc_type = type(exc).__name__
        error_msg, status_code = exception_map.get(exc_type, ("服务器内部错误", 500))
        
        return error_response(
            error=error_msg,
            code="INTERNAL_ERROR" if status_code == 500 else "VALIDATION_ERROR",
            status_code=status_code,
        )
