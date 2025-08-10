# logging_utils.py
"""
로깅 유틸리티 모듈
전체 애플리케이션의 로깅을 중앙에서 관리
"""
import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import functools
import time


class ColoredFormatter(logging.Formatter):
    """색상이 있는 콘솔 로그 포매터"""

    COLORS = {
        "DEBUG": "\033[36m",  # 청록색
        "INFO": "\033[32m",  # 녹색
        "WARNING": "\033[33m",  # 노란색
        "ERROR": "\033[31m",  # 빨간색
        "CRITICAL": "\033[35m",  # 마젠타
    }
    RESET = "\033[0m"

    def format(self, record):
        # 색상 적용
        if record.levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
            )

        return super().format(record)


class JSONFormatter(logging.Formatter):
    """JSON 형태의 로그 포매터"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        # 추가 필드가 있으면 포함
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data

        return json.dumps(log_data, ensure_ascii=False)


class LoggingManager:
    """중앙 로깅 관리자"""

    def __init__(
        self,
        log_dir: str = "logs",
        app_name: str = "ChatBot",
        default_level: str = "INFO",
        enable_file_logging: bool = True,
        enable_console_logging: bool = True,
        use_colors: bool = True,
        use_json_format: bool = False,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
    ):

        self.log_dir = Path(log_dir)
        self.app_name = app_name
        self.default_level = getattr(logging, default_level.upper())
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.use_colors = use_colors
        self.use_json_format = use_json_format
        self.max_file_size = max_file_size
        self.backup_count = backup_count

        # 로그 디렉토리 생성
        self.log_dir.mkdir(exist_ok=True)

        # 설정된 로거들 추적
        self._configured_loggers = set()

        # 기본 로거 설정
        self._setup_root_logger()

    def _setup_root_logger(self):
        """루트 로거 설정"""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.default_level)

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    def get_logger(self, name: str, level: Optional[str] = None) -> logging.Logger:
        """새 로거 생성 또는 기존 로거 반환"""
        logger = logging.getLogger(name)

        # 이미 설정된 로거면 반환
        if name in self._configured_loggers:
            return logger

        # 레벨 설정
        log_level = getattr(logging, level.upper()) if level else self.default_level
        logger.setLevel(log_level)

        # 핸들러 설정
        if self.enable_file_logging:
            self._add_file_handler(logger, name)

        if self.enable_console_logging:
            self._add_console_handler(logger)

        # 상위 로거로 전파 방지 (중복 로그 방지)
        logger.propagate = False

        self._configured_loggers.add(name)
        return logger

    def _add_file_handler(self, logger: logging.Logger, name: str):
        """파일 핸들러 추가"""
        # 로그 파일명 생성
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_"))
        log_file = self.log_dir / f"{safe_name}_{datetime.now().strftime('%Y%m%d')}.log"

        # 회전 파일 핸들러 사용
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size,
            backupCount=self.backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)

        # 포매터 설정
        if self.use_json_format:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
            )
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    def _add_console_handler(self, logger: logging.Logger):
        """콘솔 핸들러 추가"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.default_level)

        # 포매터 설정
        if self.use_colors and sys.stdout.isatty():
            formatter = ColoredFormatter("%(levelname)s - %(name)s - %(message)s")
        else:
            formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    def set_level(self, logger_name: str, level: str):
        """특정 로거의 레벨 변경"""
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level.upper()))

    def disable_logger(self, logger_name: str):
        """특정 로거 비활성화"""
        logger = logging.getLogger(logger_name)
        logger.disabled = True

    def enable_logger(self, logger_name: str):
        """특정 로거 활성화"""
        logger = logging.getLogger(logger_name)
        logger.disabled = False

    def get_log_files(self) -> list:
        """생성된 로그 파일 목록 반환"""
        return list(self.log_dir.glob("*.log"))

    def cleanup_old_logs(self, days: int = 7):
        """오래된 로그 파일 정리"""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)

        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_mtime < cutoff:
                log_file.unlink()

    def get_logger_status(self) -> Dict[str, Any]:
        """로거 상태 정보 반환"""
        return {
            "configured_loggers": list(self._configured_loggers),
            "log_directory": str(self.log_dir),
            "default_level": logging.getLevelName(self.default_level),
            "file_logging": self.enable_file_logging,
            "console_logging": self.enable_console_logging,
            "log_files": [str(f) for f in self.get_log_files()],
        }


# 전역 로깅 매니저 인스턴스
_logging_manager = None


def get_logging_manager(**kwargs) -> LoggingManager:
    """전역 로깅 매니저 인스턴스 반환"""
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = LoggingManager(**kwargs)

    return _logging_manager


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """편의 함수: 로거 생성"""
    manager = get_logging_manager()
    return manager.get_logger(name, level)


def log_execution_time(logger: Optional[logging.Logger] = None):
    """함수 실행 시간을 로그하는 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                logger = get_logger(f"{func.__module__}.{func.__qualname__}")

            start_time = time.time()
            logger.debug(f"함수 {func.__name__} 실행 시작")

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(
                    f"함수 {func.__name__} 실행 완료 - 소요시간: {execution_time:.3f}초"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(
                    f"함수 {func.__name__} 실행 실패 - 소요시간: {execution_time:.3f}초, 오류: {e}"
                )
                raise

        return wrapper

    return decorator


def log_method_calls(logger: Optional[logging.Logger] = None):
    """메서드 호출을 로그하는 데코레이터"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            nonlocal logger
            if logger is None:
                # 클래스 메서드인 경우 클래스명 포함
                if args and hasattr(args[0], "__class__"):
                    class_name = args[0].__class__.__name__
                    logger = get_logger(f"{func.__module__}.{class_name}")
                else:
                    logger = get_logger(f"{func.__module__}.{func.__qualname__}")

            logger.debug(
                f"메서드 {func.__name__} 호출 - args: {len(args)}, kwargs: {list(kwargs.keys())}"
            )

            try:
                result = func(*args, **kwargs)
                logger.debug(f"메서드 {func.__name__} 성공")
                return result
            except Exception as e:
                logger.error(f"메서드 {func.__name__} 실패: {e}")
                raise

        return wrapper

    return decorator


def setup_chatbot_logging(
    log_level: str = "INFO",
    use_colors: bool = True,
    enable_performance_logging: bool = False,
) -> LoggingManager:
    """챗봇 애플리케이션용 로깅 설정"""

    manager = get_logging_manager(
        log_dir="logs",
        app_name="EmotionChatBot",
        default_level=log_level,
        enable_file_logging=True,
        enable_console_logging=True,
        use_colors=use_colors,
        use_json_format=False,
    )

    # 주요 로거들 미리 설정
    loggers_to_setup = [
        "EmotionChatbot",
        "HierarchicalEmotionClassifier",
        "TextPreprocessor",
        "MainApp",
        "Config",
    ]

    for logger_name in loggers_to_setup:
        manager.get_logger(logger_name, log_level)

    # 성능 로깅 활성화
    if enable_performance_logging:
        perf_logger = manager.get_logger("Performance", "DEBUG")
        perf_logger.info("성능 로깅 활성화됨")

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    main_logger = manager.get_logger("ChatbotApp")
    main_logger.info("챗봇 로깅 시스템 초기화 완료")
    main_logger.info(f"로그 디렉토리: {manager.log_dir}")
    main_logger.info(f"로그 레벨: {log_level}")

    return manager


# 사용 예시
if __name__ == "__main__":
    # 로깅 시스템 테스트
    manager = setup_chatbot_logging("DEBUG", use_colors=True)

    # 테스트 로거들
    test_logger = get_logger("TestLogger")
    test_logger.debug("디버그 메시지")
    test_logger.info("정보 메시지")
    test_logger.warning("경고 메시지")
    test_logger.error("에러 메시지")

    # 상태 정보 출력
    status = manager.get_logger_status()
    print(f"로깅 상태: {status}")

    @log_execution_time()
    def test_function():
        time.sleep(0.1)
        return "완료"

    result = test_function()
    test_logger.info(f"테스트 함수 결과: {result}")
