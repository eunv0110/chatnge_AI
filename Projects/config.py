# config.py
"""설정 파일"""
import os
import logging
from pathlib import Path
from datetime import datetime


def setup_config_logger() -> logging.Logger:
    """설정 모듈용 로거"""
    logger = logging.getLogger("Config")

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 파일 핸들러
    file_handler = logging.FileHandler(
        log_dir / f"config_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    # 포매터
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


# 로거 초기화
config_logger = setup_config_logger()
config_logger.info("설정 모듈 로드 시작")

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent
config_logger.info(f"프로젝트 루트 경로: {PROJECT_ROOT}")

# 모델 관련 설정
MODEL_DIR = PROJECT_ROOT / "models"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOGS_DIR = PROJECT_ROOT / "logs"

# 디렉토리 생성
for directory in [MODEL_DIR, PROMPTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
    config_logger.debug(f"디렉토리 확인/생성: {directory}")

config_logger.info(f"모델 디렉토리: {MODEL_DIR}")
config_logger.info(f"프롬프트 디렉토리: {PROMPTS_DIR}")
config_logger.info(f"로그 디렉토리: {LOGS_DIR}")

# 기본 임계값
DEFAULT_CONFIDENCE_THRESHOLD = 0.6
DEFAULT_EMOTION_THRESHOLD = 0.3

# 모델 설정
MAX_SEQUENCE_LENGTH = 128
DROPOUT_RATE = 0.3

# LLM 설정
DEFAULT_TEMPERATURE = 0.3
MAX_TOKENS = 1200
MEMORY_TOKEN_LIMIT = 2000

# API 키 (환경변수에서 로드)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# API 키 상태 로깅 (키 자체는 로그에 남기지 않음)
api_key_status = {
    "OPENAI": "설정됨" if OPENAI_API_KEY else "없음",
    "ANTHROPIC": "설정됨" if ANTHROPIC_API_KEY else "없음",
}
config_logger.info(f"API 키 상태: {api_key_status}")

# 로그 설정
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
DEFAULT_LOG_LEVEL = "INFO"

# 지원되는 모델 목록
SUPPORTED_MODELS = {
    "openai": ["gpt-3.5-turbo", "gpt-4o"],
    "anthropic": ["claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
}

# 감정 분류 레벨 설정
EMOTION_LEVELS = {
    "level1": "일반대화 vs 감정",
    "level2": "기쁨 vs 기타감정",
    "level3": "세부 감정 분류",
}

config_logger.info("설정 모듈 로드 완료")


# 설정 검증 함수
def validate_config():
    """설정 유효성 검사"""
    config_logger.info("설정 유효성 검사 시작")

    issues = []

    # 모델 디렉토리 확인
    if not MODEL_DIR.exists():
        issues.append(f"모델 디렉토리가 존재하지 않음: {MODEL_DIR}")

    # API 키 확인
    if not OPENAI_API_KEY and not ANTHROPIC_API_KEY:
        issues.append("OpenAI 또는 Anthropic API 키 중 최소 하나는 필요함")

    # 필수 모델 파일 확인
    required_model_files = [
        "level1_best_model.pt",
        "level2_best_model.pt",
        "level3_best_model.pt",
        "level1_label_encoder.pkl",
        "level2_label_encoder.pkl",
        "level3_label_encoder.pkl",
    ]

    for file in required_model_files:
        file_path = MODEL_DIR / file
        if not file_path.exists():
            issues.append(f"필수 모델 파일 누락: {file_path}")

    if issues:
        config_logger.warning(f"설정 검증에서 {len(issues)}개 문제 발견:")
        for issue in issues:
            config_logger.warning(f"  - {issue}")
        return False, issues
    else:
        config_logger.info("설정 검증 완료 - 모든 설정이 유효함")
        return True, []


def get_config_summary():
    """설정 요약 정보 반환"""
    return {
        "project_root": str(PROJECT_ROOT),
        "model_dir": str(MODEL_DIR),
        "prompts_dir": str(PROMPTS_DIR),
        "logs_dir": str(LOGS_DIR),
        "api_keys": api_key_status,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "emotion_threshold": DEFAULT_EMOTION_THRESHOLD,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "dropout_rate": DROPOUT_RATE,
        "default_temperature": DEFAULT_TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "memory_token_limit": MEMORY_TOKEN_LIMIT,
        "supported_models": SUPPORTED_MODELS,
        "emotion_levels": EMOTION_LEVELS,
    }


def log_config_summary():
    """설정 요약을 로그에 기록"""
    config_logger.info("=== 현재 설정 요약 ===")
    summary = get_config_summary()

    for key, value in summary.items():
        if key == "api_keys":
            config_logger.info(f"{key}: {value}")
        elif isinstance(value, dict):
            config_logger.info(f"{key}:")
            for sub_key, sub_value in value.items():
                config_logger.info(f"  {sub_key}: {sub_value}")
        else:
            config_logger.info(f"{key}: {value}")

    config_logger.info("=== 설정 요약 완료 ===")


# 모듈 로드 시 설정 요약 로그 기록
if __name__ != "__main__":
    log_config_summary()


# 개발/운영 환경 설정
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
config_logger.info(f"실행 환경: {ENVIRONMENT}")

# 환경별 설정
if ENVIRONMENT == "production":
    DEFAULT_LOG_LEVEL = "INFO"
    config_logger.info("운영 환경 설정 적용")
elif ENVIRONMENT == "development":
    DEFAULT_LOG_LEVEL = "DEBUG"
    config_logger.info("개발 환경 설정 적용")
else:
    DEFAULT_LOG_LEVEL = "INFO"
    config_logger.info("기본 환경 설정 적용")


# 성능 모니터링 설정
ENABLE_PERFORMANCE_LOGGING = (
    os.getenv("ENABLE_PERFORMANCE_LOGGING", "false").lower() == "true"
)
config_logger.info(
    f"성능 로깅: {'활성화' if ENABLE_PERFORMANCE_LOGGING else '비활성화'}"
)

# 디버그 모드 설정
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
config_logger.info(f"디버그 모드: {'활성화' if DEBUG_MODE else '비활성화'}")

config_logger.info("config.py 모듈 로드 완료")
