# main.py
"""
간단한 대화형 챗봇 실행 스크립트
"""
import logging
import sys
from datetime import datetime
from pathlib import Path
from chatbot import EmotionChatbot


def setup_main_logger(log_level: str = "INFO") -> logging.Logger:
    """메인 애플리케이션 로거 설정"""
    logger = logging.getLogger("MainApp")

    # 이미 핸들러가 있으면 중복 설정 방지
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(
        log_dir / f"main_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)

    # 콘솔 핸들러 설정
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # 포매터 설정
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    )
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def simple_chat(log_level: str = "INFO"):
    """간단한 대화 인터페이스"""
    # 로거 초기화
    logger = setup_main_logger(log_level)
    logger.info("감정 인식 챗봇 애플리케이션 시작")

    print("🤖 감정 인식 챗봇을 시작합니다!")
    print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
    print("메모리 초기화는 'clear'를 입력하세요.")
    print("도움말은 'help'를 입력하세요.")
    print("=" * 50)

    try:
        # 챗봇 초기화
        logger.info("챗봇 초기화 시작")

        # 모델 경로 - 실제 경로로 수정 필요
        model_dir = "/Users/hwangeunbi/chatnge_AI/models"

        logger.info(f"모델 디렉토리: {model_dir}")

        chatbot = EmotionChatbot(
            model_dir=model_dir,
            confidence_threshold=0.6,
            emotion_threshold=0.3,
            log_level=log_level,  # 챗봇도 같은 로그 레벨 사용
        )

        logger.info("챗봇 초기화 완료")
        print("✅ 챗봇 초기화 완료!\n")

        conversation_count = 0

        while True:
            try:
                user_input = input("👤 You: ").strip()
                conversation_count += 1

                logger.debug(
                    f"사용자 입력 #{conversation_count}: {user_input[:100]}..."
                )

                # 종료 명령 처리
                if user_input.lower() in ["quit", "exit", "종료"]:
                    logger.info("사용자가 종료 요청")
                    print("👋 안녕히 가세요!")
                    break

                # 메모리 초기화 명령 처리
                if user_input.lower() in ["clear", "초기화"]:
                    logger.info("메모리 초기화 요청")
                    chatbot.clear_memory()
                    continue

                # 도움말 명령 처리
                if user_input.lower() in ["help", "도움말"]:
                    print_help()
                    continue

                # 메모리 상태 확인 명령
                if user_input.lower() in ["status", "상태"]:
                    show_memory_status(chatbot, logger)
                    continue

                # 빈 입력 처리
                if not user_input:
                    logger.debug("빈 입력 무시")
                    continue

                logger.info(f"대화 #{conversation_count} 처리 시작")

                # 스마트 응답 생성 (모델 자동 선택)
                result = chatbot.generate_smart_response(user_input)

                # 응답 출력
                if "error" in result:
                    logger.error(f"챗봇 응답 생성 오류: {result.get('error')}")
                    print(f"❌ 오류: {result['ai_response']}")
                else:
                    print(f"🤖 Bot ({result['model_used']}): {result['ai_response']}")

                    # 감정 분석 결과 출력 (간단하게)
                    if "emotion_analysis" in result:
                        emotion = result["emotion_analysis"]["final"]["prediction"]
                        confidence = result["emotion_analysis"]["final"]["confidence"]
                        print(f"📊 감정: {emotion} ({confidence:.2f})")

                        logger.info(
                            f"대화 #{conversation_count} 완료 - 감정: {emotion}, 모델: {result['model_used']}"
                        )

                    print()  # 빈 줄 추가

            except KeyboardInterrupt:
                logger.info("사용자가 Ctrl+C로 중단")
                print("\n👋 대화를 종료합니다.")
                break

            except EOFError:
                logger.info("EOF 신호로 종료")
                print("\n👋 대화를 종료합니다.")
                break

            except Exception as e:
                logger.error(f"대화 처리 중 예외 발생: {e}", exc_info=True)
                print(f"❌ 오류가 발생했습니다: {e}")
                print("다시 시도해주세요.\n")

    except FileNotFoundError as e:
        logger.error(f"모델 파일을 찾을 수 없음: {e}")
        print(f"❌ 모델 파일을 찾을 수 없습니다: {e}")
        print("모델 경로를 확인해주세요.")

    except ValueError as e:
        logger.error(f"설정 오류: {e}")
        print(f"❌ 설정 오류: {e}")
        print("API 키를 확인해주세요.")

    except Exception as e:
        logger.error(f"챗봇 초기화 실패: {e}", exc_info=True)
        print(f"❌ 챗봇 초기화 실패: {e}")
        print("모델 경로와 API 키를 확인해주세요.")

    finally:
        logger.info(f"챗봇 애플리케이션 종료 - 총 대화 수: {conversation_count}")


def print_help():
    """도움말 출력"""
    help_text = """
📋 사용 가능한 명령어:
• quit, exit, 종료     : 챗봇 종료
• clear, 초기화        : 대화 기록 초기화
• status, 상태         : 메모리 상태 확인
• help, 도움말         : 이 도움말 표시

💡 팁:
• 감정이 포함된 메시지는 더 정교한 모델로 처리됩니다
• 일반적인 대화는 빠른 모델로 처리됩니다
• 대화 기록은 자동으로 요약되어 저장됩니다
"""
    print(help_text)


def show_memory_status(chatbot, logger):
    """메모리 상태 표시"""
    try:
        status = chatbot.get_memory_status()

        if "error" in status:
            print(f"❌ 메모리 상태 확인 실패: {status['error']}")
            logger.error(f"메모리 상태 확인 실패: {status['error']}")
        else:
            print(
                f"""
📊 메모리 상태:
• 메시지 수: {status['message_count']}개
• 토큰 수: {status['token_count']}개
• 최대 토큰: {status['max_token_limit']}개
• 사용률: {status['token_count']/status['max_token_limit']*100:.1f}%
"""
            )
            logger.info(
                f"메모리 상태 조회 - 메시지: {status['message_count']}, 토큰: {status['token_count']}"
            )

    except Exception as e:
        print(f"❌ 메모리 상태 확인 중 오류: {e}")
        logger.error(f"메모리 상태 확인 중 오류: {e}")


def main():
    """메인 함수 - 명령행 인자 처리"""
    import argparse

    parser = argparse.ArgumentParser(description="감정 인식 챗봇")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="로그 레벨 설정 (기본값: INFO)",
    )
    parser.add_argument("--model-dir", type=str, help="모델 디렉토리 경로")

    args = parser.parse_args()

    # 로그 레벨 적용
    log_level = args.log_level

    # 모델 디렉토리가 지정된 경우 업데이트
    if args.model_dir:
        # 여기서 모델 경로를 동적으로 설정할 수 있음
        print(f"모델 디렉토리: {args.model_dir}")

    print(f"로그 레벨: {log_level}")
    print("로그 파일은 'logs/' 디렉토리에 저장됩니다.")
    print()

    # 챗봇 시작
    simple_chat(log_level)


if __name__ == "__main__":
    main()
