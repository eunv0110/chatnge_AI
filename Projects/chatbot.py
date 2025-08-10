# chatbot.py
import os
import yaml
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from emotion_classifier import HierarchicalEmotionClassifier

load_dotenv()


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """로거 설정 함수"""
    logger = logging.getLogger(name)

    # 이미 핸들러가 있으면 중복 설정 방지
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper()))

    # 로그 디렉토리 생성
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 파일 핸들러 설정
    file_handler = logging.FileHandler(
        log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log", encoding="utf-8"
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


class EmotionChatbot:
    def __init__(
        self,
        model_dir,
        confidence_threshold=0.6,
        emotion_threshold=0.3,
        log_level="INFO",
    ):
        # 로거 초기화
        self.logger = setup_logger("EmotionChatbot", log_level)
        self.logger.info("EmotionChatbot 초기화 시작")

        try:
            # 감정 분류기 초기화
            self.logger.info(f"감정 분류기 초기화 시작 - 모델 경로: {model_dir}")
            self.classifier = HierarchicalEmotionClassifier(
                model_dir,
                confidence_threshold=confidence_threshold,
                emotion_threshold=emotion_threshold,
            )
            self.logger.info("감정 분류기 초기화 완료")

            self.confidence_threshold = confidence_threshold
            self.emotion_threshold = emotion_threshold

            # 프롬프트 로드
            self.logger.info("프롬프트 로드 시작")
            self.prompts = self._load_prompts()
            self.logger.info("프롬프트 로드 완료")

            # LLM 초기화
            self.logger.info("LLM 초기화 시작")
            self.llms = self._initialize_llms()
            self.logger.info(
                f"LLM 초기화 완료 - 사용 가능한 모델: {list(self.llms.keys())}"
            )

            # 메모리 설정
            self.logger.info("메모리 및 체인 설정 시작")
            self._setup_memory_and_chains()
            self.logger.info("메모리 및 체인 설정 완료")

            self.logger.info("EmotionChatbot 초기화 성공")

        except Exception as e:
            self.logger.error(f"EmotionChatbot 초기화 실패: {e}", exc_info=True)
            raise

    def _load_prompts(self):
        """YAML 파일에서 프롬프트들을 로드"""
        prompts = {}

        prompt_files = {
            "basic_system": "prompts/basic_system.yaml",
            "general_conversation": "prompts/general_conversation.yaml",
        }

        for prompt_name, file_path in prompt_files.items():
            try:
                if os.path.exists(file_path):
                    self.logger.debug(f"프롬프트 파일 로드 시도: {file_path}")
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if isinstance(data, dict) and "template" in data:
                            prompts[prompt_name] = data["template"]
                        elif isinstance(data, str):
                            prompts[prompt_name] = data
                        self.logger.info(f"프롬프트 로드 성공: {prompt_name}")
                else:
                    self.logger.warning(f"프롬프트 파일 없음, 기본값 사용: {file_path}")
                    # 기본 프롬프트 설정
                    if prompt_name == "basic_system":
                        prompts[prompt_name] = (
                            "당신은 공감적이고 따뜻한 AI 상담사입니다. 사용자의 감정을 이해하고 적절한 응답을 제공해주세요."
                        )
                    elif prompt_name == "general_conversation":
                        prompts[prompt_name] = (
                            "당신은 친근하고 도움이 되는 AI 친구입니다. 자연스럽고 친근한 대화를 나누어주세요."
                        )
            except Exception as e:
                self.logger.error(f"{file_path} 로드 중 오류: {e}")
                # 기본값 설정
                if prompt_name == "basic_system":
                    prompts[prompt_name] = "당신은 공감적이고 따뜻한 AI 상담사입니다."
                elif prompt_name == "general_conversation":
                    prompts[prompt_name] = "당신은 친근한 AI 친구입니다."

        return prompts

    def _initialize_llms(self):
        """LLM 모델들 초기화"""
        llms = {}

        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")

        self.logger.info(
            f"API 키 확인 - OpenAI: {'설정됨' if openai_key else '없음'}, Anthropic: {'설정됨' if anthropic_key else '없음'}"
        )

        try:
            if openai_key:
                self.logger.debug("OpenAI 모델 초기화 시작")
                # 일반 대화용
                llms["gpt35"] = ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.6,
                    max_tokens=800,
                    api_key=openai_key,
                )

                # 감정 대화용
                llms["gpt4o"] = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.3,
                    max_tokens=1200,
                    api_key=openai_key,
                )
                self.logger.info("OpenAI 모델 초기화 완료")

            if anthropic_key:
                self.logger.debug("Anthropic 모델 초기화 시작")
                llms["claude"] = ChatAnthropic(
                    model="claude-3-sonnet-20240229",
                    temperature=0.3,
                    max_tokens=1200,
                    api_key=anthropic_key,
                )
                self.logger.info("Claude 모델 초기화 완료")

        except Exception as e:
            self.logger.error(f"LLM 초기화 오류: {e}", exc_info=True)
            raise

        if not llms:
            error_msg = "사용 가능한 LLM이 없습니다. API 키를 확인해주세요."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return llms

    def _setup_memory_and_chains(self):
        """메모리와 체인 설정"""
        # 주요 모델 선택
        if "gpt4o" in self.llms:
            primary_llm = self.llms["gpt4o"]
            primary_model_name = "gpt4o"
        elif "claude" in self.llms:
            primary_llm = self.llms["claude"]
            primary_model_name = "claude"
        else:
            primary_llm = list(self.llms.values())[0]
            primary_model_name = list(self.llms.keys())[0]

        self.logger.info(f"주요 LLM 선택: {primary_model_name}")
        self.llm = primary_llm

        # 공유 메모리 생성
        try:
            self.shared_memory = ConversationSummaryBufferMemory(
                llm=primary_llm,
                return_messages=True,
                max_token_limit=2000,
                memory_key="chat_history",
            )
            self.logger.debug("메모리 생성 완료")

            # 기본 체인 설정
            self.system_prompt_text = self.prompts.get("basic_system")
            self.chat_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", self.system_prompt_text),
                    MessagesPlaceholder(variable_name="chat_history"),
                    ("human", "{message}"),
                ]
            )

            self.chain = self.chat_prompt | self.llm
            self.logger.info("체인 설정 완료")

        except Exception as e:
            self.logger.error(f"메모리 및 체인 설정 오류: {e}", exc_info=True)
            raise

    def _choose_model(self, text):
        """텍스트 분석 후 적절한 모델 선택"""
        try:
            self.logger.debug(f"모델 선택을 위한 감정 분석 시작: {text[:50]}...")
            emotion_result = self.classify_emotion(text)
            emotion = emotion_result["final"]["prediction"]
            confidence = emotion_result["final"]["confidence"]

            self.logger.debug(f"감정 분석 결과: {emotion} (신뢰도: {confidence:.3f})")

            # 일반대화면 GPT-3.5
            if emotion == "일반대화" and confidence > 0.7:
                if "gpt35" in self.llms:
                    self.logger.info(
                        f"GPT-3.5 선택 (일반대화, 신뢰도: {confidence:.3f})"
                    )
                    return "gpt35", emotion_result

            # 감정관련이면 GPT-4o 또는 Claude
            if "gpt4o" in self.llms:
                self.logger.info(f"GPT-4o 선택 (감정 대화: {emotion})")
                return "gpt4o", emotion_result
            elif "claude" in self.llms:
                self.logger.info(f"Claude 선택 (감정 대화: {emotion})")
                return "claude", emotion_result

            # 폴백: 사용 가능한 첫 번째 모델
            available_model = list(self.llms.keys())[0]
            self.logger.warning(f"폴백 모델 사용: {available_model}")
            return available_model, emotion_result

        except Exception as e:
            self.logger.error(f"모델 선택 중 오류: {e}", exc_info=True)
            available_model = list(self.llms.keys())[0]
            return available_model, {"error": str(e)}

    def classify_emotion(self, text):
        """감정 분류"""
        try:
            self.logger.debug(f"감정 분류 시작: {text[:100]}...")
            result = self.classifier.predict_hierarchical(text)
            emotion = result["final"]["prediction"]
            confidence = result["final"]["confidence"]
            self.logger.debug(f"감정 분류 완료: {emotion} (신뢰도: {confidence:.3f})")
            return result
        except Exception as e:
            self.logger.error(f"감정 분류 중 오류: {e}", exc_info=True)
            raise

    def format_analysis(self, result):
        """분석 결과 포맷팅"""
        try:
            output = f"""
📝 입력: {result['original_text']}
🎯 최종 결과: {result['final']['prediction']} (신뢰도: {result['final']['confidence']:.4f})
📊 예측 경로: {' → '.join(result['path'])}
"""
            return output
        except Exception as e:
            self.logger.error(f"분석 결과 포맷팅 오류: {e}")
            return "분석 결과 포맷팅 실패"

    def generate_response(self, text):
        """기본 응답 생성"""
        try:
            self.logger.debug(f"기본 응답 생성 시작: {text[:50]}...")

            emotion_result = self.classify_emotion(text)
            emotion_info = f"감정: {emotion_result['final']['prediction']}, 신뢰도: {emotion_result['final']['confidence']:.2f}"

            chat_history = self.shared_memory.chat_memory.messages
            self.logger.debug(f"대화 히스토리 길이: {len(chat_history)}")

            response = self.chain.invoke(
                {
                    "message": text,
                    "emotion_info": emotion_info,
                    "chat_history": chat_history,
                }
            )

            if hasattr(response, "content"):
                ai_response = response.content
                self.shared_memory.chat_memory.add_user_message(text)
                self.shared_memory.chat_memory.add_ai_message(ai_response)
                self.logger.info("기본 응답 생성 완료")
                return ai_response
            else:
                self.logger.warning("응답에서 content 속성을 찾을 수 없음")
                return "응답 생성에 실패했습니다."

        except Exception as e:
            self.logger.error(f"응답 생성 중 오류: {e}", exc_info=True)
            return "응답 생성 중 오류가 발생했습니다. 다시 시도해주세요."

    def generate_smart_response(self, text):
        """상황에 맞는 모델로 스마트 응답 생성"""
        try:
            self.logger.info(f"스마트 응답 생성 시작: {text[:50]}...")

            selected_model, emotion_result = self._choose_model(text)

            chat_history = self.shared_memory.chat_memory.messages
            llm = self.llms[selected_model]

            emotion_info = f"감정: {emotion_result['final']['prediction']}, 신뢰도: {emotion_result['final']['confidence']:.2f}"

            # 일반대화인경우 다른 프롬프트 사용
            if emotion_result["final"]["prediction"] == "일반대화":
                self.logger.debug("일반 대화 프롬프트 사용")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            self.prompts.get(
                                "general_conversation", "당신은 친근한 AI 친구입니다."
                            ),
                        ),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{message}"),
                    ]
                )
            else:
                self.logger.debug("감정 상담 프롬프트 사용")
                prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", self.system_prompt_text),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{message}"),
                    ]
                )

            chain = prompt | llm
            llm_response = chain.invoke(
                {
                    "message": text,
                    "emotion_info": emotion_info,
                    "chat_history": chat_history,
                }
            )

            response = (
                llm_response.content
                if hasattr(llm_response, "content")
                else str(llm_response)
            )

            # 메모리에 대화 저장
            self.shared_memory.chat_memory.add_user_message(text)
            self.shared_memory.chat_memory.add_ai_message(response)

            self.logger.info(f"스마트 응답 생성 완료 - 사용 모델: {selected_model}")

            return {
                "ai_response": response,
                "model_used": selected_model,
                "emotion_analysis": emotion_result,
            }

        except Exception as e:
            self.logger.error(f"스마트 응답 생성 오류: {e}", exc_info=True)
            fallback_model = list(self.llms.keys())[0]
            return {
                "ai_response": f"죄송합니다. 응답 생성 중 문제가 발생했습니다.",
                "model_used": fallback_model + " (fallback)",
                "error": str(e),
            }

    def process_message(self, text, include_emotion_analysis=True):
        """메시지 전체 처리"""
        try:
            self.logger.info(f"메시지 처리 시작: {text[:50]}...")

            emotion_result = self.classify_emotion(text)

            result = {
                "emotion_result": emotion_result,
            }

            if include_emotion_analysis:
                result["emotion_analysis"] = self.format_analysis(emotion_result)

            ai_response = self.generate_response(text)
            result["ai_response"] = ai_response

            self.logger.info("메시지 처리 완료")
            return result

        except Exception as e:
            self.logger.error(f"메시지 처리 중 오류: {e}", exc_info=True)
            return {"error": str(e)}

    # 유틸리티 메서드들
    def get_conversation_summary(self):
        """현재 대화 요약"""
        try:
            self.logger.debug("대화 요약 생성 시작")
            summary = self.shared_memory.predict_new_summary(
                self.shared_memory.chat_memory.messages, ""
            )
            self.logger.info("대화 요약 생성 완료")
            return summary
        except Exception as e:
            self.logger.error(f"요약 생성 중 오류: {e}", exc_info=True)
            return f"요약 생성 중 오류: {e}"

    def get_chat_history(self):
        """대화 기록 조회"""
        try:
            history = (
                self.shared_memory.chat_memory.messages if self.shared_memory else []
            )
            self.logger.debug(f"대화 기록 조회 - 메시지 수: {len(history)}")
            return history
        except Exception as e:
            self.logger.error(f"대화 기록 조회 오류: {e}")
            return []

    def clear_memory(self):
        """메모리 초기화"""
        try:
            if self.shared_memory:
                self.shared_memory.clear()
                self.logger.info("대화 기록 초기화 완료")
                print("💭 대화 기록이 초기화되었습니다.")
        except Exception as e:
            self.logger.error(f"메모리 초기화 오류: {e}")

    def get_memory_status(self):
        """메모리 상태 확인"""
        try:
            messages = self.shared_memory.chat_memory.messages
            token_count = self.shared_memory.llm.get_num_tokens_from_messages(messages)
            status = {
                "message_count": len(messages),
                "token_count": token_count,
                "max_token_limit": self.shared_memory.max_token_limit,
            }
            self.logger.debug(f"메모리 상태: {status}")
            return status
        except Exception as e:
            self.logger.error(f"메모리 상태 확인 오류: {e}")
            return {"error": str(e)}
