# emotion_classifier.py
import torch
import pickle
import copy
import os
import re
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, logging as transformers_logging
from soynlp.normalizer import repeat_normalize


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


def _get_default_device():
    """torch.get_default_device() 대체 함수"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# torch에 없는 함수 추가
torch.get_default_device = _get_default_device

# 한국어 불용어 리스트
KOREAN_STOPWORDS = [
    "이",
    "그",
    "저",
    "것",
    "및",
    "에",
    "를",
    "은",
    "는",
    "이런",
    "저런",
    "그런",
    "한",
    "이르",
    "또한",
    "있",
    "하",
    "에서",
    "으로",
    "으로써",
    "로써",
    "로서",
    "로",
    "와",
    "과",
    "이고",
    "이며",
    "이다",
    "있다",
    "하다",
    "되다",
    "이",
    "가",
    "을",
    "를",
    "에게",
    "의",
    "뿐",
    "다",
    "적",
    "데",
    "때",
    "나",
    "도",
    "만",
    "께",
    "에게서",
]

# 전처리 로거
preprocess_logger = setup_logger("TextPreprocessor", "WARNING")


def preprocess_korean_text(text):
    """한국어 텍스트 전처리"""
    try:
        if pd.isna(text) or text is None or len(str(text).strip()) == 0:
            preprocess_logger.debug("빈 텍스트 입력")
            return ""

        original_text = str(text)
        preprocess_logger.debug(f"전처리 시작: {original_text[:50]}...")

        text = str(text)
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)  # 반복 문자 정규화
        text = re.sub(r"[^가-힣a-zA-Z0-9\s\.,!?]", " ", text)  # 특수문자 제거
        text = re.sub(r"\s+", " ", text).strip()  # 공백 정리

        preprocess_logger.debug(f"전처리 완료: {text[:50]}...")
        return text

    except Exception as e:
        preprocess_logger.error(f"텍스트 전처리 오류: {e}")
        return ""


def remove_stopwords(text, stopwords=KOREAN_STOPWORDS):
    """주어진 텍스트에서 불용어 제거"""
    try:
        if pd.isna(text) or text is None or len(str(text).strip()) == 0:
            return ""

        words = text.split()
        original_count = len(words)
        filtered_words = [word for word in words if word not in stopwords]

        preprocess_logger.debug(
            f"불용어 제거: {original_count}개 -> {len(filtered_words)}개 단어"
        )
        return " ".join(filtered_words)

    except Exception as e:
        preprocess_logger.error(f"불용어 제거 오류: {e}")
        return str(text) if text else ""


class EmotionClassifier(torch.nn.Module):
    """감정 분류기 모델 - BERT 기반의 감정 분류를 위한 신경망 모델"""

    def __init__(self, bert_model, num_classes, dropout_rate=0.3):
        super(EmotionClassifier, self).__init__()

        self.bert = bert_model
        self.hidden_size = self.bert.config.hidden_size
        self.dropout1 = torch.nn.Dropout(dropout_rate)

        # 어텐션 메커니즘
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1),
            torch.nn.Softmax(dim=1),
        )

        # 분류기
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, 256),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )

        sequence_output = outputs.last_hidden_state
        pooled_output = outputs.pooler_output

        attention_weights = self.attention(sequence_output)
        context_vector = torch.sum(attention_weights * sequence_output, dim=1)
        final_output = context_vector + pooled_output
        final_output = self.dropout1(final_output)
        logits = self.classifier(final_output)

        return logits


class HierarchicalEmotionClassifier:
    def __init__(
        self,
        model_dir,
        confidence_threshold=0.6,
        emotion_threshold=0.3,
        log_level="INFO",
    ):
        # 로거 초기화
        self.logger = setup_logger("HierarchicalEmotionClassifier", log_level)
        self.logger.info("HierarchicalEmotionClassifier 초기화 시작")

        self.model_dir = model_dir
        self.device = _get_default_device()
        self.max_len = 128
        self.confidence_threshold = confidence_threshold
        self.emotion_threshold = emotion_threshold

        self.logger.info(f"설정값 - 모델 경로: {model_dir}, 디바이스: {self.device}")
        self.logger.info(
            f"임계값 - 신뢰도: {confidence_threshold}, 감정: {emotion_threshold}"
        )

        self.tokenizer = None
        self.bert_model = None
        self.level1_model = None
        self.level2_model = None
        self.level3_model = None
        self.level1_encoder = None
        self.level2_encoder = None
        self.level3_encoder = None

        try:
            self._load_models()
            self.logger.info("HierarchicalEmotionClassifier 초기화 완료")
        except Exception as e:
            self.logger.error(
                f"HierarchicalEmotionClassifier 초기화 실패: {e}", exc_info=True
            )
            raise

    def _check_files(self):
        """필수 파일들 존재 확인"""
        self.logger.debug("필수 파일 존재 확인 시작")

        required_files = [
            "level1_best_model.pt",
            "level2_best_model.pt",
            "level3_best_model.pt",
            "level1_label_encoder.pkl",
            "level2_label_encoder.pkl",
            "level3_label_encoder.pkl",
        ]

        missing_files = []
        for file in required_files:
            filepath = os.path.join(self.model_dir, file)
            if not os.path.exists(filepath):
                missing_files.append(file)
                self.logger.error(f"필수 파일 누락: {filepath}")
            else:
                self.logger.debug(f"파일 확인됨: {filepath}")

        if missing_files:
            error_msg = f"파일 누락 목록: {missing_files}"
            self.logger.error(error_msg)
            raise FileExistsError(error_msg)

        self.logger.info("모든 필수 파일 확인 완료")

    def _load_label_encoders(self):
        """라벨 인코더 로드"""
        self.logger.info("라벨 인코더 로드 시작")

        encoders = [
            ("level1_label_encoder.pkl", "level1_encoder"),
            ("level2_label_encoder.pkl", "level2_encoder"),
            ("level3_label_encoder.pkl", "level3_encoder"),
        ]

        for filename, attr_name in encoders:
            try:
                filepath = os.path.join(self.model_dir, filename)
                self.logger.debug(f"라벨 인코더 로드 시도: {filepath}")

                with open(filepath, "rb") as f:
                    encoder = pickle.load(f)
                    setattr(self, attr_name, encoder)

                self.logger.info(
                    f"{attr_name} 로드 완료 - 클래스 수: {len(encoder.classes_)}"
                )
                self.logger.debug(f"{attr_name} 클래스: {list(encoder.classes_)}")

            except Exception as e:
                self.logger.error(
                    f"라벨 인코더 로드 실패 ({filename}): {e}", exc_info=True
                )
                raise

    def _load_bert_model(self):
        """BERT 모델과 토크나이저 로드"""
        self.logger.info("BERT 모델 로드 시작")

        try:
            # transformers 로깅 레벨 조정
            transformers_logging.set_verbosity_error()

            self.logger.debug("BERT 토크나이저 로드 시작")
            self.tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
            self.logger.info("BERT 토크나이저 로드 완료")

            self.logger.debug("BERT 모델 로드 시작")
            self.bert_model = AutoModel.from_pretrained("klue/bert-base")
            self.bert_model = self.bert_model.to(self.device)
            self.logger.info(f"BERT 모델 로드 완료 - 디바이스: {self.device}")

        except Exception as e:
            self.logger.error(f"BERT 모델 로드 실패: {e}", exc_info=True)
            raise ImportError(f"BERT 모델 로드 실패: {e}")

    def _load_emotion_models(self):
        """감정 분류 모델들 로드"""
        self.logger.info("감정 분류 모델 로드 시작")

        models = [
            ("level1_best_model.pt", "level1_model", self.level1_encoder),
            ("level2_best_model.pt", "level2_model", self.level2_encoder),
            ("level3_best_model.pt", "level3_model", self.level3_encoder),
        ]

        for filename, attr_name, encoder in models:
            try:
                filepath = os.path.join(self.model_dir, filename)
                self.logger.debug(f"감정 모델 로드 시도: {filepath}")

                model = EmotionClassifier(
                    copy.deepcopy(self.bert_model.to("cpu")), len(encoder.classes_)
                )

                try:
                    state_dict = torch.load(
                        filepath, map_location="cpu", weights_only=True
                    )
                except TypeError:
                    # 이전 버전 호환성
                    state_dict = torch.load(filepath, map_location="cpu")

                model.load_state_dict(state_dict)
                model = model.to(self.device)
                model.eval()

                setattr(self, attr_name, model)
                self.logger.info(
                    f"{attr_name} 로드 완료 - 클래스 수: {len(encoder.classes_)}"
                )

            except Exception as e:
                self.logger.error(
                    f"감정 모델 로드 실패 ({filename}): {e}", exc_info=True
                )
                raise

    def _load_models(self):
        """모든 모델 로드"""
        self.logger.info("전체 모델 로드 프로세스 시작")

        try:
            self._check_files()
            self._load_label_encoders()
            self._load_bert_model()
            self._load_emotion_models()
            self.logger.info("전체 모델 로드 완료")
        except Exception as e:
            self.logger.error(f"모델 로드 프로세스 실패: {e}", exc_info=True)
            raise

    def _predict_single(self, model, text):
        """단일 모델 예측"""
        try:
            self.logger.debug(f"단일 모델 예측 시작: {text[:30]}...")

            preprocessed = preprocess_korean_text(text)
            preprocessed = remove_stopwords(preprocessed)

            self.logger.debug(f"전처리된 텍스트: {preprocessed[:50]}...")

            encoding = self.tokenizer.encode_plus(
                preprocessed,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=True,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            token_type_ids = encoding["token_type_ids"].to(self.device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask, token_type_ids)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)

            pred_idx = preds.item()
            confidence = float(probs[0][pred_idx])

            self.logger.debug(
                f"예측 결과 - 인덱스: {pred_idx}, 신뢰도: {confidence:.4f}"
            )

            return pred_idx, probs[0]

        except Exception as e:
            self.logger.error(f"단일 모델 예측 오류: {e}", exc_info=True)
            raise

    def predict_hierarchical(self, text):
        """계층적 감정 예측"""
        self.logger.info(f"계층적 감정 예측 시작: {text[:50]}...")

        try:
            result = {
                "original_text": text,
                "preprocessed_text": remove_stopwords(preprocess_korean_text(text)),
                "levels": {},
            }

            # 1단계: 일반대화 vs 감정
            self.logger.debug("1단계 예측 시작: 일반대화 vs 감정")
            pred1, probs1 = self._predict_single(self.level1_model, text)
            label1 = self.level1_encoder.inverse_transform([pred1])[0]
            confidence1 = float(probs1[pred1])

            result["levels"]["level1"] = {
                "step": "1단계: 일반대화 vs 감정",
                "prediction": label1,
                "confidence": confidence1,
                "probabilities": {
                    self.level1_encoder.classes_[i]: float(probs1[i])
                    for i in range(len(self.level1_encoder.classes_))
                },
            }

            self.logger.info(f"1단계 결과: {label1} (신뢰도: {confidence1:.4f})")

            if label1 == "일반대화":
                result["final"] = {
                    "prediction": "일반대화",
                    "confidence": confidence1,
                }
                result["path"] = ["일반대화"]
                self.logger.info("예측 완료: 일반대화로 분류")
                return result

            # 2단계: 기쁨 vs 기타감정
            self.logger.debug("2단계 예측 시작: 기쁨 vs 기타감정")
            pred2, probs2 = self._predict_single(self.level2_model, text)
            label2 = self.level2_encoder.inverse_transform([pred2])[0]
            confidence2 = float(probs2[pred2])

            result["levels"]["level2"] = {
                "step": "2단계: 기쁨 vs 기타감정",
                "prediction": label2,
                "confidence": confidence2,
                "probabilities": {
                    self.level2_encoder.classes_[i]: float(probs2[i])
                    for i in range(len(self.level2_encoder.classes_))
                },
            }

            self.logger.info(f"2단계 결과: {label2} (신뢰도: {confidence2:.4f})")

            if label2 == "기쁨":
                result["final"] = {"prediction": "기쁨", "confidence": confidence2}
                result["path"] = ["감정", "기쁨"]
                self.logger.info("예측 완료: 기쁨으로 분류")
                return result

            # 3단계: 세부 감정 분류
            self.logger.debug("3단계 예측 시작: 세부 감정 분류")
            pred3, probs3 = self._predict_single(self.level3_model, text)
            label3 = self.level3_encoder.inverse_transform([pred3])[0]
            confidence3 = float(probs3[pred3])

            result["levels"]["level3"] = {
                "step": "3단계: 세부 감정 분류",
                "prediction": label3,
                "confidence": confidence3,
                "probabilities": {
                    self.level3_encoder.classes_[i]: float(probs3[i])
                    for i in range(len(self.level3_encoder.classes_))
                },
            }

            result["final"] = {"prediction": label3, "confidence": confidence3}
            result["path"] = ["감정", "기타감정", label3]

            self.logger.info(f"예측 완료: {label3} (신뢰도: {confidence3:.4f})")
            self.logger.info(f"전체 경로: {' → '.join(result['path'])}")

            return result

        except Exception as e:
            self.logger.error(f"계층적 예측 오류: {e}", exc_info=True)
            # 오류 발생 시 기본값 반환
            return {
                "original_text": text,
                "error": str(e),
                "final": {"prediction": "오류", "confidence": 0.0},
                "path": ["오류"],
            }
