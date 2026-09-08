"""
src/rag_conversational/response_generator.py
============================================
Response Generator: Gọi Gemini API để sinh phản hồi tự nhiên.

Hỗ trợ:
  - Gemini API (google-generativeai) — Mặc định
  - Mock Generator (fallback) — Dùng khi không có API key

Chức năng:
  - generate()          : Gọi LLM với prompt đã build
  - parse_suggested_items(): Trích xuất items được đề xuất từ response
  - estimate_cost()     : Ước tính chi phí token
"""

import re
import time
import logging
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================
# RESPONSE DATA
# ============================================================

@dataclass
class LLMResponse:
    """
    Kết quả từ LLM.

    Attributes:
        text            (str)        : Text phản hồi đầy đủ.
        suggested_items (list[int])  : Item IDs được đề xuất trong response.
        tokens_used     (int)        : Số token đã dùng (nếu API trả về).
        latency_ms      (float)      : Thời gian phản hồi (ms).
        model           (str)        : Tên model đã dùng.
        is_mock         (bool)       : True nếu là mock response.
    """
    text            : str
    suggested_items : list = field(default_factory=list)
    tokens_used     : int  = 0
    latency_ms      : float = 0.0
    model           : str   = ''
    is_mock         : bool  = False

    def __str__(self):
        return self.text


# ============================================================
# GEMINI RESPONSE GENERATOR
# ============================================================

class ResponseGenerator:
    """
    Sinh phản hồi bằng Gemini API.

    Args:
        api_key   (str, optional): Google API key.
                                   Nếu None → lấy từ biến môi trường GOOGLE_API_KEY.
        model_name (str)         : Tên model Gemini.
                                   Mặc định 'gemini-1.5-flash' (nhanh, rẻ).
        temperature (float)      : Độ sáng tạo [0, 1]. Mặc định 0.7.
        max_tokens  (int)        : Số token tối đa trong response. Mặc định 1024.
        retry_count (int)        : Số lần retry khi gặp lỗi. Mặc định 2.
    """

    GEMINI_MODELS = [
        'gemini-1.5-flash',      # Nhanh, rẻ — Khuyến nghị cho demo
        'gemini-1.5-pro',        # Chất lượng cao hơn, chậm hơn
        'gemini-2.0-flash-exp',  # Mới nhất, thử nghiệm
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = 'gemini-flash-latest',
        temperature: float = 0.7,
        max_tokens: int = 1024,
        retry_count: int = 2,
    ):
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.retry_count = retry_count
        self._client     = None   # google.genai Client
        self._model      = None   # google.generativeai legacy model (fallback)
        self._model_name_final = model_name

        # Khởi tạo Gemini client
        self._init_gemini(api_key)

    def _init_gemini(self, api_key: Optional[str]):
        """Khởi tạo Gemini API client."""
        try:
            import google.genai as genai
            import os

            key = api_key or os.environ.get('GOOGLE_API_KEY', '')
            if not key:
                logger.warning(
                    "GOOGLE_API_KEY chưa được set. "
                    "Dùng mock generator. "
                    "Set: $env:GOOGLE_API_KEY='your_key' hoặc truyền api_key='...'."
                )
                return

            self._client = genai.Client(api_key=key)
            self._model_name_final = self.model_name
            logger.info(f"Gemini (google.genai) initialized: {self.model_name}")

        except ImportError:
            # Fallback sang google.generativeai nếu google.genai chưa có
            try:
                import google.generativeai as genai_legacy
                import os
                import warnings
                warnings.filterwarnings('ignore', category=FutureWarning)

                key = api_key or os.environ.get('GOOGLE_API_KEY', '')
                if not key:
                    logger.warning("GOOGLE_API_KEY chưa được set. Dùng mock generator.")
                    return

                genai_legacy.configure(api_key=key)
                self._model = genai_legacy.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        'temperature': self.temperature,
                        'max_output_tokens': self.max_tokens,
                    },
                )
                logger.info(f"Gemini (legacy API) initialized: {self.model_name}")
            except ImportError:
                logger.warning(
                    "google-genai chưa cài. "
                    "Chạy: pip install google-genai\n"
                    "Đang dùng mock generator."
                )
            except Exception as e:
                logger.error(f"Lỗi khởi tạo Gemini (legacy): {e}. Dùng mock generator.")
        except Exception as e:
            logger.error(f"Lỗi khởi tạo Gemini: {e}. Dùng mock generator.")


    # ----------------------------------------------------------
    # GENERATE
    # ----------------------------------------------------------

    def generate(self, prompt: str) -> LLMResponse:
        """
        Sinh phản hồi từ prompt.

        Args:
            prompt (str): Prompt đã build từ PromptBuilder.

        Returns:
            LLMResponse: Kết quả phản hồi.
        """
        start_ms = time.time() * 1000

        # Không có client nào khả dụng
        if self._client is None and self._model is None:
            return self._mock_response(prompt, start_ms)

        candidate_models = [self._model_name_final, 'gemini-flash-latest', 'gemini-2.5-flash', 'gemini-3.6-flash', 'gemini-1.5-flash']
        # Unique preserve order
        candidate_models = list(dict.fromkeys(candidate_models))

        last_error = None
        for model_cand in candidate_models:
            for attempt in range(self.retry_count + 1):
                try:
                    if self._client is not None:
                        # google.genai (mới)
                        response = self._client.models.generate_content(
                            model=model_cand,
                            contents=prompt,
                        )
                        text = response.text
                        tokens = getattr(getattr(response, 'usage_metadata', None), 'total_token_count', 0) or 0
                    else:
                        # google.generativeai (legacy fallback)
                        result = self._model.generate_content(prompt)
                        text = result.text
                        tokens = getattr(getattr(result, 'usage_metadata', None), 'total_token_count', 0) or 0

                    latency = time.time() * 1000 - start_ms
                    items   = self.parse_suggested_items(text)

                    logger.debug(
                        f"Gemini response: {len(text)} chars, "
                        f"{tokens} tokens, {latency:.0f}ms"
                    )

                    return LLMResponse(
                        text            = text,
                        suggested_items = items,
                        tokens_used     = tokens,
                        latency_ms      = latency,
                        model           = model_cand,
                        is_mock         = False,
                    )

                except Exception as e:
                    last_error = str(e)
                    # Nếu lỗi 404 (model không hỗ trợ), break để thử model khác
                    if '404' in last_error or 'NOT_FOUND' in last_error:
                        logger.warning(f"Model {model_cand} not found (404). Thử model tiếp theo...")
                        break
                    # Nếu lỗi 429 (hết quota / credits), không cần thử lại
                    if '429' in last_error or 'RESOURCE_EXHAUSTED' in last_error:
                        logger.warning(f"API Key hết quota (429 RESOURCE_EXHAUSTED). Dùng Mock mode.")
                        return self._mock_response(prompt, start_ms, error="Tài khoản Google API Key đã hết lượt dùng miễn phí (Quota/Credits)")
                    
                    if attempt < self.retry_count:
                        wait = 2 ** attempt
                        logger.warning(f"Gemini error (attempt {attempt+1}): {e}. Retry in {wait}s...")
                        time.sleep(wait)
                    else:
                        break

        logger.error(f"Gemini failed all candidate models: {last_error}")
        return self._mock_response(prompt, start_ms, error=last_error)


    def _mock_response(
        self,
        prompt: str,
        start_ms: float,
        error: Optional[str] = None,
    ) -> LLMResponse:
        """Sinh response giả khi không có API key hoặc lỗi."""
        latency = time.time() * 1000 - start_ms

        # Trích xuất tên items từ prompt để tạo response tự nhiên
        items_in_prompt = re.findall(r'•\s([^\[]+)\s\[Score', prompt)
        if not items_in_prompt:
            # Format khác: [ID] Tên (Genres)
            items_in_prompt = re.findall(r'\[\d+\]\s([^(\n\r]+)', prompt)

        items_str = ', '.join(f'"{i.strip()}"' for i in items_in_prompt[:3]) if items_in_prompt else ""

        notice = ""
        if error:
            if "RESOURCE_EXHAUSTED" in error or "hết lượt dùng" in error:
                notice = "⚠️ *(Lưu ý: API Key của bạn đã hết hạn mức Google AI Studio, hệ thống đang dùng bộ sinh phản hồi nội bộ)*\n\n"
            else:
                notice = f"⚠️ *(Lưu ý kết nối AI: {error[:80]}... - Chuyển sang phản hồi nội bộ)*\n\n"

        if items_str:
            text = (
                f"{notice}Dựa trên sở thích của bạn và dữ liệu phân tích từ hệ thống, tôi đặc biệt gợi ý các lựa chọn hàng đầu sau:\n\n"
                f"{items_str}\n\n"
                f"Các gợi ý này được tối ưu dựa trên tương quan đánh giá và ngữ cảnh tương tác của bạn. Bạn muốn tìm hiểu thêm thông tin chi tiết về lựa chọn nào trên đây không?"
            )
        else:
            text = (
                f"{notice}Tôi đã tiếp nhận câu hỏi của bạn và tổng hợp các gợi ý phù hợp nhất từ mô hình đề xuất. Bạn có thể xem danh sách chi tiết ở bảng bên dưới!"
            )

        return LLMResponse(
            text            = text,
            suggested_items = [],
            tokens_used     = 0,
            latency_ms      = latency,
            model           = 'mock',
            is_mock         = True,
        )

    # ----------------------------------------------------------
    # PARSE SUGGESTED ITEMS
    # ----------------------------------------------------------

    def parse_suggested_items(self, response_text: str) -> list[int]:
        """
        Trích xuất item IDs từ response text của LLM.

        Tìm kiếm các pattern như:
          - "(ID: 123)"
          - "item_id: 456"
          - Số xuất hiện trong context

        Args:
            response_text (str): Text response từ LLM.

        Returns:
            list[int]: Danh sách item IDs được đề xuất.
        """
        ids = []

        # Pattern 1: "(ID: 123)" hoặc "[ID: 123]"
        matches = re.findall(r'[(\[]\s*[Ii][Dd]\s*:\s*(\d+)\s*[)\]]', response_text)
        ids.extend(int(m) for m in matches)

        # Pattern 2: "item_id: 123" hoặc "itemId: 123"
        matches = re.findall(r'item[_\s]?[Ii][Dd]\s*:\s*(\d+)', response_text)
        ids.extend(int(m) for m in matches)

        # Dedup và giữ thứ tự
        seen = set()
        result = []
        for item_id in ids:
            if item_id not in seen:
                seen.add(item_id)
                result.append(item_id)

        return result

    def estimate_cost_usd(self, prompt: str, response_text: str = '') -> float:
        """
        Ước tính chi phí API (USD).

        Gemini 1.5 Flash pricing (tháng 9/2026):
          - Input:  $0.075 per 1M tokens
          - Output: $0.30  per 1M tokens
        """
        # 1 token ≈ 4 chars (EN) / 2 chars (VI)
        input_tokens  = len(prompt) // 3
        output_tokens = len(response_text) // 3

        if 'flash' in self.model_name:
            cost = input_tokens * 0.075e-6 + output_tokens * 0.30e-6
        else:   # pro
            cost = input_tokens * 3.5e-6 + output_tokens * 10.5e-6

        return cost

    def __repr__(self):
        if self._client is not None:
            backend = 'google.genai'
        elif self._model is not None:
            backend = 'legacy'
        else:
            backend = 'mock'
        return (
            f"ResponseGenerator("
            f"model='{self.model_name}', "
            f"backend={backend}, "
            f"temp={self.temperature})"
        )

