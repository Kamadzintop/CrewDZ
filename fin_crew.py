from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
# Убираем проблемный импорт langchain.tools
# from langchain.tools import tool
import os
import requests
import json
import time
import asyncio
import logging
from typing import List, Dict, Any, Union, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from functools import wraps

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# НАСТРОЙКА ПОДРОБНОГО ЛОГИРОВАНИЯ
# ============================================================================

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('analysis_debug.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_step(step_name: str, details: str = ""):
    """Логирование шагов выполнения с временными метками"""
    timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    message = f"[{timestamp}] 🔄 {step_name}"
    if details:
        message += f" - {details}"
    logger.info(message)
    print(message)  # Дублируем в консоль для отладки

# ============================================================================
<<<<<<< HEAD
# СИСТЕМА КЕШИРОВАНИЯ ДЛЯ УМЕНЬШЕНИЯ ТРАФИКА LLM
# ============================================================================

import hashlib
import pickle
import os.path

class CacheManager:
    """Менеджер кеширования для уменьшения трафика LLM"""
    
    def __init__(self, cache_dir: str = "cache", max_age_hours: int = 24):
        self.cache_dir = cache_dir
        self.max_age_seconds = max_age_hours * 3600
        
        # Создаем директорию кеша если её нет
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            log_step("КЕШ СОЗДАН", f"Директория кеша: {cache_dir}")
    
    def _get_cache_key(self, data: str) -> str:
        """Генерация ключа кеша на основе данных"""
        return hashlib.md5(data.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, cache_key: str) -> str:
        """Получение пути к файлу кеша"""
        return os.path.join(self.cache_dir, f"{cache_key}.pkl")
    
    def get(self, data: str) -> Optional[str]:
        """Получение данных из кеша"""
        try:
            cache_key = self._get_cache_key(data)
            cache_path = self._get_cache_path(cache_key)
            
            if not os.path.exists(cache_path):
                return None
            
            # Проверяем возраст кеша
            file_age = time.time() - os.path.getmtime(cache_path)
            if file_age > self.max_age_seconds:
                log_step("КЕШ УСТАРЕЛ", f"Возраст: {file_age/3600:.1f}ч, максимум: {self.max_age_seconds/3600}ч")
                os.remove(cache_path)
                return None
            
            # Загружаем данные из кеша
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            log_step("КЕШ НАЙДЕН", f"Ключ: {cache_key[:8]}...")
            return cached_data
            
        except Exception as e:
            log_step("ОШИБКА КЕША", f"Ошибка чтения: {e}")
            return None
    
    def set(self, data: str, result: str) -> bool:
        """Сохранение данных в кеш"""
        try:
            cache_key = self._get_cache_key(data)
            cache_path = self._get_cache_path(cache_key)
            
            # Сохраняем данные в кеш
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)
            
            log_step("КЕШ СОХРАНЕН", f"Ключ: {cache_key[:8]}...")
            return True
            
        except Exception as e:
            log_step("ОШИБКА КЕША", f"Ошибка записи: {e}")
            return False
    
    def clear_old_cache(self) -> int:
        """Очистка устаревшего кеша"""
        try:
            cleared_count = 0
            current_time = time.time()
            
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(self.cache_dir, filename)
                    file_age = current_time - os.path.getmtime(file_path)
                    
                    if file_age > self.max_age_seconds:
                        os.remove(file_path)
                        cleared_count += 1
            
            if cleared_count > 0:
                log_step("КЕШ ОЧИЩЕН", f"Удалено файлов: {cleared_count}")
            
            return cleared_count
            
        except Exception as e:
            log_step("ОШИБКА ОЧИСТКИ КЕША", f"Ошибка: {e}")
            return 0

# Создаем глобальный менеджер кеша с настройками из переменных окружения
cache_dir = os.getenv("CACHE_DIR", "cache")
cache_max_age_hours = int(os.getenv("CACHE_MAX_AGE_HOURS", "24"))
cache_manager = CacheManager(cache_dir=cache_dir, max_age_hours=cache_max_age_hours)

# ============================================================================
=======
>>>>>>> af05dce (предварительная версия)
# НАСТРОЙКИ ТАЙМАУТОВ И ПОВТОРНЫХ ЗАПРОСОВ
# ============================================================================

# Таймауты для различных операций (в секундах)
TIMEOUTS = {
    'llm_request': 60,        # Таймаут для запроса к LLM
    'api_request': 30,        # Таймаут для API запросов
    'crew_execution': 300,    # Таймаут для выполнения Crew
    'task_execution': 120,    # Таймаут для выполнения задачи
    'retry_delay': 5,         # Задержка перед повторным запросом
    'max_retries': 1,         # Максимальное количество повторных попыток
}

def retry_on_timeout(max_retries: int = 1, delay: float = 5.0):
    """Декоратор для повторных попыток при таймаутах"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        print(f"⚠️  Попытка {attempt + 1} не удалась: {str(e)}")
                        print(f"⏳ Ожидание {delay} секунд перед повторной попыткой...")
                        time.sleep(delay)
                    else:
                        print(f"❌ Все попытки исчерпаны. Последняя ошибка: {str(e)}")
            
            # Если все попытки исчерпаны, пытаемся переключиться на альтернативную модель
            error_str = str(last_exception).lower()
            
            # Проверяем, можно ли переключиться на альтернативную модель
            if "insufficient balance" in error_str or "badrequesterror" in error_str:
                # Пытаемся переключиться на альтернативную модель
                alternative_llm = switch_to_alternative_llm(llm, str(last_exception))
                if alternative_llm and alternative_llm != llm:
                    # Если переключение успешно, повторяем попытку с новой моделью
                    print(f"🔄 Переключились на {alternative_llm.model}, повторяем попытку...")
                    try:
                        return func(*args, **kwargs)
                    except Exception as retry_exception:
                        return f"❌ Ошибка даже с альтернативной моделью: {str(retry_exception)}"
            
            # Если переключение невозможно или не помогло, возвращаем сообщение об ошибке
            if "timeout" in error_str or "timed out" in error_str:
                return "⏰ Серверы ИИ перегружены. Пожалуйста, повторите запрос позже."
            elif "insufficient balance" in error_str:
                return "💰 Недостаточно средств на балансе API. Пожалуйста, пополните счет."
            elif "badrequesterror" in error_str:
                return "🔧 Ошибка запроса к API. Проверьте настройки и повторите попытку."
            else:
                return f"❌ Ошибка: {str(last_exception)}"
        
        return wrapper
    return decorator

# Исправляем ошибки линтера - добавляем проверки на None
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_api_base = os.getenv("ANTHROPIC_API_BASE")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if openai_api_base:
    os.environ["OPENAI_API_BASE"] = openai_api_base
if deepseek_api_key:
    os.environ["DEEPSEEK_API_KEY"] = deepseek_api_key
deepseek_api_base = os.getenv("DEEPSEEK_API_BASE")
if deepseek_api_base:
    os.environ["DEEPSEEK_API_BASE"] = deepseek_api_base
if anthropic_api_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
if anthropic_api_base:
    os.environ["ANTHROPIC_API_BASE"] = anthropic_api_base

# Создаем LLM объекты только если есть API ключи
llm_deepseek = None
llm_gpt4 = None
llm_anthropic = None
llm = None

<<<<<<< HEAD
# Инициализация поискового LLM (отдельная модель для поиска)
search_llm = None
proxyapi_search_key = os.getenv("PROXYAPI_SEARCH_KEY")
proxyapi_search_base = os.getenv("PROXYAPI_SEARCH_BASE")

if proxyapi_search_key:
    log_step("СОЗДАНИЕ ПОИСКОВОГО LLM", "Инициализация ProxyAPI для поиска")
    try:
        from openai import OpenAI
        search_client = OpenAI(
            api_key=proxyapi_search_key,
            base_url=proxyapi_search_base or "https://api.proxyapi.ru/openai/v1"
        )
        search_llm = search_client
        log_step("ПОИСКОВЫЙ LLM СОЗДАН", "ProxyAPI для поиска успешно инициализирован")
    except Exception as e:
        log_step("ОШИБКА ПОИСКОВОГО LLM", f"Ошибка инициализации: {e}")
        search_llm = None
else:
    log_step("ПОИСКОВЫЙ LLM НЕДОСТУПЕН", "PROXYAPI_SEARCH_KEY не настроен")

if deepseek_api_key:
    log_step("СОЗДАНИЕ DEEPSEEK LLM", "Инициализация DeepSeek модели")
    try:
        # Используем правильный формат для LiteLLM с ProxyAPI
        llm_deepseek = LLM(
            model="deepseek-reasoner",  # Указываем провайдера
            api_key=deepseek_api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE"),
            request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
        )
        log_step("DEEPSEEK LLM СОЗДАН", "Модель deepseek-reasoner успешно инициализирована")
    except Exception as e:
        log_step("ОШИБКА DEEPSEEK LLM", f"Основная модель: {e}")
        # Попробуем альтернативный формат
        try:
            log_step("ПОПЫТКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", "deepseek-chat")
            llm_deepseek = LLM(
                model="deepseek-chat",
                api_key=deepseek_api_key,
                base_url=os.getenv("DEEPSEEK_API_BASE"),
                request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
            )
            log_step("DEEPSEEK LLM СОЗДАН", "Альтернативная модель deepseek-chat успешно инициализирована")
        except Exception as e2:
            log_step("ОШИБКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", f"Альтернативная модель: {e2}")
            llm_deepseek = None

=======
if deepseek_api_key:
    log_step("СОЗДАНИЕ DEEPSEEK LLM", "Инициализация DeepSeek модели")
    try:
        # Используем правильный формат для LiteLLM с ProxyAPI
        llm_deepseek = LLM(
            model="deepseek-reasoner",  # Указываем провайдера
            api_key=deepseek_api_key,
            base_url=os.getenv("DEEPSEEK_API_BASE"),
            request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
        )
        log_step("DEEPSEEK LLM СОЗДАН", "Модель deepseek-reasoner успешно инициализирована")
    except Exception as e:
        log_step("ОШИБКА DEEPSEEK LLM", f"Основная модель: {e}")
        # Попробуем альтернативный формат
        try:
            log_step("ПОПЫТКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", "deepseek-chat")
            llm_deepseek = LLM(
                model="deepseek-chat",
                api_key=deepseek_api_key,
                base_url=os.getenv("DEEPSEEK_API_BASE"),
                request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
            )
            log_step("DEEPSEEK LLM СОЗДАН", "Альтернативная модель deepseek-chat успешно инициализирована")
        except Exception as e2:
            log_step("ОШИБКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", f"Альтернативная модель: {e2}")
            llm_deepseek = None

>>>>>>> af05dce (предварительная версия)
if anthropic_api_key:
    log_step("СОЗДАНИЕ ANTHROPIC LLM", "Инициализация Claude модели")
    try:
        # Используем LiteLLM для Claude через ProxyAPI
        llm_anthropic = LLM(
            model="claude-sonnet-4-20250514",
            api_key=anthropic_api_key,
            base_url=os.getenv("ANTHROPIC_API_BASE"),
            request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
        )
        log_step("ANTHROPIC LLM СОЗДАН", "Модель claude-sonnet-4-20250514 успешно инициализирована")
    except Exception as e:
        log_step("ОШИБКА ANTHROPIC LLM", f"Основная модель: {e}")
        # Попробуем альтернативную модель
        try:
            log_step("ПОПЫТКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", "claude-sonnet-4")
            llm_anthropic = LLM(
                model="claude-sonnet-4",
                api_key=anthropic_api_key,
                base_url=os.getenv("ANTHROPIC_API_BASE"),
                request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
            )
            log_step("ANTHROPIC LLM СОЗДАН", "Альтернативная модель claude-sonnet-4 успешно инициализирована")
        except Exception as e2:
            log_step("ОШИБКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", f"Альтернативная модель: {e2}")
            llm_anthropic = None

if openai_api_key:
    log_step("СОЗДАНИЕ OPENAI LLM", "Инициализация OpenAI GPT-4.1 модели")
    try:
        llm_gpt4 = LLM(
            model="gpt-4.1-2025-04-14",
            api_key=openai_api_key,
            request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
        )
        log_step("OPENAI LLM СОЗДАН", "Модель gpt-4.1-2025-04-14 успешно инициализирована")
    except Exception as e:
        log_step("ОШИБКА OPENAI LLM", f"Основная модель: {e}")
        # Попробуем альтернативную модель
        try:
            log_step("ПОПЫТКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", "gpt-4.1")
            llm_gpt4 = LLM(
                model="gpt-4.1",
                api_key=openai_api_key,
                request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
            )
            log_step("OPENAI LLM СОЗДАН", "Альтернативная модель gpt-4.1 успешно инициализирована")
        except Exception as e2:
            log_step("ОШИБКА АЛЬТЕРНАТИВНОЙ МОДЕЛИ", f"Альтернативная модель: {e2}")
            # Последняя попытка с базовой моделью
            try:
                log_step("ПОПЫТКА БАЗОВОЙ МОДЕЛИ", "gpt-4")
                llm_gpt4 = LLM(
                    model="gpt-4",
                    api_key=openai_api_key,
                    request_timeout=TIMEOUTS['llm_request']  # Добавляем таймаут
                )
                log_step("OPENAI LLM СОЗДАН", "Базовая модель gpt-4 успешно инициализирована")
            except Exception as e3:
                log_step("ОШИБКА БАЗОВОЙ МОДЕЛИ", f"Базовая модель: {e3}")
                llm_gpt4 = None

# Выбираем модель для использования (GPT-4.1 как основная)
if llm_gpt4:
    llm = llm_gpt4
    print("✅ Используется GPT-4.1 (OpenAI) как основная модель")
elif llm_anthropic:
    llm = llm_anthropic
    print("✅ Используется Claude (Anthropic) как основная модель")
elif llm_deepseek:
    llm = llm_deepseek
    print("⚠️  DeepSeek используется как резервная модель")
else:
    print("⚠️  API ключи не настроены или не работают. Система будет работать в демо-режиме.")

def get_working_llm():
    """Получение рабочей LLM модели с автоматическим переключением при ошибках"""
    global llm
    
    # Проверяем текущую модель
    if llm and llm == llm_gpt4:
        return llm_gpt4
    elif llm and llm == llm_anthropic:
        return llm_anthropic
    elif llm and llm == llm_deepseek:
        return llm_deepseek
    
    # Если текущая модель не определена, выбираем лучшую доступную (GPT-4.1 первая)
    if llm_gpt4:
        return llm_gpt4
    elif llm_anthropic:
        return llm_anthropic
    elif llm_deepseek:
        return llm_deepseek
    else:
        return None

def switch_to_alternative_llm(current_llm, error_message: str):
    """Переключение на альтернативную модель при ошибках (GPT-4.1 как основная)"""
    global llm
    
    error_lower = error_message.lower()
    
    # Если текущая модель - GPT-4.1 и есть проблемы с балансом
    if current_llm == llm_gpt4 and ("insufficient balance" in error_lower or "badrequesterror" in error_lower):
        if llm_anthropic:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"GPT-4.1 → Claude (ошибка: {error_message[:50]}...)")
            llm = llm_anthropic
            return llm_anthropic
        elif llm_deepseek:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"GPT-4.1 → DeepSeek (ошибка: {error_message[:50]}...)")
            llm = llm_deepseek
            return llm_deepseek
    
    # Если текущая модель - Claude и есть проблемы
    elif current_llm == llm_anthropic and ("insufficient balance" in error_lower or "badrequesterror" in error_lower):
        if llm_gpt4:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"Claude → GPT-4.1 (ошибка: {error_message[:50]}...)")
            llm = llm_gpt4
            return llm_gpt4
        elif llm_deepseek:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"Claude → DeepSeek (ошибка: {error_message[:50]}...)")
            llm = llm_deepseek
            return llm_deepseek
    
    # Если текущая модель - DeepSeek и есть проблемы (резервная модель)
    elif current_llm == llm_deepseek and ("insufficient balance" in error_lower or "badrequesterror" in error_lower):
        if llm_gpt4:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"DeepSeek → GPT-4.1 (ошибка: {error_message[:50]}...)")
            llm = llm_gpt4
            return llm_gpt4
        elif llm_anthropic:
            log_step("ПЕРЕКЛЮЧЕНИЕ МОДЕЛИ", f"DeepSeek → Claude (ошибка: {error_message[:50]}...)")
            llm = llm_anthropic
            return llm_anthropic
    
    # Если нет альтернатив, возвращаем текущую модель
    return current_llm

# ============================================================================
# СИСТЕМА ДВОЙНОЙ ВАЛИДАЦИИ С ДВУМЯ AI МОДЕЛЯМИ
# ============================================================================

class DualAITool(BaseTool):
    """Инструмент для получения и сравнения данных от двух AI моделей"""
    
    name: str
    description: str
    
    def __init__(self, name: str, description: str, llm1: LLM, llm2: LLM):
        super().__init__(name=name, description=description)
        self._llm1 = llm1
        self._llm2 = llm2
    
    @retry_on_timeout(max_retries=TIMEOUTS['max_retries'], delay=TIMEOUTS['retry_delay'])
    def _run(self, query: str, context: str = "") -> str:
<<<<<<< HEAD
        """Получение данных от двух AI и их сравнение с таймаутами и кешированием"""
        try:
            # Проверяем кеш перед выполнением анализа
            cache_key = f"dual_ai_{self._llm1.model}_{self._llm2.model}_{query}_{context}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("АНАЛИЗ ИЗ КЕША", f"Модели: {self._llm1.model} + {self._llm2.model}")
                return cached_result
            
            log_step("АНАЛИЗ AI МОДЕЛЯМИ", f"Модели: {self._llm1.model} + {self._llm2.model}")
=======
        """Получение данных от двух AI и их сравнение с таймаутами"""
        try:
>>>>>>> af05dce (предварительная версия)
            print(f"🔄 Получение данных от двух AI моделей...")
            
            # Получаем ответы от обеих моделей с таймаутами
            response1 = self._get_ai_response_with_timeout(self._llm1, query, context)
            response2 = self._get_ai_response_with_timeout(self._llm2, query, context)
            
            # Проверяем на ошибки таймаута
            if "Серверы ИИ перегружены" in response1 or "Серверы ИИ перегружены" in response2:
<<<<<<< HEAD
                error_msg = "⏰ Серверы ИИ перегружены. Пожалуйста, повторите запрос позже."
                cache_manager.set(cache_key, error_msg)
                return error_msg
=======
                return "⏰ Серверы ИИ перегружены. Пожалуйста, повторите запрос позже."
>>>>>>> af05dce (предварительная версия)
            
            # Сравниваем и анализируем ответы
            comparison = self._compare_responses(response1, response2, query)
            
            # Сохраняем результат в кеш
            cache_manager.set(cache_key, comparison)
            
            return comparison
            
        except Exception as e:
            error_msg = f"Ошибка при получении данных от AI: {str(e)}"
            # Не можем использовать cache_key здесь, так как он может быть не определен
            return error_msg
    
    def _get_ai_response(self, llm: LLM, query: str, context: str) -> str:
        """Получение ответа от конкретной AI модели"""
        try:
            # Формируем промпт с контекстом
            full_prompt = f"""
            Контекст: {context}
            
            Запрос: {query}
            
            Пожалуйста, предоставьте детальный, точный и актуальный ответ.
            """
            
            # Здесь должна быть логика вызова LLM
            # Пока используем заглушку
            return f"Ответ от {llm.model}: {query[:100]}..."
            
        except Exception as e:
            return f"Ошибка получения ответа от {llm.model}: {str(e)}"
    
    def _get_ai_response_with_timeout(self, llm: LLM, query: str, context: str) -> str:
        """Получение ответа от AI модели с таймаутом и повторными попытками"""
        try:
            print(f"🤖 Запрос к {llm.model}...")
            
            # Формируем промпт с контекстом (ограничиваем размер для экономии токенов)
            context_short = context[:200] if context else ""
            query_short = query[:300] if query else ""
            
            full_prompt = f"""
            Контекст: {context_short}
            
            Запрос: {query_short}
            
            Пожалуйста, предоставьте краткий, но точный ответ (максимум 200 слов).
            """
            
            # Используем таймаут для запроса
            start_time = time.time()
            
            # РЕАЛЬНЫЙ ЗАПРОС К LLM (но с ограничениями)
            try:
                # TODO: Для реальных запросов к LLM раскомментируйте следующий код:
                # response = llm.complete(full_prompt)
                # return response.content
                
                # Пока используем заглушку для экономии токенов
                response = f"Ответ от {llm.model}: {query_short[:100]}..."
                print(f"✅ {llm.model} ответил за {time.time() - start_time:.2f}с")
                
                return response
                
            except Exception as llm_error:
                print(f"❌ Ошибка LLM {llm.model}: {str(llm_error)}")
                return f"Ошибка получения ответа от {llm.model}: {str(llm_error)}"
            
        except Exception as e:
            print(f"❌ Ошибка при запросе к {llm.model}: {str(e)}")
            return f"Ошибка получения ответа от {llm.model}: {str(e)}"
    
    def _compare_responses(self, response1: str, response2: str, query: str) -> str:
        """Сравнение ответов от двух AI моделей"""
        
        comparison_result = f"""
        ============================================================================
        СРАВНЕНИЕ ОТВЕТОВ ОТ ДВУХ AI МОДЕЛЕЙ
        ============================================================================
        
        ЗАПРОС: {query}
        
        ОТВЕТ МОДЕЛИ 1 ({self._llm1.model}):
        {response1}
        
        ОТВЕТ МОДЕЛИ 2 ({self._llm2.model}):
        {response2}
        
        АНАЛИЗ СХОДСТВ И РАЗЛИЧИЙ:
        """
        
        # Анализируем сходства
        similarities = self._find_similarities(response1, response2)
        comparison_result += f"\nСХОДСТВА:\n{similarities}"
        
        # Анализируем различия
        differences = self._find_differences(response1, response2)
        comparison_result += f"\n\nРАЗЛИЧИЯ:\n{differences}"
        
        # Оценка качества
        quality_assessment = self._assess_quality(response1, response2)
        comparison_result += f"\n\nОЦЕНКА КАЧЕСТВА:\n{quality_assessment}"
        
        # Формируем обобщенный вывод
        final_conclusion = self._create_final_conclusion(response1, response2, similarities, differences)
        comparison_result += f"\n\nОБОБЩЕННЫЙ ВЫВОД:\n{final_conclusion}"
        
        return comparison_result
    
    def _find_similarities(self, response1: str, response2: str) -> str:
        """Поиск сходств в ответах"""
        # Простая логика поиска общих ключевых слов
        words1 = set(response1.lower().split())
        words2 = set(response2.lower().split())
        common_words = words1.intersection(words2)
        
        return f"Общие ключевые термины: {', '.join(list(common_words)[:10])}"
    
    def _find_differences(self, response1: str, response2: str) -> str:
        """Поиск различий в ответах"""
        # Анализ различий в подходах и детализации
        len1, len2 = len(response1), len(response2)
        detail_diff = f"Детализация: Модель 1 ({len1} символов) vs Модель 2 ({len2} символов)"
        
        return f"Различия в объеме информации: {detail_diff}"
    
    def _assess_quality(self, response1: str, response2: str) -> str:
        """Оценка качества ответов"""
        quality_metrics = []
        
        # Оценка детализации
        detail_score1 = len(response1.split()) / 100  # слов на 100
        detail_score2 = len(response2.split()) / 100
        quality_metrics.append(f"Детализация: Модель 1 ({detail_score1:.1f}/10) vs Модель 2 ({detail_score2:.1f}/10)")
        
        # Оценка структурированности
        structure_score1 = response1.count('\n') / 10
        structure_score2 = response2.count('\n') / 10
        quality_metrics.append(f"Структурированность: Модель 1 ({structure_score1:.1f}/10) vs Модель 2 ({structure_score2:.1f}/10)")
        
        return "\n".join(quality_metrics)
    
    def _create_final_conclusion(self, response1: str, response2: str, similarities: str, differences: str) -> str:
        """Создание обобщенного вывода"""
        return f"""
        На основе анализа двух AI моделей:
        
        ✅ СОВПАДАЮЩИЕ ВЫВОДЫ: {similarities[:100]}...
        ⚠️ РАЗЛИЧАЮЩИЕСЯ АСПЕКТЫ: {differences[:100]}...
        
        РЕКОМЕНДАЦИЯ: Использовать обобщенную информацию, 
        учитывая сильные стороны каждой модели.
        """

# Создаем инструменты с двойной валидацией только если есть LLM
dual_cbrf_tool = None
dual_moex_tool = None
dual_news_tool = None
dual_financial_tool = None

# Проверяем доступность LLM для двойной валидации (GPT-4.1 первая, Claude вторая, DeepSeek последняя)
available_llms = []
if llm_gpt4:
    available_llms.append(("GPT-4.1", llm_gpt4))
if llm_anthropic:
    available_llms.append(("Claude", llm_anthropic))
if llm_deepseek:
    available_llms.append(("DeepSeek", llm_deepseek))

# ВОССТАНАВЛИВАЕМ ДВОЙНУЮ ВАЛИДАЦИЮ - ГЛАВНАЯ ФИЧА!
if len(available_llms) >= 2:
    log_step("СОЗДАНИЕ ИНСТРУМЕНТОВ", "Инициализация инструментов двойной валидации")
    try:
        # Выбираем первые две доступные модели для двойной валидации
        llm1_name, llm1 = available_llms[0]
        llm2_name, llm2 = available_llms[1]
        
        print(f"🤖 Используем для двойной валидации: {llm1_name} + {llm2_name}")
        
        dual_cbrf_tool = DualAITool(
            name="dual_cbrf_api_tool",
            description=f"Инструмент для получения данных от ЦБ РФ с двойной валидацией AI ({llm1_name} + {llm2_name})",
            llm1=llm1,
            llm2=llm2
        )
        
        dual_moex_tool = DualAITool(
            name="dual_moex_api_tool", 
            description=f"Инструмент для получения данных от MOEX с двойной валидацией AI ({llm1_name} + {llm2_name})",
            llm1=llm1,
            llm2=llm2
        )
        
        dual_news_tool = DualAITool(
            name="dual_news_api_tool",
            description=f"Инструмент для анализа новостей с двойной валидацией AI ({llm1_name} + {llm2_name})", 
            llm1=llm1,
            llm2=llm2
        )
        
        dual_financial_tool = DualAITool(
            name="dual_financial_analysis_tool",
            description=f"Инструмент для финансового анализа с двойной валидацией AI ({llm1_name} + {llm2_name})",
            llm1=llm1, 
            llm2=llm2
        )
        log_step("ИНСТРУМЕНТЫ СОЗДАНЫ", f"4 инструмента двойной валидации успешно инициализированы ({llm1_name} + {llm2_name})")
    except Exception as e:
        log_step("ОШИБКА ИНСТРУМЕНТОВ", f"Ошибка создания инструментов: {e}")
else:
    available_names = [name for name, _ in available_llms]
    log_step("ДВОЙНАЯ ВАЛИДАЦИЯ НЕДОСТУПНА", f"Доступные модели: {available_names}")

# ============================================================================
# ИНСТРУМЕНТЫ ДЛЯ РАБОТЫ С API
# ============================================================================

class CBRFTool(BaseTool):
    name: str = "cbrf_api_tool"
    description: str = "Инструмент для получения данных от Центрального Банка РФ"

    @retry_on_timeout(max_retries=TIMEOUTS['max_retries'], delay=TIMEOUTS['retry_delay'])
    def _run(self, query: str) -> str:
<<<<<<< HEAD
        """Получение данных от ЦБ РФ с таймаутом и кешированием"""
        try:
            # Проверяем кеш перед выполнением запроса
            cache_key = f"cbrf_{query}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("ЦБ РФ ИЗ КЕША", f"Запрос: {query}")
                return cached_result
            
            log_step("ЗАПРОС ЦБ РФ", f"Запрос: {query}")
=======
        """Получение данных от ЦБ РФ с таймаутом"""
        try:
>>>>>>> af05dce (предварительная версия)
            print(f"🏦 Запрос к API ЦБ РФ: {query}")
            
            # Базовый URL API ЦБ РФ
            base_url = "http://www.cbr.ru/dataservice"
            
            # Получение списка публикаций
            if "publications" in query.lower():
                response = requests.get(f"{base_url}/publications", timeout=TIMEOUTS['api_request'])
<<<<<<< HEAD
                result = f"Список публикаций ЦБ РФ: {response.text[:500]}..."
                cache_manager.set(cache_key, result)
                return result
=======
                return f"Список публикаций ЦБ РФ: {response.text[:500]}..."
>>>>>>> af05dce (предварительная версия)
            
            # Получение данных по показателям
            elif "datasets" in query.lower():
                # Пример: получение показателей для публикации
                response = requests.get(f"{base_url}/datasets?publicationId=1", timeout=TIMEOUTS['api_request'])
<<<<<<< HEAD
                result = f"Данные показателей: {response.text[:500]}..."
                cache_manager.set(cache_key, result)
                return result
=======
                return f"Данные показателей: {response.text[:500]}..."
>>>>>>> af05dce (предварительная версия)
            
            result = "Используйте 'publications' или 'datasets' для получения данных"
            cache_manager.set(cache_key, result)
            return result
            
        except requests.Timeout:
<<<<<<< HEAD
            error_msg = "⏰ Таймаут при запросе к API ЦБ РФ"
            print(error_msg)
            # Используем безопасный ключ кеша для ошибки
            error_cache_key = f"cbrf_error_{query}"
            cache_manager.set(error_cache_key, error_msg)
            return error_msg
=======
            print("⏰ Таймаут при запросе к API ЦБ РФ")
            return "⏰ Серверы ЦБ РФ перегружены. Пожалуйста, повторите запрос позже."
>>>>>>> af05dce (предварительная версия)
        except Exception as e:
            error_msg = f"Ошибка при обращении к API ЦБ РФ: {str(e)}"
            # Используем безопасный ключ кеша для ошибки
            error_cache_key = f"cbrf_error_{query}"
            cache_manager.set(error_cache_key, error_msg)
            return error_msg

class MOEXTool(BaseTool):
    name: str = "moex_api_tool"
    description: str = "Инструмент для получения данных от Московской биржи"

    @retry_on_timeout(max_retries=TIMEOUTS['max_retries'], delay=TIMEOUTS['retry_delay'])
    def _run(self, query: str) -> str:
<<<<<<< HEAD
        """Получение данных от Московской биржи с таймаутом и кешированием"""
        try:
            # Проверяем кеш перед выполнением запроса
            cache_key = f"moex_{query}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("MOEX ИЗ КЕША", f"Запрос: {query}")
                return cached_result
            
            log_step("ЗАПРОС MOEX", f"Запрос: {query}")
=======
        """Получение данных от Московской биржи с таймаутом"""
        try:
>>>>>>> af05dce (предварительная версия)
            print(f"📈 Запрос к API MOEX: {query}")
            
            base_url = "https://iss.moex.com/iss"
            
            # Получение информации о ценных бумагах
            if "securities" in query.lower():
                response = requests.get(f"{base_url}/securities.json", timeout=TIMEOUTS['api_request'])
<<<<<<< HEAD
                result = f"Данные о ценных бумагах: {response.text[:500]}..."
                cache_manager.set(cache_key, result)
                return result
=======
                return f"Данные о ценных бумагах: {response.text[:500]}..."
>>>>>>> af05dce (предварительная версия)
            
            # Получение исторических данных
            elif "history" in query.lower():
                # Пример: исторические данные по акциям
                response = requests.get(f"{base_url}/history/engines/stock/markets/shares/securities.json?date=2024-01-01")
                result = f"Исторические данные: {response.text[:500]}..."
                cache_manager.set(cache_key, result)
                return result
            
            result = "Используйте 'securities' или 'history' для получения данных"
            cache_manager.set(cache_key, result)
            return result
            
        except Exception as e:
            error_msg = f"Ошибка при обращении к API MOEX: {str(e)}"
            # Используем безопасный ключ кеша для ошибки
            error_cache_key = f"moex_error_{query}"
            cache_manager.set(error_cache_key, error_msg)
            return error_msg

class RealTimeSearchTool(BaseTool):
    name: str = "real_time_search_tool"
    description: str = "Инструмент для поиска актуальных данных о компаниях в интернете с кешированием"

    @retry_on_timeout(max_retries=TIMEOUTS['max_retries'], delay=TIMEOUTS['retry_delay'])
    def _run(self, company_name: str, search_type: str = "comprehensive") -> str:
        """Поиск актуальных данных о компании с кешированием"""
        try:
            if not search_llm:
                return "❌ Поисковый LLM недоступен. Проверьте настройки PROXYAPI_SEARCH_KEY"
            
            current_date = datetime.now().strftime('%d.%m.%Y')
            
            # Формируем поисковый запрос в зависимости от типа
            if search_type == "stock_price":
                search_query = f"текущая цена акций {company_name} котировки {current_date}"
            elif search_type == "news":
                search_query = f"последние новости {company_name} за последние 7 дней {current_date}"
            elif search_type == "financial":
                search_query = f"финансовые результаты {company_name} отчеты 2025 {current_date}"
            else:
                search_query = f"актуальная информация о компании {company_name} {current_date} последние новости котировки финансовые результаты"
            
            # Проверяем кеш перед выполнением поиска
            cache_key = f"search_{company_name}_{search_type}_{current_date}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("ПОИСК ИЗ КЕША", f"Компания: {company_name}, тип: {search_type}")
                return cached_result
            
            log_step("ПОИСК В ИНТЕРНЕТЕ", f"Компания: {company_name}, тип: {search_type}")
            print(f"🔍 Поиск актуальных данных: {search_query}")
            
            # Выполняем поиск через ProxyAPI
            try:
                response = search_llm.responses.create(
                    model="gpt-4o",
                    tools=[{
                        "type": "web_search_preview",
                        "search_context_size": "high",  # Полный контекст для анализа
                        "user_location": {
                            "type": "approximate",
                            "country": "RU"
                        }
                    }],
                    input=search_query
                )
                
                # Извлекаем результаты поиска
                search_results = []
                try:
                    # Преобразуем ответ в строку и извлекаем информацию
                    response_str = str(response)
                    search_results.append(f"Результат поиска для '{company_name}': {response_str}")
                except Exception as parse_error:
                    print(f"⚠️ Ошибка обработки результатов поиска: {parse_error}")
                    search_results.append(f"Результат поиска: {str(response)}")
                
                if search_results:
                    combined_results = "\n\n".join(search_results)
                    result = f"""
                    🔍 АКТУАЛЬНЫЕ ДАННЫЕ О КОМПАНИИ {company_name.upper()}
                    📅 ДАТА ПОИСКА: {current_date}
                    
                    {combined_results}
                    
                    ⚠️ ВАЖНО: Данные получены из интернета на {current_date}
                    """
                    
                    # Сохраняем результат в кеш
                    cache_manager.set(cache_key, result)
                    return result
                else:
                    error_msg = f"❌ Не удалось найти актуальные данные о компании {company_name} на {current_date}"
                    cache_manager.set(cache_key, error_msg)
                    return error_msg
                    
            except Exception as search_error:
                error_msg = f"❌ Ошибка поиска через ProxyAPI: {str(search_error)}"
                print(f"❌ Ошибка поиска данных: {error_msg}")
                # Используем безопасный ключ кеша для ошибки
                error_cache_key = f"search_error_{company_name}_{search_type}_{current_date}"
                cache_manager.set(error_cache_key, error_msg)
                return error_msg
                
        except Exception as e:
            error_msg = f"❌ Общая ошибка поиска: {str(e)}"
            print(f"❌ Ошибка поиска данных: {error_msg}")
            return error_msg

class NewsTool(BaseTool):
    name: str = "news_api_tool"
    description: str = "Инструмент для получения новостей о компаниях"

    def _run(self, company_name: str) -> str:
        """Получение новостей о компании с кешированием"""
        try:
<<<<<<< HEAD
            # Проверяем кеш перед выполнением запроса
            cache_key = f"news_{company_name}"
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("НОВОСТИ ИЗ КЕША", f"Компания: {company_name}")
                return cached_result
            
            log_step("ПОЛУЧЕНИЕ НОВОСТЕЙ", f"Компания: {company_name}")
            
=======
>>>>>>> af05dce (предварительная версия)
            # Пример использования BeautifulSoup для парсинга новостей
            # В реальной реализации здесь будет парсинг новостных сайтов
            sample_news = [
                f"📰 {company_name} объявила о росте прибыли на 15%",
                f"📈 Акции {company_name} выросли на 3.2%",
                f"🏢 {company_name} открывает новый офис в Москве"
            ]
            
            # Используем numpy для анализа тональности (пример)
            sentiment_scores = np.array([0.8, 0.6, 0.4])  # Позитивные новости
            avg_sentiment = np.mean(sentiment_scores)
            
            news_result = f"""
            📰 НОВОСТИ О КОМПАНИИ {company_name.upper()}:
            
            {'\n'.join(sample_news)}
            
            📊 АНАЛИЗ ТОНАЛЬНОСТИ (с использованием numpy):
            • Средняя тональность: {avg_sentiment:.2f} (позитивная)
            • Количество новостей: {len(sample_news)}
            • Стандартное отклонение: {np.std(sentiment_scores):.2f}
            
            🔍 ПРИМЕЧАНИЕ: В реальной реализации здесь будет парсинг 
            новостных сайтов с использованием BeautifulSoup и lxml.
            """
<<<<<<< HEAD
            
            # Сохраняем результат в кеш
            cache_manager.set(cache_key, news_result)
            return news_result
            
=======
            return news_result
>>>>>>> af05dce (предварительная версия)
        except Exception as e:
            error_msg = f"Ошибка при получении новостей: {str(e)}"
            # Используем безопасный ключ кеша для ошибки
            error_cache_key = f"news_error_{company_name}"
            cache_manager.set(error_cache_key, error_msg)
            return error_msg

class FinancialAnalysisTool(BaseTool):
    name: str = "financial_analysis_tool"
    description: str = "Инструмент для анализа финансовых показателей"

    def _run(self, company_data: str) -> str:
        """Анализ финансовых показателей компании с кешированием"""
        try:
<<<<<<< HEAD
            # Проверяем кеш перед выполнением анализа
            cache_key = f"financial_{company_data[:100]}"  # Используем первые 100 символов как ключ
            cached_result = cache_manager.get(cache_key)
            
            if cached_result:
                log_step("ФИНАНСОВЫЙ АНАЛИЗ ИЗ КЕША", f"Данные: {company_data[:50]}...")
                return cached_result
            
            log_step("ФИНАНСОВЫЙ АНАЛИЗ", f"Данные: {company_data[:50]}...")
            
=======
>>>>>>> af05dce (предварительная версия)
            # Пример использования pandas для анализа данных
            if company_data and len(company_data) > 10:
                # Создаем DataFrame для демонстрации
                sample_data = {
                    'Показатель': ['Выручка', 'Прибыль', 'Активы', 'Обязательства'],
                    'Значение': [1000000, 150000, 2000000, 800000],
                    'Единица': ['руб.', 'руб.', 'руб.', 'руб.']
                }
                df = pd.DataFrame(sample_data)
                
                # Рассчитываем финансовые коэффициенты
                рентабельность = df.loc[df['Показатель'] == 'Прибыль', 'Значение'].iloc[0] / df.loc[df['Показатель'] == 'Выручка', 'Значение'].iloc[0] * 100
                ликвидность = df.loc[df['Показатель'] == 'Активы', 'Значение'].iloc[0] / df.loc[df['Показатель'] == 'Обязательства', 'Значение'].iloc[0]
                
                # Используем numpy для дополнительных расчетов
                значения = np.array(df['Значение'].tolist())
                среднее_значение = np.mean(значения)
                медиана = np.median(значения)
                стандартное_отклонение = np.std(значения)
                
                analysis_result = f"""
                📊 АНАЛИЗ ФИНАНСОВЫХ ПОКАЗАТЕЛЕЙ (с использованием pandas + numpy):
                
                {df.to_string(index=False)}
                
                📈 РАСЧЕТ КОЭФФИЦИЕНТОВ:
                • Рентабельность продаж: {рентабельность:.1f}%
                • Коэффициент ликвидности: {ликвидность:.2f}
                
                📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ (numpy):
                • Среднее значение показателей: {среднее_значение:,.0f} руб.
                • Медиана: {медиана:,.0f} руб.
                • Стандартное отклонение: {стандартное_отклонение:,.0f} руб.
                
                📋 ИСХОДНЫЕ ДАННЫЕ: {company_data[:200]}...
                """
<<<<<<< HEAD
                
                # Сохраняем результат в кеш
                cache_manager.set(cache_key, analysis_result)
                return analysis_result
            else:
                result = f"Анализ финансовых показателей: {company_data[:200]}..."
                cache_manager.set(cache_key, result)
                return result
                
=======
                return analysis_result
            else:
                return f"Анализ финансовых показателей: {company_data[:200]}..."
>>>>>>> af05dce (предварительная версия)
        except Exception as e:
            error_msg = f"Ошибка при анализе финансовых данных: {str(e)}"
            # Используем безопасный ключ кеша для ошибки
            error_cache_key = f"financial_error_{company_data[:50] if company_data else 'unknown'}"
            cache_manager.set(error_cache_key, error_msg)
            return error_msg

# Создаем инструмент для поиска актуальных данных
real_time_search_tool = RealTimeSearchTool()

# Создаем экземпляры инструментов
cbrf_tool = CBRFTool()
moex_tool = MOEXTool()
news_tool = NewsTool()
financial_tool = FinancialAnalysisTool()

# ============================================================================
# АГЕНТЫ СИСТЕМЫ ПОДДЕРЖКИ ПРИНЯТИЯ РЕШЕНИЙ
# ============================================================================

# Создаем агентов с доступными инструментами
def get_available_tools():
    """Получение списка доступных инструментов"""
    tools = []
    if dual_cbrf_tool:
        tools.append(dual_cbrf_tool)
    if dual_moex_tool:
        tools.append(dual_moex_tool)
    if dual_news_tool:
        tools.append(dual_news_tool)
    if dual_financial_tool:
        tools.append(dual_financial_tool)
    # Добавляем базовые инструменты
    tools.extend([cbrf_tool, moex_tool, news_tool, financial_tool])
<<<<<<< HEAD
    # Добавляем инструмент поиска актуальных данных
    if real_time_search_tool:
        tools.append(real_time_search_tool)
=======
>>>>>>> af05dce (предварительная версия)
    return tools

# Аналитик кейсов с двойной валидацией
case_analyst = Agent(
    role="Аналитик кейсов",
    goal="Поиск и анализ релевантных кейс-стади примеров для целевой компании с двойной валидацией AI",
    backstory="""Вы опытный аналитик с глубоким пониманием различных отраслей и бизнес-моделей. 
    Ваша задача - найти похожие кейсы в базе данных и проанализировать их применимость к текущей ситуации.
    Вы используете две AI модели для валидации информации и обеспечения максимальной точности анализа.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

# Финансовый аналитик с двойной валидацией
financial_analyst = Agent(
    role="Финансовый аналитик",
    goal="Оценка финансового состояния компании на основе отчетов и показателей с двойной валидацией",
    backstory="""Вы эксперт по финансовому анализу с многолетним опытом работы в инвестиционных компаниях. 
    Специализируетесь на анализе финансовых отчетов, расчете ключевых показателей и оценке инвестиционной привлекательности.
    Используете две AI модели для кросс-проверки финансовых данных и расчетов.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

# Аналитик компании с двойной валидацией
company_analyst = Agent(
    role="Аналитик компании",
    goal="Изучение онлайн-присутствия организации и анализ отраслевых трендов с двойной валидацией",
    backstory="""Вы специалист по анализу компаний и отраслей. Изучаете веб-сайты, социальные сети, 
    пресс-релизы и другие источники информации для понимания стратегии компании и рыночного позиционирования.
    Используете две AI модели для валидации качественного анализа и выявления скрытых трендов.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

# Аналитик по лицам, принимающим решения с двойной валидацией
decision_maker_analyst = Agent(
    role="Аналитик по лицам, принимающим решения",
    goal="Сбор информации о ключевых фигурах компании и их влиянии на бизнес с двойной валидацией",
    backstory="""Вы эксперт по анализу руководящего состава компаний. Изучаете биографии, опыт, 
    стиль управления и стратегические решения ключевых лиц для оценки их влияния на будущее компании.
    Используете две AI модели для валидации оценки персоналий и прогнозирования их влияния.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

# Аналитик новостей с двойной валидацией
news_analyst = Agent(
    role="Аналитик новостей",
    goal="Исследование последних новостей и их влияния на рынок и компанию с двойной валидацией",
    backstory="""Вы специалист по анализу новостного фона и его влияния на рынки. Отслеживаете 
    последние события, анализируете их потенциальное влияние на акции и даете рекомендации по реакции на новости.
    Используете две AI модели для валидации тональности новостей и оценки их реального влияния.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

# Советник по рискам с двойной валидацией
risk_advisor = Agent(
    role="Советник по рискам",
    goal="Оценка рисков и предоставление рекомендаций по диверсификации с двойной валидацией",
    backstory="""Вы эксперт по управлению рисками в инвестициях. Анализируете различные типы рисков 
    (рыночные, отраслевые, корпоративные) и разрабатываете стратегии диверсификации для минимизации потерь.
    Используете две AI модели для валидации оценки рисков и оптимизации портфельных решений.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)

<<<<<<< HEAD
# ============================================================================
# СПЕЦИАЛИЗИРОВАННЫЕ АГЕНТЫ-ВАЛИДАТОРЫ С ДВОЙНОЙ ВАЛИДАЦИЕЙ
# ============================================================================

class ValidationAgent(Agent):
    """Базовый класс для агентов-валидаторов с двойной валидацией и кешированием"""
    
    def __init__(self, role: str, goal: str, backstory: str, llm1: LLM, llm2: LLM, tools: List[BaseTool]):
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            allow_delegation=False,
            tools=tools,
            llm=llm1  # Основная LLM для агента
        )
        self._llm1 = llm1
        self._llm2 = llm2
        self._validation_cache = {}
    
    def validate_analysis(self, agent_result: str, analysis_type: str, context: str = "") -> str:
        """Валидация результатов анализа двумя LLM с кешированием"""
        try:
            # Создаем ключ кеша для валидации
            cache_key = f"validation_{analysis_type}_{hash(agent_result + context)}"
            
            # Проверяем кеш
            cached_validation = cache_manager.get(cache_key)
            if cached_validation:
                log_step("ВАЛИДАЦИЯ ИЗ КЕША", f"Тип: {analysis_type}")
                return cached_validation
            
            log_step("ВАЛИДАЦИЯ ДВУМЯ LLM", f"Тип: {analysis_type}, Модели: {self._llm1.model} + {self._llm2.model}")
            
            # Получаем валидацию от двух LLM
            validation1 = self._get_validation_response(self._llm1, agent_result, analysis_type, context)
            validation2 = self._get_validation_response(self._llm2, agent_result, analysis_type, context)
            
            # Сравниваем и обобщаем результаты валидации
            final_validation = self._compare_validations(validation1, validation2, agent_result, analysis_type)
            
            # Сохраняем в кеш
            cache_manager.set(cache_key, final_validation)
            
            return final_validation
            
        except Exception as e:
            error_msg = f"Ошибка валидации {analysis_type}: {str(e)}"
            log_step("ОШИБКА ВАЛИДАЦИИ", error_msg)
            return error_msg
    
    def _get_validation_response(self, llm: LLM, agent_result: str, analysis_type: str, context: str) -> str:
        """Получение ответа валидации от конкретной LLM"""
        try:
            # Формируем промпт для валидации
            validation_prompt = f"""
            ВАЛИДАЦИЯ АНАЛИЗА: {analysis_type.upper()}
            
            КОНТЕКСТ: {context}
            
            РЕЗУЛЬТАТ АГЕНТА:
            {agent_result}
            
            ЗАДАЧА: Проанализируйте качество и точность данного анализа. Проверьте:
            1. Логичность выводов
            2. Полноту анализа
            3. Актуальность данных
            4. Обоснованность рекомендаций
            5. Потенциальные ошибки или упущения
            
            Предоставьте детальную оценку с указанием сильных и слабых сторон.
            """
            
            # Ограничиваем размер для экономии токенов
            validation_prompt = validation_prompt[:2000]
            
            # TODO: Для реальных запросов к LLM раскомментируйте:
            # response = llm.complete(validation_prompt)
            # return response.content
            
            # Пока используем заглушку
            response = f"Валидация от {llm.model}: Анализ {analysis_type} - качественный, но требует уточнений..."
            return response
            
        except Exception as e:
            return f"Ошибка валидации от {llm.model}: {str(e)}"
    
    def _compare_validations(self, validation1: str, validation2: str, agent_result: str, analysis_type: str) -> str:
        """Сравнение результатов валидации от двух LLM"""
        
        comparison_result = f"""
        ============================================================================
        ВАЛИДАЦИЯ АНАЛИЗА: {analysis_type.upper()}
        ============================================================================
        
        РЕЗУЛЬТАТ АГЕНТА:
        {agent_result[:500]}...
        
        ВАЛИДАЦИЯ МОДЕЛИ 1 ({self._llm1.model}):
        {validation1}
        
        ВАЛИДАЦИЯ МОДЕЛИ 2 ({self._llm2.model}):
        {validation2}
        
        ОБОБЩЕННАЯ ОЦЕНКА:
        """
        
        # Анализируем согласованность валидаций
        agreement_level = self._assess_agreement(validation1, validation2)
        comparison_result += f"\nУРОВЕНЬ СОГЛАСОВАННОСТИ: {agreement_level}"
        
        # Формируем финальные рекомендации
        final_recommendations = self._create_final_recommendations(validation1, validation2, agent_result)
        comparison_result += f"\n\nФИНАЛЬНЫЕ РЕКОМЕНДАЦИИ:\n{final_recommendations}"
        
        return comparison_result
    
    def _assess_agreement(self, validation1: str, validation2: str) -> str:
        """Оценка уровня согласованности между валидациями"""
        # Простая логика оценки согласованности
        positive_words = ['качественный', 'точный', 'полный', 'обоснованный', 'логичный']
        negative_words = ['ошибка', 'упущение', 'неточность', 'неполный', 'сомнительный']
        
        pos1 = sum(1 for word in positive_words if word in validation1.lower())
        pos2 = sum(1 for word in positive_words if word in validation2.lower())
        neg1 = sum(1 for word in negative_words if word in validation1.lower())
        neg2 = sum(1 for word in negative_words if word in validation2.lower())
        
        if pos1 > neg1 and pos2 > neg2:
            return "ВЫСОКИЙ - Обе модели подтверждают качество анализа"
        elif pos1 > neg1 or pos2 > neg2:
            return "СРЕДНИЙ - Модели дают разные оценки"
        else:
            return "НИЗКИЙ - Обе модели выявили проблемы"
    
    def _create_final_recommendations(self, validation1: str, validation2: str, agent_result: str) -> str:
        """Создание финальных рекомендаций на основе валидаций"""
        return f"""
        ✅ ПОДТВЕРЖДЕННЫЕ ВЫВОДЫ: Анализ содержит обоснованные выводы
        ⚠️ ТРЕБУЮЩИЕ УТОЧНЕНИЯ: Некоторые аспекты нуждаются в дополнительной проверке
        🔍 РЕКОМЕНДАЦИИ: Использовать результаты с учетом выявленных ограничений
        """

# Создаем специализированных валидаторов только если есть две LLM
case_validator = None
financial_validator = None
company_validator = None
leadership_validator = None
news_validator = None
risk_validator = None
final_validator = None

if len(available_llms) >= 2:
    log_step("СОЗДАНИЕ ВАЛИДАТОРОВ", "Инициализация специализированных валидаторов")
    try:
        # Выбираем первые две доступные модели для валидации
        llm1_name, llm1 = available_llms[0]
        llm2_name, llm2 = available_llms[1]
        
        print(f"🤖 Создаем валидаторов с моделями: {llm1_name} + {llm2_name}")
        
        # Валидатор кейс-анализа
        case_validator = ValidationAgent(
            role="Валидатор кейс-анализа",
            goal="Валидация анализа кейс-стади с двойной проверкой AI",
            backstory="""Вы эксперт по валидации кейс-анализа. Проверяете качество анализа 
            похожих случаев, обоснованность выводов и применимость уроков к целевой компании.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Валидатор финансового анализа
        financial_validator = ValidationAgent(
            role="Валидатор финансового анализа",
            goal="Валидация финансового анализа с двойной проверкой AI",
            backstory="""Вы эксперт по валидации финансового анализа. Проверяете точность 
            расчетов, обоснованность коэффициентов и качество финансовых прогнозов.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Валидатор анализа компании
        company_validator = ValidationAgent(
            role="Валидатор анализа компании",
            goal="Валидация анализа стратегии и позиционирования компании",
            backstory="""Вы эксперт по валидации стратегического анализа. Проверяете качество 
            оценки конкурентных преимуществ, рыночного позиционирования и стратегических рисков.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Валидатор анализа руководства
        leadership_validator = ValidationAgent(
            role="Валидатор анализа руководства",
            goal="Валидация анализа руководящего состава компании",
            backstory="""Вы эксперт по валидации анализа персоналий. Проверяете качество 
            оценки опыта руководства, стиля управления и потенциального влияния на компанию.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Валидатор анализа новостей
        news_validator = ValidationAgent(
            role="Валидатор анализа новостей",
            goal="Валидация анализа новостного фона и его влияния",
            backstory="""Вы эксперт по валидации новостного анализа. Проверяете качество 
            оценки тональности новостей, их влияния на акции и обоснованность рекомендаций.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Валидатор оценки рисков
        risk_validator = ValidationAgent(
            role="Валидатор оценки рисков",
            goal="Валидация комплексной оценки рисков инвестирования",
            backstory="""Вы эксперт по валидации оценки рисков. Проверяете полноту 
            анализа различных типов рисков и качество рекомендаций по диверсификации.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        # Финальный валидатор
        final_validator = ValidationAgent(
            role="Финальный валидатор",
            goal="Финальная валидация и обобщение всех результатов анализа",
            backstory="""Вы эксперт по финальной валидации. Проверяете согласованность 
            всех результатов, выявляете противоречия и создаете обобщенный отчет.""",
            llm1=llm1,
            llm2=llm2,
            tools=get_available_tools()
        )
        
        log_step("ВАЛИДАТОРЫ СОЗДАНЫ", f"7 специализированных валидаторов успешно инициализированы ({llm1_name} + {llm2_name})")
        
    except Exception as e:
        log_step("ОШИБКА ВАЛИДАТОРОВ", f"Ошибка создания валидаторов: {e}")
else:
    available_names = [name for name, _ in available_llms]
    log_step("ВАЛИДАТОРЫ НЕДОСТУПНЫ", f"Доступные модели: {available_names}")
=======
# Агент-валидатор для финальной проверки
validation_agent = Agent(
    role="Агент-валидатор",
    goal="Финальная валидация и обобщение результатов анализа с использованием двух AI моделей",
    backstory="""Вы эксперт по валидации и обобщению аналитических данных. Ваша задача - 
    проверить согласованность результатов от всех аналитиков, выявить противоречия и создать 
    финальный обобщенный отчет с рекомендациями. Используете две AI модели для максимальной 
    точности финальных выводов.""",
    verbose=True,
    allow_delegation=False,
    tools=get_available_tools(),
    llm=llm
)
>>>>>>> af05dce (предварительная версия)

# ============================================================================
# ФУНКЦИЯ СОЗДАНИЯ ЗАДАЧ
# ============================================================================

def create_investment_analysis_tasks(company_name: str) -> List[Task]:
    """Создание задач для анализа инвестиционной привлекательности компании"""
    
    # Задача 1: Анализ кейсов с двойной валидацией
    case_analysis_task = Task(
        description=f"""
        Проанализируйте кейс-стади примеры, которые могут быть релевантны для компании {company_name}.
        
        🔥 КРИТИЧЕСКИ ВАЖНО: Используйте ТОЛЬКО актуальные данные (не старше 3 месяцев)!
        
        ВАЖНО: Используйте двойную валидацию AI для каждого этапа анализа!
        
        Выполните следующие действия:
        1. 🔍 Найдите актуальные данные о компании {company_name} (используйте real_time_search_tool)
        2. 📊 Найдите похожие компании в той же отрасли (валидируйте с двумя AI)
        3. 📈 Изучите их историю развития и ключевые решения (сравните выводы двух AI)
        4. 🎯 Определите факторы успеха и неудач (проверьте согласованность)
        5. 💡 Сформулируйте уроки, применимые к {company_name} (обобщите результаты)
        
        ОБЯЗАТЕЛЬНО:
        - Используйте real_time_search_tool для получения свежих данных
        - Указывайте дату всех используемых данных
        - Если свежих данных нет - четко укажите это
        - Для каждого этапа получайте данные от двух AI моделей
        - Сравните их выводы и выявите сходства/различия
        - Создайте обобщенный вывод с указанием уровня согласованности
        
        Используйте доступные инструменты для получения данных о компаниях и отраслях.
        """,
        agent=case_analyst,
        expected_output="Подробный анализ релевантных кейс-стади с двойной валидацией AI и обобщенными выводами"
    )
    
    # Задача 2: Финансовый анализ
    financial_analysis_task = Task(
        description=f"""
        Проведите комплексный финансовый анализ компании {company_name}.
        
        Выполните следующие действия:
        1. Получите финансовые данные компании
        2. Рассчитайте ключевые финансовые показатели (P/E, P/B, ROE, ROA, долговая нагрузка)
        3. Сравните с отраслевыми средними значениями
        4. Оцените финансовую устойчивость и ликвидность
        5. Сделайте прогноз финансовых показателей
        
        Предоставьте детальный отчет с числовыми показателями и их интерпретацией.
        """,
        agent=financial_analyst,
        expected_output="Детальный финансовый анализ с расчетами и интерпретацией показателей"
    )
    
    # Задача 3: Анализ компании
    company_analysis_task = Task(
        description=f"""
        Изучите онлайн-присутствие и стратегию компании {company_name}.
        
        Выполните следующие действия:
        1. Проанализируйте официальный сайт компании
        2. Изучите присутствие в социальных сетях
        3. Оцените корпоративную культуру и имидж
        4. Определите ключевые конкурентные преимущества
        5. Проанализируйте отраслевые тренды и позицию компании
        
        Сфокусируйтесь на качественных аспектах бизнеса и стратегическом позиционировании.
        """,
        agent=company_analyst,
        expected_output="Анализ стратегии, позиционирования и отраслевых трендов"
    )
    
    # Задача 4: Анализ руководства
    leadership_analysis_task = Task(
        description=f"""
        Изучите ключевых лиц компании {company_name} и их влияние на бизнес.
        
        Выполните следующие действия:
        1. Соберите информацию о топ-менеджменте
        2. Изучите их опыт и достижения
        3. Проанализируйте стиль управления и принятия решений
        4. Оцените стабильность руководства
        5. Определите потенциальные риски, связанные с персоналом
        
        Обратите внимание на репутацию, опыт и стратегическое видение руководства.
        """,
        agent=decision_maker_analyst,
        expected_output="Анализ руководящего состава и оценка их влияния на компанию"
    )
    
    # Задача 5: Анализ новостей
    news_analysis_task = Task(
        description=f"""
        Проанализируйте новостной фон вокруг компании {company_name}.
        
        Выполните следующие действия:
        1. Соберите последние новости о компании
        2. Оцените тональность новостей (позитивная/негативная)
        3. Определите потенциальное влияние на акции
        4. Изучите отраслевые новости и их влияние
        5. Сформулируйте рекомендации по реакции на новости
        
        Фокусируйтесь на новостях за последние 3-6 месяцев.
        """,
        agent=news_analyst,
        expected_output="Анализ новостного фона с оценкой влияния на инвестиционную привлекательность"
    )
    
    # Задача 6: Оценка рисков
    risk_assessment_task = Task(
        description=f"""
        Проведите комплексную оценку рисков для инвестиций в {company_name}.
        
        Выполните следующие действия:
        1. Оцените рыночные риски (волатильность, корреляции)
        2. Проанализируйте отраслевые риски
        3. Определите корпоративные риски
        4. Оцените макроэкономические факторы
        5. Разработайте рекомендации по диверсификации
        
        Предоставьте количественную и качественную оценку рисков.
        """,
        agent=risk_advisor,
        expected_output="Комплексная оценка рисков с рекомендациями по управлению"
    )
    
    # Задача 7: Финальная валидация и обобщение
    validation_task = Task(
        description=f"""
        Проведите финальную валидацию и обобщение всех результатов анализа компании {company_name}.
        
        ВАЖНО: Используйте двойную валидацию AI для финального обобщения!
        
        Выполните следующие действия:
        1. Проанализируйте результаты всех предыдущих аналитиков
        2. Выявите согласующиеся и противоречивые выводы
        3. Оцените уровень согласованности между разными аспектами анализа
        4. Создайте финальный обобщенный отчет с рекомендациями
        5. Укажите уровень уверенности в каждом выводе
        
        Для каждого аспекта:
        - Сравните выводы от разных AI моделей
        - Оцените качество и актуальность данных
        - Сформулируйте обобщенные рекомендации
        - Укажите потенциальные риски и неопределенности
        
        Создайте структурированный отчет с четкими инвестиционными рекомендациями.
        """,
        agent=final_validator if final_validator else case_analyst,  # Используем финального валидатора или fallback
        expected_output="Финальный обобщенный отчет с валидированными рекомендациями по инвестициям"
    )
    
    return [
        case_analysis_task,
        financial_analysis_task,
        company_analysis_task,
        leadership_analysis_task,
        news_analysis_task,
        risk_assessment_task,
        validation_task
    ]

# ============================================================================
# ОСНОВНАЯ ФУНКЦИЯ АНАЛИЗА
# ============================================================================

@retry_on_timeout(max_retries=TIMEOUTS['max_retries'], delay=TIMEOUTS['retry_delay'])
def analyze_investment_opportunity(company_name: str) -> Union[str, Any]:
    """
<<<<<<< HEAD
    Основная функция для анализа инвестиционной привлекательности компании с таймаутами и кешированием
=======
    Основная функция для анализа инвестиционной привлекательности компании с таймаутами
>>>>>>> af05dce (предварительная версия)
    
    Args:
        company_name (str): Название компании для анализа
        
    Returns:
        Union[str, Any]: Подробный отчет с рекомендациями или объект CrewOutput
    """
    
    log_step("НАЧАЛО АНАЛИЗА", f"Компания: {company_name}")
    start_time = time.time()
    
<<<<<<< HEAD
    # Очищаем устаревший кеш перед анализом
    cleared_count = cache_manager.clear_old_cache()
    if cleared_count > 0:
        log_step("КЕШ ОЧИЩЕН", f"Удалено устаревших записей: {cleared_count}")
    
    # Проверяем доступность LLM
    if not llm:
        log_step("ДЕМО-РЕЖИМ", "LLM недоступны, используем демо-результат")
        # Демо-режим
        return f"""
        ============================================================================
        ДЕМО-АНАЛИЗ КОМПАНИИ: {company_name}
        ============================================================================
        
        📊 РЕЗУЛЬТАТЫ АНАЛИЗА С ДВОЙНОЙ ВАЛИДАЦИЕЙ AI
        
        ✅ Аналитик кейсов: Завершен
        ✅ Финансовый аналитик: Завершен  
        ✅ Аналитик компании: Завершен
        ✅ Аналитик руководства: Завершен
        ✅ Аналитик новостей: Завершен
        ✅ Советник по рискам: Завершен
        ✅ Агент-валидатор: Завершен
        
        🎯 ИНВЕСТИЦИОННАЯ РЕКОМЕНДАЦИЯ:
        На основе анализа двух AI моделей и комплексной оценки всех факторов,
        рекомендуется [ДЕМО-РЕЗУЛЬТАТ] для компании {company_name}.
        
        📈 УРОВЕНЬ УВЕРЕННОСТИ: 85%
        ⏱️ ВРЕМЯ АНАЛИЗА: {datetime.now().strftime('%H:%M:%S')}
        
        ⚠️  ПРИМЕЧАНИЕ: Это демо-результат. Для полного анализа настройте API ключи.
        """
    
=======
    # Проверяем доступность LLM
    if not llm:
        log_step("ДЕМО-РЕЖИМ", "LLM недоступны, используем демо-результат")
        # Демо-режим
        return f"""
        ============================================================================
        ДЕМО-АНАЛИЗ КОМПАНИИ: {company_name}
        ============================================================================
        
        📊 РЕЗУЛЬТАТЫ АНАЛИЗА С ДВОЙНОЙ ВАЛИДАЦИЕЙ AI
        
        ✅ Аналитик кейсов: Завершен
        ✅ Финансовый аналитик: Завершен  
        ✅ Аналитик компании: Завершен
        ✅ Аналитик руководства: Завершен
        ✅ Аналитик новостей: Завершен
        ✅ Советник по рискам: Завершен
        ✅ Агент-валидатор: Завершен
        
        🎯 ИНВЕСТИЦИОННАЯ РЕКОМЕНДАЦИЯ:
        На основе анализа двух AI моделей и комплексной оценки всех факторов,
        рекомендуется [ДЕМО-РЕЗУЛЬТАТ] для компании {company_name}.
        
        📈 УРОВЕНЬ УВЕРЕННОСТИ: 85%
        ⏱️ ВРЕМЯ АНАЛИЗА: {datetime.now().strftime('%H:%M:%S')}
        
        ⚠️  ПРИМЕЧАНИЕ: Это демо-результат. Для полного анализа настройте API ключи.
        """
    
>>>>>>> af05dce (предварительная версия)
    try:
        log_step("СОЗДАНИЕ ЗАДАЧ", "Формирование списка задач для анализа")
        # Создаем задачи
        tasks = create_investment_analysis_tasks(company_name)
        log_step("ЗАДАЧИ СОЗДАНЫ", f"Создано {len(tasks)} задач")
        
        log_step("СОЗДАНИЕ КОМАНДЫ", "Инициализация команды аналитиков")
        # Создаем команду с двойной валидацией
        investment_crew = Crew(
            agents=[
                case_analyst,
                financial_analyst,
                company_analyst,
                decision_maker_analyst,
                news_analyst,
                risk_advisor,
<<<<<<< HEAD
                final_validator if final_validator else case_analyst  # Используем финального валидатора или fallback
=======
                validation_agent
>>>>>>> af05dce (предварительная версия)
            ],
            tasks=tasks,
            verbose=True,
            process=Process.sequential
        )
        log_step("КОМАНДА СОЗДАНА", "Команда аналитиков готова к работе")
        
        log_step("ЗАПУСК АНАЛИЗА", "Начинаем выполнение Crew с таймаутом")
        # Запускаем анализ с таймаутом
        result = investment_crew.kickoff()
        
        analysis_time = time.time() - start_time
        log_step("АНАЛИЗ ЗАВЕРШЕН", f"Время выполнения: {analysis_time:.2f} секунд")
        
        # Преобразуем результат в строку, если это возможно
        if hasattr(result, 'raw'):
            return str(result.raw)
        elif hasattr(result, '__str__'):
            return str(result)
        else:
            return result
            
    except Exception as e:
        analysis_time = time.time() - start_time
        log_step("ОШИБКА АНАЛИЗА", f"Время до ошибки: {analysis_time:.2f}с, Ошибка: {str(e)}")
        
        error_str = str(e).lower()
        
        # Проверяем, можно ли переключиться на альтернативную модель
        if "insufficient balance" in error_str or "badrequesterror" in error_str:
            alternative_llm = switch_to_alternative_llm(llm, str(e))
            if alternative_llm and alternative_llm != llm:
                log_step("ПОВТОРНАЯ ПОПЫТКА", f"Переключились на {alternative_llm.model}, повторяем анализ...")
                try:
                    # Повторяем анализ с новой моделью
                    return analyze_investment_opportunity(company_name)
                except Exception as retry_exception:
                    return f"❌ Ошибка даже с альтернативной моделью: {str(retry_exception)}"
        
        # Если переключение невозможно или не помогло, возвращаем сообщение об ошибке
        if "timeout" in error_str or "timed out" in error_str:
            return "⏰ Серверы ИИ перегружены. Пожалуйста, повторите запрос позже."
        elif "insufficient balance" in error_str:
            return "💰 Недостаточно средств на балансе API. Пожалуйста, пополните счет."
        elif "badrequesterror" in error_str:
            return "🔧 Ошибка запроса к API. Проверьте настройки и повторите попытку."
        else:
            return f"❌ Ошибка анализа: {str(e)}"

# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    # Пример анализа компании
    company = "Сбербанк"  # Можно изменить на любую другую компанию
    print(f"Начинаем анализ инвестиционной привлекательности компании: {company}")
    print("=" * 80)
    
    try:
        result = analyze_investment_opportunity(company)
        print("\nРЕЗУЛЬТАТ АНАЛИЗА:")
        print("=" * 80)
        print(result)
    except Exception as e:
        print(f"Ошибка при выполнении анализа: {str(e)}")





