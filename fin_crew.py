from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
# Убираем проблемный импорт langchain.tools
# from langchain.tools import tool
import os
import requests
import json
from typing import List, Dict, Any, Union
from datetime import datetime, timedelta
# Убираем pandas, так как он не установлен
# import pandas as pd

from dotenv import load_dotenv
load_dotenv()

# Исправляем ошибки линтера - добавляем проверки на None
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_base_gpt4 = os.getenv("OPENAI_API_BASE_GPT4")

if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
if openai_api_base:
    os.environ["OPENAI_API_BASE"] = openai_api_base
if openai_api_base_gpt4:
    os.environ["OPENAI_API_BASE_GPT4"] = openai_api_base_gpt4

llm_deepseek = LLM(
    model="deepseek-chat",
    api_key=openai_api_key or "",
    base_url=openai_api_base or "",
    messages=[]
)

llm_gpt4 = LLM(
    model="gpt-4.1",
    api_key=openai_api_key or "",
    base_url=openai_api_base or "",
    messages=[]
)

# Выбираем модель для использования
llm = llm_deepseek  # или llm_gpt4

# ============================================================================
# СИСТЕМА ДВОЙНОЙ ВАЛИДАЦИИ С ДВУМЯ AI МОДЕЛЯМИ
# ============================================================================

class DualAITool(BaseTool):
    """Инструмент для получения и сравнения данных от двух AI моделей"""
    
    def __init__(self, name: str, description: str, llm1: LLM, llm2: LLM):
        super().__init__(name=name, description=description)
        self.llm1 = llm1
        self.llm2 = llm2
    
    def _run(self, query: str, context: str = "") -> str:
        """Получение данных от двух AI и их сравнение"""
        try:
            # Получаем ответы от обеих моделей
            response1 = self._get_ai_response(self.llm1, query, context)
            response2 = self._get_ai_response(self.llm2, query, context)
            
            # Сравниваем и анализируем ответы
            comparison = self._compare_responses(response1, response2, query)
            
            return comparison
            
        except Exception as e:
            return f"Ошибка при получении данных от AI: {str(e)}"
    
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
    
    def _compare_responses(self, response1: str, response2: str, query: str) -> str:
        """Сравнение ответов от двух AI моделей"""
        
        comparison_result = f"""
        ============================================================================
        СРАВНЕНИЕ ОТВЕТОВ ОТ ДВУХ AI МОДЕЛЕЙ
        ============================================================================
        
        ЗАПРОС: {query}
        
        ОТВЕТ МОДЕЛИ 1 ({self.llm1.model}):
        {response1}
        
        ОТВЕТ МОДЕЛИ 2 ({self.llm2.model}):
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

# Создаем инструменты с двойной валидацией
dual_cbrf_tool = DualAITool(
    name="dual_cbrf_api_tool",
    description="Инструмент для получения данных от ЦБ РФ с двойной валидацией AI",
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_moex_tool = DualAITool(
    name="dual_moex_api_tool", 
    description="Инструмент для получения данных от MOEX с двойной валидацией AI",
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_news_tool = DualAITool(
    name="dual_news_api_tool",
    description="Инструмент для анализа новостей с двойной валидацией AI", 
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_financial_tool = DualAITool(
    name="dual_financial_analysis_tool",
    description="Инструмент для финансового анализа с двойной валидацией AI",
    llm1=llm_deepseek, 
    llm2=llm_gpt4
)

# ============================================================================
# ИНСТРУМЕНТЫ ДЛЯ РАБОТЫ С API
# ============================================================================

class CBRFTool(BaseTool):
    name: str = "cbrf_api_tool"
    description: str = "Инструмент для получения данных от Центрального Банка РФ"

    def _run(self, query: str) -> str:
        """Получение данных от ЦБ РФ"""
        try:
            # Базовый URL API ЦБ РФ
            base_url = "http://www.cbr.ru/dataservice"
            
            # Получение списка публикаций
            if "publications" in query.lower():
                response = requests.get(f"{base_url}/publications")
                return f"Список публикаций ЦБ РФ: {response.text[:500]}..."
            
            # Получение данных по показателям
            elif "datasets" in query.lower():
                # Пример: получение показателей для публикации
                response = requests.get(f"{base_url}/datasets?publicationId=1")
                return f"Данные показателей: {response.text[:500]}..."
            
            return "Используйте 'publications' или 'datasets' для получения данных"
            
        except Exception as e:
            return f"Ошибка при обращении к API ЦБ РФ: {str(e)}"

class MOEXTool(BaseTool):
    name: str = "moex_api_tool"
    description: str = "Инструмент для получения данных от Московской биржи"

    def _run(self, query: str) -> str:
        """Получение данных от Московской биржи"""
        try:
            base_url = "https://iss.moex.com/iss"
            
            # Получение информации о ценных бумагах
            if "securities" in query.lower():
                response = requests.get(f"{base_url}/securities.json")
                return f"Данные о ценных бумагах: {response.text[:500]}..."
            
            # Получение исторических данных
            elif "history" in query.lower():
                # Пример: исторические данные по акциям
                response = requests.get(f"{base_url}/history/engines/stock/markets/shares/securities.json?date=2024-01-01")
                return f"Исторические данные: {response.text[:500]}..."
            
            return "Используйте 'securities' или 'history' для получения данных"
            
        except Exception as e:
            return f"Ошибка при обращении к API MOEX: {str(e)}"

class NewsTool(BaseTool):
    name: str = "news_api_tool"
    description: str = "Инструмент для получения новостей о компаниях"

    def _run(self, company_name: str) -> str:
        """Получение новостей о компании"""
        try:
            # Здесь можно интегрировать с различными новостными API
            # Пока возвращаем заглушку
            return f"Новости о компании {company_name}: [Заглушка - здесь будет интеграция с новостными API]"
        except Exception as e:
            return f"Ошибка при получении новостей: {str(e)}"

class FinancialAnalysisTool(BaseTool):
    name: str = "financial_analysis_tool"
    description: str = "Инструмент для анализа финансовых показателей"

    def _run(self, company_data: str) -> str:
        """Анализ финансовых показателей компании"""
        try:
            # Здесь будет логика анализа финансовых данных
            return f"Анализ финансовых показателей: {company_data[:200]}..."
        except Exception as e:
            return f"Ошибка при анализе финансовых данных: {str(e)}"

# Создаем экземпляры инструментов
cbrf_tool = CBRFTool()
moex_tool = MOEXTool()
news_tool = NewsTool()
financial_tool = FinancialAnalysisTool()

# Создаем экземпляры инструментов с двойной валидацией
dual_cbrf_tool = DualAITool(
    name="dual_cbrf_api_tool",
    description="Инструмент для получения данных от ЦБ РФ с двойной валидацией AI",
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_moex_tool = DualAITool(
    name="dual_moex_api_tool", 
    description="Инструмент для получения данных от MOEX с двойной валидацией AI",
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_news_tool = DualAITool(
    name="dual_news_api_tool",
    description="Инструмент для анализа новостей с двойной валидацией AI", 
    llm1=llm_deepseek,
    llm2=llm_gpt4
)

dual_financial_tool = DualAITool(
    name="dual_financial_analysis_tool",
    description="Инструмент для финансового анализа с двойной валидацией AI",
    llm1=llm_deepseek, 
    llm2=llm_gpt4
)

# ============================================================================
# АГЕНТЫ СИСТЕМЫ ПОДДЕРЖКИ ПРИНЯТИЯ РЕШЕНИЙ
# ============================================================================

# Аналитик кейсов с двойной валидацией
case_analyst = Agent(
    role="Аналитик кейсов",
    goal="Поиск и анализ релевантных кейс-стади примеров для целевой компании с двойной валидацией AI",
    backstory="""Вы опытный аналитик с глубоким пониманием различных отраслей и бизнес-моделей. 
    Ваша задача - найти похожие кейсы в базе данных и проанализировать их применимость к текущей ситуации.
    Вы используете две AI модели для валидации информации и обеспечения максимальной точности анализа.""",
    verbose=True,
    allow_delegation=False,
    tools=[dual_cbrf_tool, dual_moex_tool],
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
    tools=[dual_financial_tool, dual_cbrf_tool, dual_moex_tool],
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
    tools=[dual_news_tool],
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
    tools=[dual_news_tool],
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
    tools=[dual_news_tool],
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
    tools=[dual_cbrf_tool, dual_moex_tool],
    llm=llm
)

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
    tools=[dual_financial_tool, dual_news_tool],
    llm=llm
)

# ============================================================================
# ФУНКЦИЯ СОЗДАНИЯ ЗАДАЧ
# ============================================================================

def create_investment_analysis_tasks(company_name: str) -> List[Task]:
    """Создание задач для анализа инвестиционной привлекательности компании"""
    
    # Задача 1: Анализ кейсов с двойной валидацией
    case_analysis_task = Task(
        description=f"""
        Проанализируйте кейс-стади примеры, которые могут быть релевантны для компании {company_name}.
        
        ВАЖНО: Используйте двойную валидацию AI для каждого этапа анализа!
        
        Выполните следующие действия:
        1. Найдите похожие компании в той же отрасли (валидируйте с двумя AI)
        2. Изучите их историю развития и ключевые решения (сравните выводы двух AI)
        3. Определите факторы успеха и неудач (проверьте согласованность)
        4. Сформулируйте уроки, применимые к {company_name} (обобщите результаты)
        
        Для каждого этапа:
        - Получите данные от двух AI моделей
        - Сравните их выводы
        - Выявите сходства и различия
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
        agent=validation_agent,
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

def analyze_investment_opportunity(company_name: str) -> Union[str, Any]:
    """
    Основная функция для анализа инвестиционной привлекательности компании
    
    Args:
        company_name (str): Название компании для анализа
        
    Returns:
        Union[str, Any]: Подробный отчет с рекомендациями или объект CrewOutput
    """
    
    # Создаем задачи
    tasks = create_investment_analysis_tasks(company_name)
    
    # Создаем команду с двойной валидацией
    investment_crew = Crew(
        agents=[
            case_analyst,
            financial_analyst,
            company_analyst,
            decision_maker_analyst,
            news_analyst,
            risk_advisor,
            validation_agent
        ],
        tasks=tasks,
        verbose=True,
        process=Process.sequential
    )
    
    # Запускаем анализ
    result = investment_crew.kickoff()
    
    # Преобразуем результат в строку, если это возможно
    if hasattr(result, 'raw'):
        return str(result.raw)
    elif hasattr(result, '__str__'):
        return str(result)
    else:
        return result

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





