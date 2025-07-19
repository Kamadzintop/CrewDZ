"""
Расширенные инструменты для системы поддержки принятия решений инвесторов
с двойной валидацией AI
"""

import requests
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import time

class AdvancedCBRFTool:
    """Расширенный инструмент для работы с API ЦБ РФ"""
    
    def __init__(self):
        self.base_url = "http://www.cbr.ru/dataservice"
        self.session = requests.Session()
    
    def get_publications(self) -> Dict[str, Any]:
        """Получение списка публикаций ЦБ РФ"""
        try:
            response = self.session.get(f"{self.base_url}/publications")
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка получения публикаций: {str(e)}"}
    
    def get_datasets(self, publication_id: int) -> Dict[str, Any]:
        """Получение показателей для публикации"""
        try:
            response = self.session.get(f"{self.base_url}/datasets", 
                                      params={"publicationId": publication_id})
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка получения показателей: {str(e)}"}
    
    def get_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Получение данных по параметрам"""
        try:
            response = self.session.get(f"{self.base_url}/data", params=params)
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка получения данных: {str(e)}"}

class AdvancedMOEXTool:
    """Расширенный инструмент для работы с API Московской биржи"""
    
    def __init__(self):
        self.base_url = "https://iss.moex.com/iss"
        self.session = requests.Session()
    
    def get_securities(self, query: str = "") -> Dict[str, Any]:
        """Поиск ценных бумаг"""
        try:
            params = {"q": query} if query else {}
            response = self.session.get(f"{self.base_url}/securities.json", params=params)
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка поиска ценных бумаг: {str(e)}"}
    
    def get_security_info(self, security: str) -> Dict[str, Any]:
        """Получение информации о ценной бумаге"""
        try:
            response = self.session.get(f"{self.base_url}/securities/{security}.json")
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка получения информации о бумаге: {str(e)}"}
    
    def get_history(self, security: str, date_from: str, date_to: str) -> Dict[str, Any]:
        """Получение исторических данных"""
        try:
            params = {
                "from": date_from,
                "till": date_to,
                "iss.meta": "off",
                "iss.only": "history"
            }
            response = self.session.get(
                f"{self.base_url}/history/engines/stock/markets/shares/securities/{security}.json",
                params=params
            )
            return response.json()
        except Exception as e:
            return {"error": f"Ошибка получения исторических данных: {str(e)}"}

class NewsAnalysisTool:
    """Инструмент для анализа новостей"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def search_news(self, company_name: str, days_back: int = 30) -> List[Dict[str, Any]]:
        """Поиск новостей о компании"""
        try:
            # Здесь можно интегрировать с различными новостными API
            # Пока возвращаем заглушку с примером структуры
            return [
                {
                    "title": f"Новость о {company_name}",
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "source": "Пример источника",
                    "content": f"Пример новости о компании {company_name}",
                    "sentiment": "neutral"
                }
            ]
        except Exception as e:
            return [{"error": f"Ошибка поиска новостей: {str(e)}"}]
    
    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Анализ тональности текста"""
        try:
            # Простой анализ тональности по ключевым словам
            positive_words = ["рост", "увеличение", "прибыль", "успех", "развитие"]
            negative_words = ["падение", "убыток", "кризис", "проблемы", "риски"]
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "positive_score": positive_count,
                "negative_score": negative_count,
                "confidence": min(positive_count + negative_count, 10) / 10
            }
        except Exception as e:
            return {"error": f"Ошибка анализа тональности: {str(e)}"}

class FinancialAnalysisTool:
    """Инструмент для финансового анализа"""
    
    def __init__(self):
        self.moex_tool = AdvancedMOEXTool()
    
    def calculate_pe_ratio(self, price: float, earnings: float) -> float:
        """Расчет P/E ratio"""
        return price / earnings if earnings > 0 else None
    
    def calculate_pb_ratio(self, price: float, book_value: float) -> float:
        """Расчет P/B ratio"""
        return price / book_value if book_value > 0 else None
    
    def calculate_roe(self, net_income: float, equity: float) -> float:
        """Расчет ROE"""
        return (net_income / equity) * 100 if equity > 0 else None
    
    def calculate_roa(self, net_income: float, assets: float) -> float:
        """Расчет ROA"""
        return (net_income / assets) * 100 if assets > 0 else None
    
    def analyze_financial_health(self, financial_data: Dict[str, float]) -> Dict[str, Any]:
        """Комплексный анализ финансового здоровья"""
        try:
            analysis = {
                "pe_ratio": self.calculate_pe_ratio(
                    financial_data.get("price", 0),
                    financial_data.get("earnings", 0)
                ),
                "pb_ratio": self.calculate_pb_ratio(
                    financial_data.get("price", 0),
                    financial_data.get("book_value", 0)
                ),
                "roe": self.calculate_roe(
                    financial_data.get("net_income", 0),
                    financial_data.get("equity", 0)
                ),
                "roa": self.calculate_roa(
                    financial_data.get("net_income", 0),
                    financial_data.get("assets", 0)
                )
            }
            
            # Оценка финансового здоровья
            health_score = 0
            if analysis["pe_ratio"] and analysis["pe_ratio"] < 20:
                health_score += 25
            if analysis["pb_ratio"] and analysis["pb_ratio"] < 3:
                health_score += 25
            if analysis["roe"] and analysis["roe"] > 10:
                health_score += 25
            if analysis["roa"] and analysis["roa"] > 5:
                health_score += 25
            
            analysis["health_score"] = health_score
            analysis["health_status"] = "excellent" if health_score >= 80 else \
                                      "good" if health_score >= 60 else \
                                      "fair" if health_score >= 40 else "poor"
            
            return analysis
            
        except Exception as e:
            return {"error": f"Ошибка финансового анализа: {str(e)}"}

class DataValidator:
    """Класс для валидации данных"""
    
    @staticmethod
    def validate_numeric_data(data: Any, min_value: float = None, max_value: float = None) -> bool:
        """Валидация числовых данных"""
        try:
            if data is None:
                return False
            value = float(data)
            if min_value is not None and value < min_value:
                return False
            if max_value is not None and value > max_value:
                return False
            return True
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_date_format(date_str: str, format: str = "%Y-%m-%d") -> bool:
        """Валидация формата даты"""
        try:
            datetime.strptime(date_str, format)
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_company_name(name: str) -> bool:
        """Валидация названия компании"""
        if not name or len(name.strip()) < 2:
            return False
        # Проверка на наличие специальных символов
        if re.search(r'[<>"\']', name):
            return False
        return True

class AICrossValidator:
    """Класс для кросс-валидации результатов от двух AI"""
    
    def __init__(self, llm1, llm2):
        self.llm1 = llm1
        self.llm2 = llm2
    
    def validate_financial_data(self, data1: Dict[str, Any], data2: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация финансовых данных от двух AI"""
        validation_result = {
            "agreement_score": 0,
            "discrepancies": [],
            "recommendations": []
        }
        
        # Сравнение ключевых показателей
        for key in ["pe_ratio", "pb_ratio", "roe", "roa"]:
            if key in data1 and key in data2:
                val1, val2 = data1[key], data2[key]
                if val1 is not None and val2 is not None:
                    diff = abs(val1 - val2) / max(abs(val1), abs(val2)) * 100
                    if diff < 5:  # Различие менее 5%
                        validation_result["agreement_score"] += 25
                    else:
                        validation_result["discrepancies"].append({
                            "metric": key,
                            "value1": val1,
                            "value2": val2,
                            "difference_percent": diff
                        })
        
        # Формирование рекомендаций
        if validation_result["agreement_score"] >= 75:
            validation_result["recommendations"].append("Высокая согласованность данных")
        elif validation_result["agreement_score"] >= 50:
            validation_result["recommendations"].append("Умеренная согласованность, требуется дополнительная проверка")
        else:
            validation_result["recommendations"].append("Низкая согласованность, необходима перепроверка данных")
        
        return validation_result
    
    def validate_sentiment_analysis(self, sentiment1: Dict[str, Any], sentiment2: Dict[str, Any]) -> Dict[str, Any]:
        """Валидация анализа тональности"""
        validation_result = {
            "agreement": False,
            "confidence": 0,
            "recommendation": ""
        }
        
        if "sentiment" in sentiment1 and "sentiment" in sentiment2:
            if sentiment1["sentiment"] == sentiment2["sentiment"]:
                validation_result["agreement"] = True
                validation_result["confidence"] = 0.9
                validation_result["recommendation"] = "Согласованная оценка тональности"
            else:
                validation_result["confidence"] = 0.5
                validation_result["recommendation"] = "Противоречивые оценки тональности, требуется дополнительный анализ"
        
        return validation_result

# Создание экземпляров инструментов
cbrf_tool = AdvancedCBRFTool()
moex_tool = AdvancedMOEXTool()
news_tool = NewsAnalysisTool()
financial_tool = FinancialAnalysisTool()
data_validator = DataValidator() 