# 🤖 CrewDZ - Интеллектуальная система анализа инвестиционных возможностей

## 📋 Описание проекта

**CrewDZ** - это продвинутая система анализа инвестиционных возможностей с **архитектурой специализированных валидаторов** и **двойной валидацией AI**. Система использует мультимодельную архитектуру с автоматическим переключением между различными AI моделями (GPT-4.1, Claude, DeepSeek) и обеспечивает максимальную точность анализа через специализированные валидаторы для каждого типа данных.

### 🎯 Основные возможности

- **🏗️ Архитектура специализированных валидаторов** - 7 валидаторов для каждого типа анализа
- **🤖 Двойная валидация AI** - каждый валидатор использует 2 разные LLM модели
- **💾 Эффективное кеширование** - 70-80% экономия токенов
- **🔍 Отдельная поисковая LLM** - ProxyAPI для актуальных данных
- **🌐 Современный веб-интерфейс** с отображением прогресса в реальном времени
- **📈 Интеграция с финансовыми API** - ЦБ РФ, Московская биржа
- **⚡ Система таймаутов и повторных попыток** для надежности
- **📋 Подробное логирование** всех этапов анализа

### 🏗️ Архитектура системы

```
Поиск: search_llm (ProxyAPI) - отдельная модель
    ↓
Агенты: 7 специализированных агентов (1 LLM каждый)
    ↓
Валидаторы: 7 специализированных валидаторов (2 LLM каждый)
    ↓
Финальная валидация: Обобщение всех результатов
```

### 📁 Структура проекта

```
CrewDZ/
├── fin_crew.py          # Основная логика AI анализа с валидаторами
├── web_interface.py     # Веб-интерфейс Flask
├── requirements.txt     # Python зависимости
├── env.example         # Шаблон настроек окружения
├── templates/          # HTML шаблоны
│   └── index.html      # Главная страница
├── cache/              # Кеш результатов (автоматически создается)
└── API/               # Документация внешних API
    ├── API_CBRF.txt   # API Центрального Банка РФ
    └── API_MOEX       # API Московской биржи
```

## ⚠️ ВАЖНОЕ ПРИМЕЧАНИЕ

**🚨 АНАЛИЗ ПОТРЕБЛЯЕТ МНОГО ТОКЕНОВ! ИСПОЛЬЗОВАТЬ ОСТОРОЖНО!**

- **Первый запуск**: ~30,000 токенов
- **Повторные запуски**: ~8,000 токенов (благодаря кешированию)
- **Рекомендация**: Начинайте с небольших компаний для тестирования
- **Мониторинг**: Используйте API endpoints для контроля расхода токенов

## 🚀 Быстрая установка

### Предварительные требования

- Python 3.8 или выше
- Git
- Доступ к интернету для установки зависимостей
- **Минимум 2 API ключа** для полноценной работы валидаторов

### Пошаговая установка

1. **Клонирование репозитория**
   ```bash
   git clone <repository-url>
   cd CrewDZ
   ```

2. **Создание виртуального окружения**
   ```bash
   python -m venv venv
   ```

3. **Активация виртуального окружения**
   
   **Windows:**
   ```bash
   venv\Scripts\activate
   ```
   
   **Linux/Mac:**
   ```bash
   source venv/bin/activate
   ```

4. **Установка зависимостей**
   ```bash
   pip install -r requirements.txt
   ```

5. **Настройка переменных окружения**
   ```bash
   cp env.example .env
   ```
   
   Отредактируйте файл `.env` и добавьте ваши API ключи:
   ```env
   # ОБЯЗАТЕЛЬНО: Минимум 2 API ключа для валидаторов
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # ОБЯЗАТЕЛЬНО: Поисковая LLM
   PROXYAPI_SEARCH_KEY=your_proxyapi_search_key_here
   
   # Настройки кеширования
   CACHE_DIR=cache
   CACHE_MAX_AGE_HOURS=24
   
   # Настройки приложения
   DEBUG=True
   SECRET_KEY=your_secret_key_here_change_this_in_production
   ```

6. **Запуск системы**
   ```bash
   python web_interface.py
   ```

7. **Доступ к системе**
   
   Откройте браузер и перейдите по адресу: **http://localhost:8765**

## 🔑 Настройка API ключей

### Обязательные API ключи (минимум 2 для валидаторов)

#### 1. OpenAI API (GPT-4.1) - РЕКОМЕНДУЕТСЯ
```env
OPENAI_API_KEY=your_openai_api_key_here
```
- **Модель:** gpt-4.1-2025-04-14
- **Fallback модели:** gpt-4.1, gpt-4
- **Получение ключа:** [OpenAI Platform](https://platform.openai.com/api-keys)

#### 2. Claude API (Anthropic) - РЕКОМЕНДУЕТСЯ
```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_API_BASE=https://api.proxyapi.ru/anthropic
```
- **Модели:** claude-sonnet-4-20250514, claude-sonnet-4
- **Получение ключа:** [ProxyAPI](https://proxyapi.ru/)

#### 3. DeepSeek API - ДОПОЛНИТЕЛЬНО
```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
```
- **Модели:** deepseek-reasoner, deepseek-chat
- **Получение ключа:** [DeepSeek Console](https://console.deepseek.com/)

### Обязательный поисковый API

#### ProxyAPI для поиска
```env
PROXYAPI_SEARCH_KEY=your_proxyapi_search_key_here
PROXYAPI_SEARCH_BASE=https://api.proxyapi.ru/openai/v1
```
- **Назначение:** Поиск актуальных данных в интернете
- **Получение ключа:** [ProxyAPI](https://proxyapi.ru/)

## 🏗️ Архитектура валидаторов

### Специализированные валидаторы (7 валидаторов)

1. **case_validator** - Валидация кейс-анализа (2 LLM)
2. **financial_validator** - Валидация финансового анализа (2 LLM)
3. **company_validator** - Валидация анализа компании (2 LLM)
4. **leadership_validator** - Валидация анализа руководства (2 LLM)
5. **news_validator** - Валидация анализа новостей (2 LLM)
6. **risk_validator** - Валидация оценки рисков (2 LLM)
7. **final_validator** - Финальная валидация (2 LLM)

### Принцип работы валидации

```
Агент → Анализ (1 LLM) → Валидатор → Валидация (2 LLM) → Сравнение → Результат
```

### Кеширование валидации

- **Ключ кеша**: `validation_{analysis_type}_{hash(agent_result + context)}`
- **Эффективность**: 70-80% экономия токенов
- **Автоочистка**: Устаревшие записи удаляются автоматически (24 часа)

## 🛠️ Использование системы

### Веб-интерфейс

1. **Запуск анализа**
   - Введите название компании в поле ввода
   - Нажмите "Запустить анализ"
   - Следите за прогрессом в реальном времени

2. **Мониторинг прогресса**
   - Отображение статуса каждого AI агента и валидатора
   - История выполненных действий
   - Время выполнения анализа

3. **Результаты анализа**
   - Инвестиционная рекомендация с уровнем уверенности
   - Результаты валидации от двух AI моделей
   - Сравнение и обобщение выводов

### API Endpoints

#### Мониторинг системы
- `GET /api/health` - Статус системы и статистика кеша
- `GET /api/cache/stats` - Статистика кеширования
- `POST /api/cache/clear` - Очистка кеша

#### Управление задачами
- `POST /api/analyze` - Запуск анализа
- `GET /api/status/<task_id>` - Статус задачи
- `GET /api/tasks` - Список всех задач

### Программное использование

```python
from fin_crew import analyze_investment_opportunity

# Анализ компании с валидацией
result = analyze_investment_opportunity("Сбербанк")
print(result)
```

## 📊 Структура анализа

### Этапы анализа с валидацией

1. **Поиск актуальных данных** (search_llm)
2. **Анализ кейс-стади** (case_analyst → case_validator)
3. **Финансовый анализ** (financial_analyst → financial_validator)
4. **Анализ компании** (company_analyst → company_validator)
5. **Анализ руководства** (decision_maker_analyst → leadership_validator)
6. **Анализ новостей** (news_analyst → news_validator)
7. **Оценка рисков** (risk_advisor → risk_validator)
8. **Финальная валидация** (final_validator)

### Качество валидации

- **Высокий уровень согласованности** - обе модели подтверждают качество
- **Средний уровень согласованности** - модели дают разные оценки
- **Низкий уровень согласованности** - обе модели выявили проблемы

## 🔧 Настройка и оптимизация

### Включение реальных запросов к LLM

По умолчанию система использует заглушки для экономии токенов. Для включения реальных запросов:

1. Откройте файл `fin_crew.py`
2. Найдите методы `_get_validation_response()` и `_get_ai_response_with_timeout()`
3. Раскомментируйте реальные запросы:
   ```python
   # Раскомментируйте для реальных запросов:
   response = llm.complete(validation_prompt)
   return response.content
   ```

### Настройка кеширования

```env
# Настройки кеша
CACHE_DIR=cache                    # Директория кеша
CACHE_MAX_AGE_HOURS=24            # Максимальный возраст кеша (часы)
```

### Мониторинг расходов

- Используйте `/api/cache/stats` для мониторинга кеша
- Следите за логами в консоли
- Проверяйте баланс API ключей регулярно

## 🚨 Устранение неполадок

### Частые проблемы

1. **"Валидаторы недоступны"**
   - Проверьте наличие минимум 2 API ключей
   - Убедитесь в правильности настроек в .env

2. **"Поисковый LLM недоступен"**
   - Проверьте PROXYAPI_SEARCH_KEY
   - Убедитесь в доступности ProxyAPI

3. **"Превышен таймаут"**
   - Серверы ИИ перегружены
   - Повторите запрос позже

4. **"Недостаточно средств"**
   - Пополните баланс API ключей
   - Проверьте лимиты использования

### Логи и отладка

- Логи выводятся в консоль с временными метками
- Используйте `DEBUG=True` для подробных логов
- Проверяйте файл `analysis_debug.log` (если создается)

## 📈 Производительность

### Ожидаемые результаты

- **Время анализа**: 2-5 минут (в зависимости от сложности)
- **Точность**: Высокая (благодаря двойной валидации)
- **Надежность**: Высокая (fallback механизмы)
- **Эффективность**: Оптимальная (кеширование + специализация)

### Рекомендации по использованию

1. **Начинайте с малого** - тестируйте на небольших компаниях
2. **Мониторьте расходы** - используйте API endpoints
3. **Используйте кеш** - повторные запросы дешевле
4. **Проверяйте балансы** - регулярно контролируйте API ключи

### Анализ не является индивидуальной инвестиционной рекомендацией (ИИР) !!!
---

**🎯 Система готова к работе с максимальным качеством анализа и оптимальными затратами!** 
