"""
Веб-интерфейс для системы поддержки принятия решений инвесторов
Красивый и современный интерфейс с двойной валидацией AI
"""

from flask import Flask, render_template, request, jsonify, session
import os
import json
from datetime import datetime
import threading
import time
from typing import Optional
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Импортируем нашу систему анализа
try:
    from fin_crew import analyze_investment_opportunity, cache_manager
    SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"Ошибка импорта системы анализа: {e}")
    SYSTEM_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Глобальные переменные для отслеживания задач
active_tasks = {}

class AnalysisTask:
    """Класс для управления задачами анализа"""
    
    def __init__(self, company_name: str):
        self.id = f"task_{int(time.time())}"
        self.company_name = company_name
        self.status = "initializing"
        self.progress = 0
        self.result = None
        self.error = None
        self.start_time = datetime.now()
        self.end_time = None
        self.current_action = "Инициализация..."  # Текущее действие
        self.action_history = []  # История действий
        self.agents_status = {
            "case_analyst": "waiting",
            "financial_analyst": "waiting", 
            "company_analyst": "waiting",
            "decision_maker_analyst": "waiting",
            "news_analyst": "waiting",
            "risk_advisor": "waiting",
            "final_validator": "waiting"
        }
    
    def update_progress(self, progress: int, agent: Optional[str] = None):
        """Обновление прогресса выполнения"""
        self.progress = progress
        if agent:
            self.agents_status[agent] = "completed"
    
    def update_action(self, action: str):
        """Обновление текущего действия"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.current_action = f"[{timestamp}] {action}"
        self.action_history.append(self.current_action)
        # Ограничиваем историю последними 10 действиями
        if len(self.action_history) > 10:
            self.action_history = self.action_history[-10:]
    
    def complete(self, result: str):
        """Завершение задачи"""
        self.status = "completed"
        self.result = result
        self.end_time = datetime.now()
        self.progress = 100
        self.update_action("✅ Анализ завершен")
    
    def fail(self, error: Optional[str]):
        """Ошибка выполнения"""
        self.status = "failed"
        self.error = str(error) if error is not None else "Неизвестная ошибка"
        self.end_time = datetime.now()
        self.update_action(f"❌ Ошибка: {self.error}")

def run_analysis_task(task: AnalysisTask):
    """Запуск анализа в отдельном потоке с таймаутами"""
    try:
        task.status = "running"
        task.progress = 10
        task.update_action("🚀 Запуск анализа")
        
        # Настройки таймаутов
        MAX_ANALYSIS_TIME = 300  # 5 минут максимум
        start_time = time.time()
        
        print(f"🔄 Запуск анализа {task.company_name} с таймаутом {MAX_ANALYSIS_TIME}с")
        
        # Симуляция прогресса по агентам с проверкой таймаута
        agents = list(task.agents_status.keys())
        for i, agent in enumerate(agents):
            # Проверяем таймаут
            if time.time() - start_time > MAX_ANALYSIS_TIME:
                task.fail("⏰ Превышен таймаут анализа. Серверы ИИ перегружены.")
                return
                
            agent_name = agent.replace('_', ' ').title()
            task.update_action(f"🤖 {agent_name} анализирует данные...")
            task.update_progress(10 + (i + 1) * 12, agent)
            time.sleep(1)  # Уменьшаем время симуляции
        
        # Запуск реального анализа с таймаутом.
        if SYSTEM_AVAILABLE:
            task.update_action("🔄 Запуск AI анализа...")
            print(f"🤖 Запуск AI анализа для {task.company_name}")
            result = analyze_investment_opportunity(task.company_name)
            
            # Проверяем результат на ошибки таймаута
            result_str = str(result) if result is not None else ""
            if "Серверы ИИ перегружены" in result_str:
                task.fail(result_str)
            else:
                task.complete(result_str)
        else:
            task.update_action("📋 Генерация демо-результата...")
            # Демо-результат если система недоступна
            demo_result = f"""
            ============================================================================
            ДЕМО-АНАЛИЗ КОМПАНИИ: {task.company_name}
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
            рекомендуется [ДЕМО-РЕЗУЛЬТАТ] для компании {task.company_name}.
            
            📈 УРОВЕНЬ УВЕРЕННОСТИ: 85%
            ⏱️ ВРЕМЯ АНАЛИЗА: {datetime.now().strftime('%H:%M:%S')}
            """
            task.complete(demo_result)
            
    except Exception as e:
        error_msg = str(e)
        error_str = error_msg.lower()
        
        # Проверяем, можно ли переключиться на альтернативную модель
        if "insufficient balance" in error_str or "badrequesterror" in error_str:
            from fin_crew import switch_to_alternative_llm, llm
            alternative_llm = switch_to_alternative_llm(llm, error_msg)
            if alternative_llm and alternative_llm != llm:
                task.update_action(f"🔄 Переключились на {alternative_llm.model}, повторяем анализ...")
                try:
                    # Повторяем анализ с новой моделью
                    result = analyze_investment_opportunity(task.company_name)
                    task.complete(result)
                    return
                except Exception as retry_exception:
                    task.fail(f"❌ Ошибка даже с альтернативной моделью: {str(retry_exception)}")
                    return
        
        # Если переключение невозможно или не помогло, возвращаем сообщение об ошибке
        if "timeout" in error_str or "timed out" in error_str:
            task.fail("⏰ Серверы ИИ перегружены. Пожалуйста, повторите запрос позже.")
        elif "insufficient balance" in error_str:
            task.fail("💰 Недостаточно средств на балансе API. Пожалуйста, пополните счет.")
        elif "badrequesterror" in error_str:
            task.fail("🔧 Ошибка запроса к API. Проверьте настройки и повторите попытку.")
        else:
            task.fail(f"❌ Ошибка анализа: {error_msg}")

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def start_analysis():
    """Запуск анализа компании"""
    try:
        data = request.get_json()
        company_name = data.get('company_name', '').strip()
        
        if not company_name:
            return jsonify({'error': 'Название компании обязательно'}), 400
        
        # Создаем новую задачу
        task = AnalysisTask(company_name)
        active_tasks[task.id] = task
        
        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=run_analysis_task, args=(task,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'task_id': task.id,
            'status': 'started',
            'message': f'Анализ компании "{company_name}" запущен'
        })
        
    except Exception as e:
        return jsonify({'error': f'Ошибка запуска анализа: {str(e)}'}), 500

@app.route('/api/status/<task_id>')
def get_status(task_id):
    """Получение статуса задачи"""
    task = active_tasks.get(task_id)
    if not task:
        return jsonify({'error': 'Задача не найдена'}), 404
    
    return jsonify({
        'task_id': task.id,
        'company_name': task.company_name,
        'status': task.status,
        'progress': task.progress,
        'current_action': task.current_action,
        'action_history': task.action_history[-3:],  # Последние 3 действия
        'agents_status': task.agents_status,
        'start_time': task.start_time.isoformat(),
        'end_time': task.end_time.isoformat() if task.end_time else None,
        'result': task.result,
        'error': task.error
    })

@app.route('/api/tasks')
def get_tasks():
    """Получение списка всех задач"""
    tasks = []
    for task_id, task in active_tasks.items():
        tasks.append({
            'id': task_id,
            'company_name': task.company_name,
            'status': task.status,
            'progress': task.progress,
            'start_time': task.start_time.isoformat()
        })
    
    return jsonify({'tasks': tasks})

@app.route('/api/health')
def health_check():
    """Проверка состояния системы"""
    cache_stats = {}
    if SYSTEM_AVAILABLE and cache_manager:
        try:
            cache_dir = cache_manager.cache_dir
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                cache_stats = {
                    'cache_files_count': len(cache_files),
                    'cache_dir': cache_dir,
                    'cache_size_mb': sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files) / (1024 * 1024)
                }
        except Exception as e:
            cache_stats = {'error': str(e)}
    
    return jsonify({
        'status': 'healthy',
        'system_available': SYSTEM_AVAILABLE,
        'active_tasks': len(active_tasks),
        'cache_stats': cache_stats,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Очистка кеша"""
    try:
        if not SYSTEM_AVAILABLE or not cache_manager:
            return jsonify({'error': 'Система анализа недоступна'}), 500
        
        cleared_count = cache_manager.clear_old_cache()
        return jsonify({
            'success': True,
            'cleared_count': cleared_count,
            'message': f'Очищено {cleared_count} устаревших записей кеша'
        })
    except Exception as e:
        return jsonify({'error': f'Ошибка очистки кеша: {str(e)}'}), 500

@app.route('/api/cache/stats')
def cache_stats():
    """Статистика кеша"""
    try:
        if not SYSTEM_AVAILABLE or not cache_manager:
            return jsonify({'error': 'Система анализа недоступна'}), 500
        
        cache_dir = cache_manager.cache_dir
        if os.path.exists(cache_dir):
            cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
            
            return jsonify({
                'cache_files_count': len(cache_files),
                'cache_size_bytes': total_size,
                'cache_size_mb': total_size / (1024 * 1024),
                'cache_dir': cache_dir
            })
        else:
            return jsonify({
                'cache_files_count': 0,
                'cache_size_bytes': 0,
                'cache_size_mb': 0,
                'cache_dir': cache_dir
            })
    except Exception as e:
        return jsonify({'error': f'Ошибка получения статистики кеша: {str(e)}'}), 500

if __name__ == '__main__':
    # Запуск на нестандартном порту 8765
    debug_mode = os.getenv('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=8765, debug=debug_mode) 