"""
orchestrator.py — ядро системы: конечный автомат, переключающий роли агентов.

ПРАКТИКА 3. Что делает студент:
  1. описать роли в ROLES — минимум две, с разными зонами ответственности
  2. заполнить INTENT_TO_ROLE и TRANSITIONS — граф переходов автомата
  3. реализовать detect_intent() — определение интента пользователя

Готово и трогать не нужно: route(), handle(), подключение MCP и LLM.

Критерии приёмки:
  - ролей не меньше двух, у каждой описана своя зона ответственности
  - граф переходов задан структурой данных, а не цепочкой if по всему файлу
  - смена интента в диалоге переключает роль, это видно в логе
  - неизвестный интент не роняет процесс, а уходит в состояние по умолчанию
  - контекст при переходе не теряется: вторая роль видит найденное первой
  - вызов инструмента идёт через MCP-клиент, а не прямым импортом функции поиска

НЕ МЕНЯТЬ: сигнатуру handle(session_id, text) -> str. Это единственная точка
входа в систему, и её вызывают оба канала (app.py, vk_bot.py) и test_suite.py.

Порядок сдачи. На П3 сквозного ответа ещё не будет: build_messages() и клиент LLM
пишутся на П4. Проверяйте автомат отдельно — вызовом detect_intent() и route()
на наборе фраз. Сквозной ответ появится на П4, память подключится на П5.
"""

import logging

import guardrails
import memory
import prompts
from contracts import Chunk
from llm_client import build_client
from mcp_client import MCPTools

log = logging.getLogger(__name__)

BLOCKED_REPLY = "Запрос отклонён слоем безопасности. Переформулируйте вопрос по существу темы."


class Orchestrator:
    """Оркестратор ролей на конечном автомате."""

    ROLES: dict[str, dict] = {
        "analyst": {
            "description": (
                "находит нормы о промежуточной аттестации, пересдачах, "
                "академических задолженностях, переводах и отчислениях в базе знаний "
                "и отвечает со ссылками на документы; условия ситуации студента "
                "(курс, форму обучения, направление) сама не выясняет"
            ),
            "temperature": 0.2,
            "filters": {},
        },
        "clarifier": {
            "description": (
                "когда вопрос зависит от курса, формы обучения, направления "
                "подготовки или статуса задолженности и условия не указаны, "
                "задаёт уточняющий вопрос вместо вынесения решения по норме"
            ),
            "temperature": 0.3,
            "filters": {},
        },
        "verifier": {
            "description": (
                "перепроверяет утверждения и применимость норм по документам "
                "при запросе на проверку или спорном случае; если подтверждения "
                "в корпусе нет или норма не применима, меняет исход на отказ/уточнение"
            ),
            "temperature": 0.1,
            "filters": {},
        },
    }

    INITIAL_ROLE = "analyst"
    FALLBACK_ROLE = "analyst"

    TRANSITIONS: dict[str, set[str]] = {
        "analyst": {"analyst", "clarifier", "verifier"},
        "clarifier": {"analyst", "clarifier"},
        "verifier": {"analyst", "clarifier"},
    }

    INTENT_TO_ROLE: dict[str, str] = {
        "question": "analyst",
        "clarify": "clarifier",
        "verify": "verifier",
    }

    def __init__(self):
        self.tools = MCPTools()
        self.llm = build_client()
        memory.create_schema()
        self._state: dict[str, str] = {}
        self._context: dict[str, list[Chunk]] = {}
        self.usage: list[dict] = []

    def detect_intent(self, text: str) -> str:
        """Определить интент реплики без LLM и сложных словарей.

        На нераспознанной фразе возвращает "question".
        """
        cleaned = (text or "").strip().lower().replace("ё", "е")
        if not cleaned:
            return "question"

        # Короткая уточняющая реплика без вопросительного знака (например, "на 2 курсе очно")
        if len(cleaned.split()) <= 4 and "?" not in cleaned:
            return "clarify"

        # Запрос на перепроверку или сомнение в предыдущем выводе
        if "провер" in cleaned or "уверен" in cleaned:
            return "verify"

        return "question"

    # --- Готовая механика: менять не нужно ---

    def route(self, session_id: str, intent: str) -> str:
        """Перевести автомат в новое состояние и вернуть активную роль."""
        current = self._state.get(session_id, self.INITIAL_ROLE)
        target = self.INTENT_TO_ROLE.get(intent, self.FALLBACK_ROLE)

        allowed = self.TRANSITIONS.get(current, {self.FALLBACK_ROLE})
        rolled_back = target not in allowed
        if rolled_back:
            log.warning("Переход %s -> %s запрещён, откат в %s",
                        current, target, self.FALLBACK_ROLE)
            target = self.FALLBACK_ROLE

        if target != current:
            log.info("Сессия %s: роль %s -> %s (интент %s%s)",
                     session_id, current, target, intent, ", откат" if rolled_back else "")

        self._state[session_id] = target
        return target

    def handle(self, session_id: str, text: str) -> str:
        """Единственная точка входа. Оба канала и тесты идут сюда.

        Конвейер: guardrail -> память -> интент -> роль -> поиск через MCP ->
        сборка промпта -> LLM -> память -> ответ.
        """
        verdict = guardrails.check(text)
        if not verdict.allowed:
            guardrails.log_attempt(session_id, text, verdict)
            log.warning("Сессия %s: запрос заблокирован (%s)", session_id, verdict.reason)
            return BLOCKED_REPLY

        previous_role = self._state.get(session_id, self.INITIAL_ROLE)
        intent = self.detect_intent(text)
        role = self.route(session_id, intent)
        role_config = self.ROLES[role]

        filters = role_config.get("filters") or {}
        if filters:
            chunks = self.tools.search_filtered(text, filters)
        else:
            chunks = self.tools.search(text)

        if role != previous_role:
            known = {(c.source, c.page, c.text[:80]) for c in chunks}
            chunks = chunks + [c for c in self._context.get(session_id, [])
                               if (c.source, c.page, c.text[:80]) not in known]
        self._context[session_id] = chunks

        history = memory.window(session_id)
        memory.save(session_id, "user", text)

        messages = prompts.build_messages(role, text, chunks, history)

        prompt_size = self.llm.count_tokens("\n".join(m["content"] for m in messages))
        log.info("Сессия %s: роль %s, чанков %d, промпт ~%d токенов",
                 session_id, role, len(chunks), prompt_size)

        answer = self.llm.generate(messages, temperature=role_config["temperature"])

        self.usage.append({
            "session_id": session_id,
            "role": role,
            "prompt_tokens": prompt_size,
            "completion_tokens": self.llm.count_tokens(answer),
        })

        memory.save(session_id, "assistant", answer)
        return answer

    def reset(self, session_id: str) -> None:
        """Сбросить сессию: состояние автомата, контекст и историю."""
        self._state.pop(session_id, None)
        self._context.pop(session_id, None)
        memory.clear(session_id)

    def close(self) -> None:
        self.tools.close()