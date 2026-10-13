# Changelog

All notable changes to Agentbox are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- User profiles define language, tone, verbosity, structure, code style, and
  constraints through versioned `/api/user-profiles` resources.
- Session tasks persist planning, execution, validation, pause state, steps,
  criteria, and transitions; the task strip remains visible beside the chat.
- Task control is available through `/api/sessions/{id}/task`,
  `/task/transition`, `/task/pause`, `/task/resume`, and `/task/steps/{index}`.
- Task tracking reports progress through `task`, `task_rejected`, and
  `task_error` SSE events without cancelling a completed agent response.
- Configuration drafts, a preset/profile library, quick chat controls,
  on-demand diagnostics, and reproducible saved experiments.
- Matrix runs accept `user_profile_ids`, fix the selected profile versions,
  and group results by profile.
- Sessions enforce versioned invariant sets before requests, answers, and
  mutating tools; conflicts, checker failures, costs, and versioned decisions
  remain visible after reload and in matrix runs.

### Changed

- Task transitions now bind human approval to exact plan, invariant, and task
  revisions; guarded stages cannot be forced, and blocked moves explain the safe path.
- Agent configuration profiles are now presets: `/api/profiles` is
  `/api/presets`, and matrix `profile_ids` are `preset_ids`.
- User profiles are included in agent, planning, input-policy, and memory-router
  calls; usage records retain the exact profile version hash.
- Explicit future preferences become reviewable profile proposals; accepting
  one creates a new version instead of silently changing shared sessions.
- Profile language remains authoritative when a question uses another language;
  an explicit instruction in the current message can still override it.
- Judge calls allow 2,000 output tokens with low reasoning effort to avoid empty
  verdicts from reasoning models.

### Removed

- `/plan/start`, `/plan/execute`, and `/plan/cancel`; task transitions replace
  the two-state planning mode.

## 2026-09-14

- Каталог моделей ограничен по цене: дороже `AGENTBOX_MAX_MODEL_PRICE` (5 $ за
  1M входных токенов) модель не предлагается в списках, но сохраняет цену и
  окно для уже открытых сессий и оценки матричного прогона.
- Выбор модели в форме больше не подменяет спеку: модель, которой нет в
  предложенном списке или в каталоге вовсе, остаётся видна как выбранная.
- Память разделена на слои: рабочая память задачи и долговременная память
  пространства складываются в payload независимо от стратегии контекста, а
  `context.mode=facts` снят и мигрирует в «окно + рабочий слой».
- SQLite v11: `memory_scopes` и `memory_versions` вместо `session_facts`,
  `sessions.memory_scope_id`, `ledger.kind=memory` вместо `facts`; спеки
  переадресованы под новую секцию `memory`.
- Ветка копирует рабочий слой на точке развилки и делит с родителем
  пространство долговременной памяти.
- Бюджеты `memory.working_max_tokens` и `memory.longterm_max_tokens` теперь
  ограничивают блок слоя в вызове: сверх бюджета в payload не уходят самые
  старые элементы. Хранилище не меняется, вытесненные ключи помечены «вне
  вызова» в панели памяти и перечислены во вкладке «контекст».
- Миграция `session_facts` кладёт элементы под видом из словаря рабочего слоя,
  поэтому мигрированный ключ можно править и переносить, а не только читать.
- Роутер памяти называет слой сам: ответ раскладывается по `working` и
  `longterm`, вид берётся из словаря слоя, чужой вид отклоняется с именем ключа
  вместо тихой записи, а ответ без имени слоя не принимается.
- Долговременная память пишется по `longterm_write`: `confirm` складывает
  кандидатов в `memory_proposals` и ждёт решения человека, `auto` пишет сразу,
  `manual` не тратит вызов роутера вовсе. Принять или отклонить — `POST
  /api/sessions/{id}/memory/proposals/{proposal_id}`.
- SQLite v12: таблица `memory_proposals` и `sessions.memory_covered_seq` —
  отметка прочитанного роутером принадлежит сессии, а не версии слоя.
- Ручная правка слоя проверяет вид по словарю слоя, а размер версии считается
  по её содержимому у любого происхождения.
- Оценка следующего вызова разбита по слоям: у роли, долговременной и рабочей
  памяти, сводки и хвоста своё число и доля вызова, а разница до измеренного
  запроса показана как разметка и схемы инструментов. Хвост больше не включает
  память в свою оценку.
- Вкладка «память»: три слоя карточками с местом хранения, версией, ценой блока
  в вызове и списком элементов; краткосрочный только показывается. У элемента
  видны маршрут и `seq` источника, значение и вид правятся на месте, элемент
  можно забыть или перенести в соседний слой. Кандидаты роутера принимаются и
  отклоняются здесь же, редактор памяти ушёл из вкладки «контекст».
- Кнопка «запомнить» на сообщении ленты пишет запись из него с явным выбором
  слоя и вида.
- Вид памяти принадлежит элементу, а не слою: `PUT
  /api/sessions/{id}/memory/{layer}` принимает `key → {value, kind}`, а
  точечные `POST .../items`, `DELETE .../items/{key}` и `POST
  .../items/{key}/move` меняют один ключ, не пересылая слой.
- Опустевший слой сохраняет версию, но в payload не уходит и не стоит токенов.
- Сессия вступает в пространство памяти при создании: новая сессия того же
  пространства открывается с его долговременной памятью, а карточка слоя
  показывает пространство и число сессий в нём.
- «Завершить задачу» (`POST /api/sessions/{id}/memory/working/finish`):
  отмеченные элементы уезжают в долговременный слой переносом, рабочий слой
  получает пустую версию, отметка прочитанного роутером остаётся на месте.
- SSE памяти разделён на явные события: `memory` публикует записанную версию,
  `memory_proposals` — кандидатов для подтверждения, `memory_rejected` — ключи
  с чужим видом, `memory_error` — ошибку роутера без отмены ответа агента.
- Контракт дешёвого роутера закрепляет именованный верхний уровень даже для
  одного или пустого слоя, а служебный запас reasoning увеличен для ответов
  `gpt-5-nano`, превышающих 1024 reasoning-токена.
- Проба P3 дня 11 выталкивает эхо P1 из краткосрочного окна перед проверкой
  завершённой задачи, поэтому не принимает хвост диалога за рабочую память.

## 2026-09-12

- Черновик новой сессии больше не пишется в SQLite: сессия создаётся только
  вместе с первым сообщением, а старые пустые записи не восстанавливаются.
- Подсказки настроек заменены на мгновенные интерфейсные тултипы вместо
  задерживаемых браузером `title`.

## 2026-09-11

- Исправлена память фактов: обновление модели накладывается на предыдущую
  версию вместо замены, нестроковые значения приводятся к тексту, а служебные
  вызовы сводки и фактов получают запас `max_output_tokens` на reasoning вместо
  пустого ответа.
- Кнопка «ветка отсюда» спрашивает название ветки, как и checkpoint.
- Добавлен реестр LLM-провайдеров: клиент, каталог и capabilities выбираются по
  живой `AgentSpec`, Zen владеет своим transport routing и переменными доступа,
  а каталог, интерфейс и матричные прогоны сохраняют provider identity.
- Добавлены checkpoints и независимые ветки диалога: снимок истории и памяти
  создаётся от выбранного сообщения, расход начинается с нуля, а сайдбар
  показывает продолжения деревом.
- Добавлена версионируемая память фактов: отдельное извлечение и учёт после
  ответа, payload с последними сообщениями, ручная таблица правок и перенос
  существующего диалога при включении стратегии `facts`.
- Контекст вынесен из служебной вкладки в сам диалог: ответы получили метрики
  ledger, лента — границы окна и сводки, композер — состояние следующего
  вызова, а правая вкладка показывает полный состав контекста и рост по ходам.
- Панель агента переведена на живую настройку открытой сессии с автосохранением,
  условными полями и профильным подвалом; навигация слева теперь содержит только
  сессии и переход к матричному прогону.
- Контекст вынесен в плагины `none`, `window`, `tokens`, `summary`: ядро
  сохраняет полную стенограмму, применяет переполнение к выбранному срезу и
  публикует наблюдаемый `ContextView` через SSE и API сессии.
- Конфигурация агента переведена на вложенную `AgentSpec` v2 с миграцией
  SQLite, версионной привязкой ответов в ledger и безопасной сменой спеки
  открытой сессии через API.
- Восстановлено открытие старых диалогов, в конфигурации которых лимит ответа
  был сохранён как `null`; при чтении используется текущее значение по умолчанию.
- Интерфейс разделён на независимые зоны сессий, чата и настроек; матричный
  прогон вынесен на отдельный экран, добавлена адаптивная навигация.
- Каталог расширен до вызываемых моделей Zen с транспортами `/responses` и
  `/chat/completions`, сведениями о цене, контекстном окне и поддерживаемых
  параметрах.
- Добавлены версионируемые профили настроек для сессий и матричных прогонов.
- Добавлены оценка контекста до запроса, фактический учёт токенов и стоимости,
  режимы обрезки и отказа при переполнении, панель расхода в интерфейсе и отчёт
  задания дня 8.
- Добавлены версионируемые сводки старой истории, отдельный учёт стоимости
  сжатия, состояние сводки в интерфейсе и сравнительный отчёт задания дня 9.
