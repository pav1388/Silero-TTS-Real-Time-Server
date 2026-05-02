WIP. [тестовый CPU сервер для win7+ x64 Всё-в-Одном](https://drive.google.com/file/d/1yBYmxAb43OktS8t_-VyOr0bL8weJpiJY/view?usp=sharing) - обновляемая ссылка

Обсуждение - [Luna Translator на форуме 4PDA](https://4pda.to/forum/index.php?showtopic=1100472)

# Silero TTS Server
Сервер синтеза речи на базе модели Silero TTS v5_5_ru.

## Особенности

- **Качество голоса:** Используется модель Silero v5_5_ru. Голоса выровнены по звучанию.
- **Поддержка языков:** Корректное чтение русского текста, числовых значений.
- **Транслитерация:** Добавлена автоматическая транслитерация латиницы. Возможность читать английский текст.
- **Производительность:** Оптимизирована текстовая обработка. Снижение качества генерируемого голоса при высокой нагрузке на CPU. Возможность вычислений на GPU Nvidia (CUDA) вместо CPU (по умолчанию используется CPU).

## Благодарности

- **HIllya51:** За LunaTranslator [Github](https://github.com/HIllya51/LunaTranslator)
- **Silero:** За доступные модели [Github](https://github.com/snakers4/silero-models), [Silero.ai](https://silero.ai/)
- **Штакет:** За базовый скрипт и исходные материалы [Youtube](https://www.youtube.com/watch?v=r7eI_gON3X0).
- **Виктор Шацков:** За адаптацию голосов [Youtube](https://www.youtube.com/watch?v=JGq7Xxvr5oI).

## Требования

- Python 3.8.10 x64 для WIN7+
- Python 3.11.9 x64 для WIN8+

## Установка

1. Установите [Python 3.8.10](https://www.python.org/downloads/release/python-3810/) или [Python 3.11.9](https://www.python.org/downloads/release/python-3119/).

2. Установите необходимые зависимости:
   ```bash
   pip install numpy psutil bottle num2words signal
   ```

3. Установите PyTorch в зависимости от желаемого используемого оборудования:

   **Для CPU (размер пакета ~120-180 Мб):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   **Для GPU Nvidia CUDA (размер пакета ~2.8 Гб, поддержка CPU так же включена в пакет):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

4. При первом запуске сервера разрешите доступ в брандмауэре ОС, если появится уведомление.

## Подготовка моделей

1. Создайте папку `models` в директории рядом с файлом `silero-tts-for-luna-translator.py`.
2. Скачайте основную модель (v5_5_ru, ~140 Мб) и поместите её в папку `models`:
   - [Ссылка на модель v5_5_ru.pt](https://models.silero.ai/models/tts/ru/v5_5_ru.pt)
   - или программа сама скачает модель в случае её отсутствия при запуске
3. Другие доступные модели можно найти в [репозитории Silero](https://models.silero.ai/models/tts/ru/).

## Интеграция с Luna Translator

1. В разделе синтеза речи выберите оффлайн-модель: `vits-simpl-api`.
2. Откройте настройки (иконка шестеренки).
3. Укажите URL сервера: `http://127.0.0.1:23456`.
4. Остальные параметры удалите, нажав на крестики справа.
5. Если надо чтобы настройка Pitch тоже передавалась на сервер (помимо Speed), то используйте исправление из папки `vitsSimpleAPI_fix`.

## Запуск сервера

Запустите файл `silero-tts-rt-server.py` или bat-файл с выбором устройства.

## Тесты и отладка

Откройте `tts-rt-server-simple-tester.html`, простой HTML интерфейс имитирующий запросы от LunaTranslator.
