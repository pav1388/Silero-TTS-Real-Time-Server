WIP. [тестовый CPU сервер для win8+ x64 Всё-в-Одном](https://drive.google.com/file/d/1yBYmxAb43OktS8t_-VyOr0bL8weJpiJY/view?usp=sharing)

Обсуждение - [Luna Translator на форуме 4PDA](https://4pda.to/forum/index.php?showtopic=1100472)

# Silero TTS Server
Сервер синтеза речи на базе модели Silero TTS v5_5_ru.

Данная версия сервера является доработкой решения Виктора Шацкова (ноябрь 2025 г.) и включает оптимизации, поддержку транслитерации латиницы.

## Особенности

- **Качество голоса:** Используется модель Silero v5_5_ru. Голоса максимально выровнены по звучанию.
- **Поддержка языков:** Корректное чтение русского текста, числовых значений.
- **Транслитерация:** Добавлена автоматическая транслитерация латиницы. Возможность читать английский текст.
- **Производительность:** Оптимизирована текстовая обработка. Снижение качества генерируемого голоса при высокой нагрузке на CPU. Возможность вычислений на GPU Nvidia (CUDA) вместо CPU (по умолчанию используется CPU).

## Благодарности

- **HIllya51:** За LunaTranslator [Github](https://github.com/HIllya51/LunaTranslator)
- **Silero:** За доступные модели [Github](https://github.com/snakers4/silero-models), [Silero.ai](https://silero.ai/)
- **Штакет:** За базовый скрипт и исходные материалы [Youtube](https://www.youtube.com/watch?v=r7eI_gON3X0).
- **Виктор Шацков:** За адаптацию голосов [Youtube](https://www.youtube.com/watch?v=JGq7Xxvr5oI).

## Требования

- Python 3.11.9 x64

## Установка

1. Установите Python 3.11.9 с [официального сайта](https://www.python.org/downloads/release/python-3119/).

2. Установите необходимые зависимости:
   ```bash
   pip install numpy psutil bottle num2words
   ```

3. Установите PyTorch в зависимости от используемого оборудования:

   **Для CPU (размер пакета ~120 Мб):**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```

   **Для GPU Nvidia CUDA (размер пакета ~3 Гб, требует ~5 Гб места после распаковки):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. При первом запуске сервера разрешите доступ в брандмауэре ОС, если появится соответствующее уведомление.

## Подготовка моделей

1. Создайте папку `models` в директории рядом с файлом `silero-tts-for-luna-translator.py`.
2. Скачайте основную модель (v5_5_ru, ~140 Мб) и поместите её в папку `models`:
   - [Ссылка на модель v5_5_ru.pt](https://models.silero.ai/models/tts/ru/v5_5_ru.pt)
3. Другие доступные модели можно найти в [репозитории Silero](https://models.silero.ai/models/tts/ru/).

## Интеграция с Luna Translator

1. В разделе синтеза речи выберите оффлайн-модель: `vits-simpl-api`.
2. Откройте настройки (иконка шестеренки).
3. Укажите URL сервера: `http://127.0.0.1:5000`.
4. Остальные параметры удалите, нажав на крестики справа.
5. Чтобы настройка Pitch тоже передавалась на сервер (помимо Volume и Speed), то используйте исправление из папки `vitsSimpleAPI_fix`.

## Запуск

Запустите файл `silero-tts-for-luna-translator.py`.
