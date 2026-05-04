WIP. [CPU сервер для win7+ x64 Всё-в-Одном](https://drive.google.com/file/d/1yBYmxAb43OktS8t_-VyOr0bL8weJpiJY/view?usp=sharing) - обновляемая ссылка


Обсуждение - [Luna Translator на форуме 4PDA](https://4pda.to/forum/index.php?showtopic=1100472)

Исходный код - [GitHub](https://github.com/pav1388/Silero-TTS-Real-Time-Server/)

# Silero TTS Real-Time Server

![](scripts/title.png)

Сервер синтеза речи на базе модели `Silero 5.5 ru`.

## Особенности

- **Потоковый режим генерации** — начинает озвучивать сразу, независимо от длины текста.
- **Качество голоса** — используется модель Silero v5_5_ru.
- **Расстановка ударений и ё-фикация** — обрабатываются моделью (корректное чтение русского текста).
- **Чтение числовых значений и транслитерация латиницы** — обрабатываются сервером.
- **Производительность** — снижение качества генерируемого голоса при высокой нагрузке на CPU.
- **Вычисления на GPU Nvidia CUDA** (по умолчанию CPU, так как он быстрее на коротких репликах).
- **Тестирование** — есть простой HTML5 клиент tts-rt-server-simple-tester.html.
- **Адаптация под свой проект** — достаточно переписать в классе HTTPServer функцию _setup_routes().

## Благодарности

- **HIllya51:** За LunaTranslator [Github](https://github.com/HIllya51/LunaTranslator)
- **Silero:** За доступные модели [Github](https://github.com/snakers4/silero-models), [Silero.ai](https://silero.ai/)
- **Штакет:** За идею и исходные материалы [Youtube](https://www.youtube.com/watch?v=r7eI_gON3X0).
- **Виктор Шацков:** За идею адаптации голосов [Youtube](https://www.youtube.com/watch?v=JGq7Xxvr5oI).

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

## Подготовка модели

1. Создайте папку `models` в директории рядом с файлом `silero-tts-for-luna-translator.py`.
2. Скачайте основную модель (v5_5_ru, ~140 Мб) и поместите её в папку `models`:
   - [Ссылка на модель v5_5_ru.pt](https://models.silero.ai/models/tts/ru/v5_5_ru.pt)
   - или программа сама скачает модель в случае её отсутствия при запуске
   - другие доступные модели можно найти в [репозитории Silero](https://models.silero.ai/models/tts/ru/).

## Интеграция с Luna Translator (проверялось на версиях около v10.15.7.12)

1. Поместите файл `LunaTranslator\selfbuild_tts.py` в `\LunaTranslator_x64_win10\userconfig\selfbuild_tts.py`.
2. Запустите сервер (после запуска можно просто свернуть консольное окно), затем запустите Luna.
3. В разделе настроек Синтеза речи включите онлайн-модель: `Custom ("Настройка" в русском переводе)`.
4. Выберите голос, скорость, высоту тона.
5. Пользуйтесь как обычно.
6. Примечание: В файле `selfbuild_tts.py` предусмотрены индивидуальные начальные подстройки голосов.

## Запуск сервера

Запустите файл `silero-tts-rt-server.py` или bat-файл с выбором устройства из папки `scripts`.

## Тесты и отладка

Запустите скрипт с `debug` аргументом из папки `scripts`.

Откройте `tts-rt-server-simple-tester.html`, простой HTML5 интерфейс имитирующий запросы от LunaTranslator.
