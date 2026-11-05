# Прогнозирование дефолта по кредитной карте

Учебный MLOps-проект: проверка данных, создание признаков, обучение модели,
учёт экспериментов, воспроизводимый DVC-пайплайн и API прогнозирования.

## Данные и модель

Используется Default of Credit Card Clients: 30 000 наблюдений.
Цель -  `default.payment.next.month`: `1` означает дефолт, `0` -  отсутствие дефолта.
`ID` исключается из входов модели.
Источник: [UCI Machine Learning Repository](https://doi.org/10.24432/C55S3H).

Разбиение стратифицированное: 24 000 строк для обучения и 6 000 для теста,
`random_state=42`. Great Expectations проверяет исходные данные по 72 правилам.

Pipeline объединяет создание признаков, заполнение пропусков, масштабирование
числовых признаков, One-Hot Encoding категорий и логистическую регрессию.
Добавляются шесть признаков: число месяцев с просрочкой, максимальная просрочка,
средний счёт, средний платёж, отношение последнего счёта к лимиту и возрастная группа.
Получается 29 признаков до кодирования категорий.

В предыдущих экспериментах сравнивались пять значений `C`:
`0.001`, `0.01`, `0.1`, `1`, `10`; также сохранён базовый запуск.
Итоговые параметры: `C=0.1`, порог классификации `0.3`.
Они зафиксированы в `params.json`; скрипт обучения воспроизводит выбранный вариант.
Тестовая выборка используется для итоговой оценки.

## Результаты на отложенном тесте

| Метрика | Значение |
| --- | ---: |
| ROC-AUC | 0.7613 |
| Precision | 0.5392 |
| Recall | 0.5026 |
| F1 | 0.5203 |

Матрица ошибок при пороге 0.3:

| Фактический класс | Прогноз: без дефолта | Прогноз: дефолт |
| --- | ---: | ---: |
| Без дефолта | 4103 | 570 |
| Дефолт | 660 | 667 |

## Основные файлы

| Путь | Назначение |
| --- | --- |
| `src/features/build_features.py` | Очистка категорий и создание признаков |
| `src/validation/validate_data.py` | Проверки Great Expectations |
| `src/data/prepare.py` | Валидация и разбиение данных |
| `src/models/pipeline.py` | Создание sklearn Pipeline |
| `src/models/train.py` | Обучение, оценка и запись в MLflow |
| `src/api/` | FastAPI, схема запроса и загрузка модели |
| `src/monitoring/monitor.py` | Имитация запросов и PSI распределения вероятностей |
| `tests/` | Проверки валидации, API и мониторинга |
| `params.json` | Параметры разбиения, модели и порога |
| `dvc.yaml`, `dvc.lock` | Этапы и зафиксированные зависимости DVC |
| `examples/request.json` | Готовый пример запроса |
| `notebooks/credit_default_scoring.ipynb` | Исследовательский ноутбук Colab с сохранёнными результатами |
| `.github/workflows/ci.yml` | Black, Flake8 и pytest в GitHub Actions |
| `Dockerfile` | Описание образа API |

## Установка

Команды выполняются из корня проекта. Исходная среда: Python 3.13.18.
Версии библиотек зафиксированы в файлах `requirements*.txt`.

```bash
python -m venv .venv
```

Активация Linux/macOS:

```bash
source .venv/bin/activate
```

Активация Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Полная среда разработки, обучения и тестирования:

```bash
python -m pip install -r requirements-dev.txt
```

Для запуска готового API достаточно:

```bash
python -m pip install -r requirements-serve.txt
```

## Восстановление данных и модели

Код и DVC-метаданные предназначены для Git. CSV, обученные модели, MLflow-хранилища
и DVC-кэш не включаются в Git. Полная резервная копия проекта содержит эти файлы.
Сам ноутбук Colab необходимо сохранять отдельно в формате `.ipynb`.

В Colab настроен локальный DVC remote в подключённой папке Google Диска:
`/content/drive/MyDrive/credit_default_project_dvc`.
Его настройка находится в `.dvc/config.local` и не публикуется в Git.

После подключения Google Диска и проверки наличия remote:

```bash
dvc remote add --local -d -f colab_drive /content/drive/MyDrive/credit_default_project_dvc
dvc pull
dvc status
```

Этот путь доступен только в соответствующей среде Colab. На другом компьютере
нужен доступ к тому же хранилищу с настройкой фактического локального пути либо
полная резервная копия. Один `git clone` не загружает данные и модель.

## Обучение и DVC

При наличии исходного `data/raw/UCI_Credit_Card.csv` этапы можно выполнить вручную:

```bash
python -m src.data.prepare
python -m src.models.train
```

Управление этапами через DVC:

```bash
dvc repro
dvc metrics show
dvc status
```

Если зависимости не изменились, `dvc repro` пропускает актуальные этапы.
Для принудительного повторного выполнения:

```bash
dvc repro --force
```

Модель сохраняется в `models/reproduced/model.joblib`, настройки — в соседнем
`model_config.json`, метрики — в `reports/reproduced_metrics.json`.

Новые эксперименты записываются в `experiments/reproduction/mlflow.db` и
`experiments/reproduction/artifacts`. Просмотр из корня проекта:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/reproduction/mlflow.db --port 5000
```

Исходные шесть запусков сохранены в `experiments/mlflow_checkpoint.zip` полной
резервной копии. Они не импортируются скриптом обучения автоматически.
При переносе SQLite-хранилища старые абсолютные пути артефактов могут потребовать
восстановления. API загружает модель напрямую из `models/reproduced`.

## Запуск API

Убедитесь, что в `models/reproduced` есть модель и её конфигурация.

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

- Проверка состояния: `http://127.0.0.1:8000/health`.
- Интерактивная документация: `http://127.0.0.1:8000/docs`.
- Прогноз: `POST /predict`.

Пример запроса для Linux/macOS:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  --data-binary @examples/request.json
```

API принимает 23 исходных признака, без `ID` и целевого столбца.
Ответ на сохранённый пример:

```json
{
  "default_probability": 0.7033040958180882,
  "prediction": 1,
  "threshold": 0.3
}
```

`prediction` равен `1`, если оценка вероятности не меньше порога.
Некорректные входные данные возвращают HTTP 422.

## Проверки и CI

```bash
python -m black --check src tests
python -m flake8 src tests
python -m pytest tests -v --tb=short
```

В Colab пройдены 10 тестов. Проверки API используют временную тестовую модель,
поэтому для CI не требуется скачивание полного датасета и итоговой модели.
GitHub Actions запускает проверки при push в `main`, pull request в `main`
и вручную. Первый запуск GitHub Actions успешно завершён 5 ноября 2026 года:
[результат проверок](https://github.com/Amalficode/credit-default-mlops/actions/runs/41939478841).

После добавления мониторинга успешно пройдены **30 тестов**, Black и Flake8:
[проверки версии с мониторингом](https://github.com/Amalficode/credit-default-mlops/actions/runs/42105354563).

## Docker

Для сборки нужны Docker Engine и оба файла из `models/reproduced`.
Команды выполняются из корня проекта при запущенном Docker Desktop.

```powershell
docker build --progress=plain -t credit-default-api:local .
docker run --rm -d --name credit-default-api -p 127.0.0.1:8000:8000 credit-default-api:local
```

После запуска приложения проверка из Windows PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health | ConvertTo-Json
$body = Get-Content .\examples\request.json -Raw
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body $body | ConvertTo-Json
```

Локальная проверка в Windows выполнена 5 ноября 2026 года:

- Образ `credit-default-api:local` успешно собран.
- Контейнер запущен; `/health` вернул `{"status":"ok","model_loaded":true}`.
- Для `examples/request.json` получена вероятность `0.7033040958180882`,
  класс `1` и порог `0.3` — результат совпал с проверенным примером в Colab.
- Запрос с `PAY_AMT1=-100` отклонён с HTTP `422`.

Документация работающего API: `http://127.0.0.1:8000/docs`.

Остановка контейнера:

```powershell
docker stop credit-default-api
```

Флаг `--rm` удаляет контейнер после остановки; собранный образ сохраняется.
Образ запускает Uvicorn от непривилегированного пользователя и содержит
healthcheck для `/health`.
Текущий CI выполняет проверки Python; сборка Docker в него не включена.

## Мониторинг PSI

Скрипт выбирает 200 клиентов из отложенного теста, отправляет их в API
и сравнивает распределение полученных вероятностей с оценками сохранённой
модели на обучающей выборке. Границы интервалов определяются только по эталону.
Сохраняются `reports/drift_report.json` и CSV с ответами API.

При работающем контейнере `credit-default-api` выполнить из полной папки проекта:

```powershell
$projectDir = (Get-Location).Path
docker run --rm --network container:credit-default-api --mount "type=bind,source=$projectDir,target=/workspace" --workdir /workspace credit-default-api:local python -m src.monitoring.monitor
```

PSI показывает сдвиг распределения оценок. Часть теста используется как учебная
имитация новых данных; значение PSI само по себе не измеряет качество модели.
[Описание расчёта, параметров и отчётов](docs/monitoring.md).
Локальный запуск через Docker 5 ноября 2026 года подтверждён выводом скрипта:
эталон — 24 000 обучающих строк; в API отправлены все 200 выбранных строк
из 6 000 тестовых. Получен **PSI = 0.039658** (округление до шести знаков),
условный уровень сдвига — низкий. Все прогнозы API совпали с сохранённой моделью.
Скрипт сообщил о сохранении JSON-отчёта и CSV с предсказаниями.
[Полный отчёт мониторинга](reports/drift_report.json): PSI = 0.03965831478603082;
расчёт проверен по сохранённым частотам интервалов.

## Статус проекта

Проверены в Colab: перенос модели из резервной копии, совпадение прогнозов,
валидация train/test, повторное обучение с зафиксированными параметрами,
работа API, 10 тестов, Black, Flake8 и актуальность DVC-пайплайна.

GitHub Actions: установка зависимостей, Black, Flake8 и pytest успешно прошли.
Docker: локально подтверждены сборка образа, запуск контейнера, ответ `/health`
и совпадение прогноза на сохранённом примере с результатом в Colab.
Мониторинг PSI реализован, покрыт тестами и проверен через Docker на 200 запросах;
PSI = 0.039658, условный уровень сдвига — низкий. Проект является учебным прототипом.

## Исследовательский ноутбук

[Открыть ноутбук](notebooks/credit_default_scoring.ipynb).

В ноутбуке сохранены анализ данных, эксперименты, результаты валидации, DVC и тестов.
Добавлены пояснение и заголовки разделов; исходные 98 ячеек кода и их результаты сохранены.
Ноутбук отражает ход разработки в Colab и содержит промежуточные диагностические ошибки,
после которых показано успешное восстановление модели. Он не проверен как единый запуск
«Выполнить всё» в новой сессии. Для воспроизводимого запуска итогового проекта используйте
модули, DVC-пайплайн и команды из этого README.
