# Постановка задания

## Цель

Реализовать минимальную multimodal language model для задач визуально-математического рассуждения. Модель получает изображение и вопрос, а возвращает ответ или вариант ответа.

Примеры задач:

- найти значение функции по графику;
- определить длину стороны в геометрической схеме;
- прочитать величину из столбчатой диаграммы;
- выбрать правильный ответ по визуальной формуле или таблице.

## Базовая архитектура

Рекомендуемая архитектура:

```text
image → ViT encoder → trainable adapter → visual embeddings
text question → tokenizer → text embeddings
visual embeddings + text embeddings → frozen/instruct LLM → answer
```

В обязательной части можно считать, что:

```text
vision encoder заморожен
LLM заморожен
обучается только adapter
LoRA используется только в Track C или как бонус
```

## Обязательные компоненты

### 1. `dataset.py`

Нужно реализовать загрузку примеров из `manifest.jsonl`.

Ожидаемый формат примера:

```json
{
  "id": "toy_train_000",
  "split": "train",
  "image": "images/line_plot_0.png",
  "question": "На графике дана прямая. Чему равен y при x=2?",
  "options": ["A) 3", "B) 5", "C) 6", "D) 7"],
  "answer": "B",
  "subject": "algebra"
}
```

### 2. `processor.py`

Нужно реализовать:

- приведение изображения к RGB;
- resize/crop/pad до `image_size`;
- разбиение на `num_tiles` тайлов;
- нормализацию изображения;
- построение prompt с visual special tokens;
- токенизацию question/options/answer;
- `labels`, где prompt замаскирован `IGNORE_INDEX`, а loss считается только на ответе;
- `collate_fn` для батча.

### 3. `model.py`

Нужно реализовать:

- adapter из hidden states vision encoder в размерность LLM embeddings;
- функцию вставки visual embeddings на позиции `<image>`-токенов;
- forward pass с loss;
- generate/inference wrapper;
- корректную заморозку vision encoder и LLM в Track A/B.

### 4. `train.py`

Нужно реализовать:

- загрузку YAML-конфига;
- создание датасета, processor, модели, optimizer;
- gradient accumulation;
- `fast_train` режим для smoke-тестов;
- сохранение adapter/checkpoint;
- проверку, что loss конечный.

### 5. `benchmark.py`

Нужно реализовать:

- построение benchmark prompt;
- запуск `generate`;
- извлечение ответа `A/B/C/D` или normalised text answer;
- подсчёт accuracy по subject и overall.

## Ограничения

- В public-тестах не должно быть интернета.
- Нельзя требовать у всех студентов полноценного GPU-обучения.
- Нельзя хранить hidden labels в student-template.
- Все параметры должны задаваться через YAML-конфиги.
- Код должен быть воспроизводимым: seed, config, логирование.

## Профильные источники для математиков

Рекомендуемые источники для преподавателя:

- MathVista — visual mathematical reasoning benchmark;
- MAVIS — mathematical visual instruction tuning datasets;
- MATH-Vision — visual math tasks from real math competitions;
- We-Math — benchmark with hierarchical visual mathematical reasoning concepts.

В student-template включён только toy-набор. Реальные источники подключаются преподавателем отдельно с учётом лицензий и правил курса.
