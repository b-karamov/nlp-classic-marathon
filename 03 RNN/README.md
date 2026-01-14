# RNN (Elman)

## Архитектура
Классическая Elman RNN для предсказания следующего символа.

- Вход: one-hot $x_t \in \mathbb{R}^V$
- Скрытое состояние: $h_t \in \mathbb{R}^H$
- Выход: логиты $o_t \in \mathbb{R}^V$

## Диаграмма
```mermaid
flowchart LR
    classDef input fill:#f7f3e9,stroke:#c8bda8,color:#3b2f2f;
    classDef core fill:#e3f2fd,stroke:#64b5f6,color:#0d47a1;
    classDef state fill:#e8f5e9,stroke:#81c784,color:#1b5e20;
    classDef output fill:#fff3e0,stroke:#ffb74d,color:#e65100;

    subgraph Legend
        direction TB
        L1["input"]:::input
        L2["state"]:::state
        L3["transform"]:::core
        L4["output"]:::output
    end

    subgraph Input
        direction TB
        X["x_t"]:::input
        H_prev["h_{t-1}"]:::state
    end

    subgraph Cell["RNN step t"]
        direction TB
        CellAct["tanh"]:::core
        H["h_t"]:::state
        CellAct --> H
    end

    subgraph Output
        direction TB
        O["Linear"]:::core
        P["softmax"]:::output
        O --> P
    end

    X --> CellAct
    H_prev --> CellAct
    H --> O
```

## Теория
RNN передает состояние по времени, что позволяет моделировать зависимости
между шагами последовательности. Обучение идет через BPTT и softmax.

## Формулы
**Прямой проход (шаг $t$)**
$$
a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h,\quad
h_t = \tanh(a_t)
$$
$$
o_t = W_{hy} h_t + b_y,\quad
p_t = \text{softmax}(o_t)
$$

**Лосс (по времени)**
$$
L = -\frac{1}{T}\sum_t \sum_i y_{t,i}\log p_{t,i}
$$

**BPTT (кратко)**
$$
d o_t = p_t - y_t,\quad
d h_t = W_{hy}^T d o_t + W_{hh}^T d h_{t+1}
$$
$$
d a_t = d h_t \odot (1 - h_t^2)
$$

## Применимые задачи
- Предсказание следующего символа/токена
- Простые последовательностные задачи
- Бейзлайн для NLP

## Плюсы
- Простая и быстрая модель
- Учитывает порядок и контекст
- Хороший стартовый бейзлайн

## Минусы
- Затухание/взрыв градиентов на длинных последовательностях
- Ограниченная память о дальнем контексте
