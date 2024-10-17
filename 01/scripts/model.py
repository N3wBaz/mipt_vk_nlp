import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

class Model(nn.Module):
    """
    Класс Model представляет собой нейронную сеть на основе LSTM для обработки последовательностей, таких как текст.
    Она состоит из слоев эмбеддингов, LSTM и линейного слоя для получения логитов, соответствующих размерам словаря.

    Аргументы:
        vocab_size (int): Размер словаря (количество уникальных слов).
        emb_size (int, необязательный): Размерность эмбеддингов. По умолчанию 128.
        num_layers (int, необязательный): Количество слоев в LSTM. По умолчанию 1.
        hidden_size (int, необязательный): Размерность скрытого состояния LSTM. По умолчанию 256.
        dropout (float, необязательный): Вероятность отключения нейронов (dropout) между слоями LSTM. По умолчанию 0.0.
    """
    def __init__(
            self,
            vocab_size: int,
            emb_size: int = 128,
            num_layers: int = 1,
            hidden_size: int = 256,
            dropout: float = 0.0
    ):
        super().__init__()
        # Слой эмбеддингов: размер словаря vocab_size, размер эмбеддингов emb_size
        self.embeddings = nn.Embedding(vocab_size, emb_size)
        
        # LSTM: входной размер = размер эмбеддингов, скрытый размер = hidden_size, количество слоев = num_layers
        self.lstm = nn.LSTM(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,  # Дропаут применяем только если больше одного слоя
            batch_first=True  # Для удобства, чтобы батч был первым измерением (batch_size, seq_len, emb_size)
        )
        
        # Линейный слой для преобразования скрытых состояний в логиты размером vocab_size
        self.logits = nn.Linear(hidden_size, vocab_size)

    def forward(
            self,
            x: Tensor,
            hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Проводит прямое распространение через сеть.

        Аргументы:
            x (Tensor): Входные данные (индексы слов) размером (batch_size, seq_len).
            hx (Optional[Tuple[Tensor, Tensor]]): Начальные скрытые состояния (h_n, c_n) для LSTM. По умолчанию None.

        Возвращает:
            Tuple[Tensor, Tuple[Tensor, Tensor]]:
                - Логиты (предсказания для каждого слова в последовательности) размером (batch_size, seq_len, vocab_size).
                - Пара скрытых состояний (h_n, c_n), где h_n и c_n — это последние скрытые и клеточные состояния LSTM.
        """
        # Проходим через слой эмбеддингов
        x = self.embeddings(x)  # Результат: (batch_size, seq_len, emb_size)
        
        # Пропускаем данные через LSTM
        x, (h_n, c_n) = self.lstm(x, hx)  # Результат: (batch_size, seq_len, hidden_size)
        
        # Преобразуем скрытые состояния в логиты через линейный слой
        logits = self.logits(x)  # Результат: (batch_size, seq_len, vocab_size)
        
        return logits, (h_n, c_n)
