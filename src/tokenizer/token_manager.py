"""Управление триггер-токенами для персон."""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenValidator:
    """Валидатор токенов."""
    
    @staticmethod
    def validate_token_integrity(tokenizer: AutoTokenizer, token: str) -> Tuple[bool, List[int]]:
        """
        Проверить целостность токена.
        
        Returns:
            (is_valid, token_ids): True если токен не разбивается на части
        """
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        # Токен должен декодироваться обратно в исходную строку
        is_valid = decoded.strip() == token.strip()
        
        if not is_valid:
            logger.warning(f"Токен '{token}' распадается на части: {decoded}")
        
        return is_valid, token_ids
    
    @staticmethod
    def check_token_uniqueness(tokenizer: AutoTokenizer, token: str, existing_tokens: List[str]) -> bool:
        """Проверить уникальность токена."""
        if token in existing_tokens:
            return False
        
        # Проверить, что токен не является подстрокой других токенов
        for existing in existing_tokens:
            if token in existing or existing in token:
                logger.warning(f"Токен '{token}' конфликтует с '{existing}'")
                return False
        
        return True


class TokenManager:
    """Менеджер триггер-токенов."""
    
    def __init__(self, tokenizer_path: str, persons_dir: Path):
        """
        Инициализировать менеджер токенов.
        
        Args:
            tokenizer_path: Путь к токенайзеру модели
            persons_dir: Директория с данными персон
        """
        self.tokenizer_path = tokenizer_path
        self.persons_dir = Path(persons_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.validator = TokenValidator()
        
        # Загрузить существующие токены
        self._load_existing_tokens()
    
    def _load_existing_tokens(self):
        """Загрузить существующие токены из манифестов персон."""
        self.persons_dir.mkdir(parents=True, exist_ok=True)
        self.existing_tokens = {}
        
        for person_dir in self.persons_dir.iterdir():
            if person_dir.is_dir():
                manifest_path = person_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, 'r', encoding='utf-8') as f:
                            manifest = json.load(f)
                            token = manifest.get('trigger_token')
                            if token:
                                self.existing_tokens[person_dir.name] = token
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить манифест для {person_dir.name}: {e}")
    
    def create_token(self, person_name: str, suggested_token: Optional[str] = None) -> str:
        """
        Создать новый токен для персоны.
        
        Args:
            person_name: Имя персоны
            suggested_token: Предложенный токен (например, "<qwn_alex>")
        
        Returns:
            Созданный токен
        """
        if person_name in self.existing_tokens:
            logger.info(f"Токен для персоны '{person_name}' уже существует: {self.existing_tokens[person_name]}")
            return self.existing_tokens[person_name]
        
        # Генерировать токен, если не предложен
        if not suggested_token:
            suggested_token = f"<qwn_{person_name.lower()}>"
        
        # Проверить уникальность
        existing_token_list = list(self.existing_tokens.values())
        if not self.validator.check_token_uniqueness(self.tokenizer, suggested_token, existing_token_list):
            # Попробовать альтернативный вариант
            counter = 1
            while True:
                alternative = f"{suggested_token}_{counter}"
                if self.validator.check_token_uniqueness(self.tokenizer, alternative, existing_token_list):
                    suggested_token = alternative
                    break
                counter += 1
                if counter > 100:
                    raise ValueError(f"Не удалось создать уникальный токен для {person_name}")
        
        # Проверить целостность
        is_valid, token_ids = self.validator.validate_token_integrity(self.tokenizer, suggested_token)
        
        if not is_valid:
            logger.warning(f"Токен '{suggested_token}' распадается на части. Попытка добавить в токенайзер...")
            # Попытаться добавить токен в токенайзер
            try:
                self.tokenizer.add_tokens([suggested_token])
                is_valid, token_ids = self.validator.validate_token_integrity(self.tokenizer, suggested_token)
            except Exception as e:
                logger.error(f"Не удалось добавить токен в токенайзер: {e}")
        
        if not is_valid:
            raise ValueError(f"Токен '{suggested_token}' не может быть использован: распадается на части")
        
        logger.info(f"Создан токен для '{person_name}': {suggested_token} (IDs: {token_ids})")
        return suggested_token
    
    def add_token_to_tokenizer(self, token: str, init_from: str = "person") -> bool:
        """
        Добавить токен в токенайзер.
        
        Args:
            token: Токен для добавления
            init_from: Слово для инициализации эмбеддинга ("person", "portrait")
        
        Returns:
            True если успешно добавлен
        """
        try:
            # Получить ID для инициализации
            init_ids = self.tokenizer.encode(init_from, add_special_tokens=False)
            if not init_ids:
                init_ids = self.tokenizer.encode("person", add_special_tokens=False)
            
            # Добавить токен
            num_added = self.tokenizer.add_tokens([token])
            
            if num_added > 0:
                logger.info(f"Токен '{token}' добавлен в токенайзер")
                return True
            else:
                logger.warning(f"Токен '{token}' уже существует в токенайзере")
                return False
        except Exception as e:
            logger.error(f"Ошибка при добавлении токена: {e}")
            return False
    
    def validate_runtime_token(self, token: str) -> Tuple[bool, List[int]]:
        """
        Runtime-проверка токена при старте графа.
        
        Returns:
            (is_valid, token_ids)
        """
        is_valid, token_ids = self.validator.validate_token_integrity(self.tokenizer, token)
        
        if not is_valid:
            logger.error(f"RUNTIME CHECK FAILED: Токен '{token}' распался на части!")
            logger.error(f"Используйте tokenizer.json из набора обучения")
        
        return is_valid, token_ids
    
    def get_token(self, person_name: str) -> Optional[str]:
        """Получить токен для персоны."""
        return self.existing_tokens.get(person_name)
    
    def save_tokenizer(self, output_path: Path):
        """Сохранить токенайзер с добавленными токенами."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Токенайзер сохранён в {output_path}")

