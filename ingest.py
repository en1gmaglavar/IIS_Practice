"""  
ingest.py — сбор исходных данных и наполнение векторного индекса.  
  
ПРАКТИКА 2. Что делает студент:  
  1. clean_text()  — предварительная очистка: переносы, колонтитулы, мусорные пробелы  
  2. chunk_text()  — нарезка на чанки фиксированного размера с перекрытием  
  3. detect_category() — рубрикация чанка; словарь рубрик задаёте вы под свой домен  
  
  4. черновик gold_dataset.json — 5 вопросов с указанием файлов-источников  
  
Готово и трогать не нужно: обход data/raw/, чтение PDF и TXT, запись в ChromaDB,  
замер hit-rate в evaluate_retrieval().  
  
Каждый запуск пересобирает индекс с нуля — коллекция очищается перед наполнением.  
Поэтому при подборе параметров достаточно поправить CHUNK_SIZE или CHUNK_OVERLAP  
в config.py и запустить ingest.py снова: удалять data/chroma/ руками не нужно.  
  
Критерии приёмки:  
  - python ingest.py отрабатывает без ошибок, коллекция непуста  
  - у каждого чанка заполнены source, page и category; пустых category нет  
  - поиск с фильтром по category возвращает чанки только этой категории  
  - в gold_dataset.json не меньше 5 вопросов; expected_sources заполнено у всех,  
    кроме вопроса типа out_of_scope, у которого оно пусто по определению  
  - показаны не менее двух замеров с разными CHUNK_SIZE / CHUNK_OVERLAP,  
    в каждом сняты hit-rate@k и hit-rate@1, и выбор итоговых значений объяснён  
    числом из таблицы, а не вкусом  
  - провизорное число метрики из П1 отработано в README: изменено с объяснением  
    либо оставлено как цель с названной причиной промаха  
  - корпус закоммичен в data/raw/, имена файлов совпадают со списком источников  
    в README.md  
  - CHUNK_SIZE и CHUNK_OVERLAP берутся из config.py, а не зашиты в код  
  
Про метрику. hit-rate@k отвечает на один вопрос: попал ли фрагмент с ответом  
в первые k результатов. Считается механически, сверкой поля source с ожидаемым —  
ни модели, ни судьи для этого не нужно, поэтому мерить можно уже здесь, на П2.  
Не путайте её с Context Precision из триады RAG: та оценивает, насколько найденное  
относится к делу, её считает LLM-судья, и появляется она только на П6. В плане  
дисциплины обе назывались «точностью извлечения контекста», хотя это разные вещи.  
  
Запуск:  
    python ingest.py          # проиндексировать и сразу замерить  
    python ingest.py --eval   # только замер, без переиндексации  
"""  
  
import hashlib  
import re  
import sys  
import unicodedata  
import warnings  
from pathlib import Path  
  
sys.stdout.reconfigure(encoding="utf-8")  
import config  
  
# Подавляем предупреждение pypdf о некорректных указателях в PDF.  
# Некоторые файлы имеют дефекты структуры, но текст извлекается нормально.  
try:  
    from pypdf.errors import PdfReadWarning  
    warnings.filterwarnings("ignore", category=PdfReadWarning)  
except ImportError:  
    pass  
  
  
# --- Читатели форматов: готовы ---  
  
def read_txt(path: Path) -> list[tuple[int, str]]:  
    """Возвращает [(номер страницы, текст)]. У TXT страниц нет, поэтому страница 0."""  
    return [(0, path.read_text(encoding="utf-8", errors="ignore"))]  
  
  
def read_pdf(path: Path) -> list[tuple[int, str]]:  
    """Возвращает [(номер страницы, текст)] — постранично, чтобы метаданное page было честным."""  
    from pypdf import PdfReader  
  
    reader = PdfReader(str(path))  
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]  
  
  
READERS = {".txt": read_txt, ".md": read_txt, ".pdf": read_pdf}  
  
  
# --- Ваша часть (П2) ---  
  
def clean_text(text: str) -> str:  
    r"""Очистить сырой текст перед нарезкой.  
  
    Убираем разрывы слов по переносу строки, повторяющиеся колонтитулы,  
    номера страниц отдельной строкой, цепочки пробелов и пустых строк.  
    """  
    if not text:  
        return ""  
  
    # Неразрывные пробелы, табуляции и «мягкие» переносы  
    text = text.replace("\xa0", " ").replace("\u00ad", "")  
    text = text.replace("\r\n", "\n").replace("\r", "\n")  
  
    # Склейка слов, разорванных переносом с дефисом на границе строки  
    text = re.sub(r"([А-ЯЁA-Za-z])-\s*\n\s*([а-яёa-z])", r"\1\2", text)  
  
    # Перенос строки внутри предложения (если следующая строка начинается со строчной буквы)  
    text = re.sub(r"(?<=[а-яёa-z,;])\n+(?=[а-яёa-z])", " ", text)  
  
    # Одиночные номера страниц отдельной строкой  
    text = re.sub(r"\n[ \t]*\d{1,3}[ \t]*(?=\n|$)", "\n", text)  
  
    # Типовые служебные вставки при копировании с правовых/нормативных страниц  
    noise_patterns = [  
        r"(?im)^[ \t]*информация об изменениях[ \t]*$",  
        r"(?im)^[ \t]*судебная практика[ \t]*$",  
        r"(?im)^[ \t]*редакция с изменениями[^\n]*$",  
    ]  
    for pat in noise_patterns:  
        text = re.sub(pat, "", text)  
  
    # Нормализация пробелов и пустых строк  
    text = re.sub(r"[ \t]+", " ", text)  
    text = re.sub(r" *\n *", "\n", text)  
    text = re.sub(r"\n{3,}", "\n\n", text)  
  
    return text.strip()  
  
  
def chunk_text(  
    text: str,  
    size: int = config.CHUNK_SIZE,  
    overlap: int = config.CHUNK_OVERLAP,  
) -> list[str]:  
    """Нарезать текст на чанки размером size с перекрытием overlap.  
  
    Перекрытие нужно, чтобы мысль, попавшая на границу нарезки, не потерялась:  
    её хвост окажется в начале следующего чанка.  
    """  
    if size <= 0:  
        raise ValueError("CHUNK_SIZE должен быть положительным")  
    if overlap < 0 or overlap >= size:  
        raise ValueError("CHUNK_OVERLAP должен быть в [0, CHUNK_SIZE)")  
    if not text:  
        return []  
  
    step = size - overlap  
    chunks: list[str] = []  
    for start in range(0, len(text), step):  
        end = start + size  
        chunk = text[start:end]  
        if chunk.strip():  
            chunks.append(chunk)  
        if end >= len(text):  
            break  
    return chunks  
  
  
# Словарь рубрик на уровне модуля под предметную область  
RUBRICS: dict[str, list[str]] = {  
    "retake": [  
        "пересдач",  
        "академическ",  
        "задолженност",  
        "ликвидац",  
        "комисси",  
        "неудовлетворительн",  
        "неявк",  
        "долг",  
    ],  
    "attestation": [  
        "промежуточн",  
        "аттестац",  
        "экзамен",  
        "зачет",  
        "зачёт",  
        "сесси",  
        "успеваемост",  
        "ведомост",  
        "балльн",  
        "рейтинг",  
    ],  
    "transfer": [  
        "перевод",  
        "восстановлен",  
        "отчислен",  
        "платн",  
        "бюджет",  
        "вакантн",  
        "курс",  
        "направлен",  
        "форма обучен",  
        "очн",  
        "заочн",  
    ],  
    "discipline": [  
        "внутренн",  
        "распоряд",  
        "дисциплин",  
        "устав",  
        "нарушен",  
        "взыскан",  
        "замечан",  
        "выговор",  
        "поведен",  
        "правил",  
    ],  
}  
  
DEFAULT_RUBRIC = "general"  
  
# Подсказки по именам файлов-источников  
SOURCE_HINTS: list[tuple[tuple[str, ...], str]] = [  
    (("перевод", "восстанов"), "transfer"),  
    (("аттестац",), "attestation"),  
    (("распоряд", "пвр", "устав"), "discipline"),  
]  
  
  
def _normalize_str(value: str) -> str:  
    return unicodedata.normalize("NFKC", value).lower().replace("ё", "е")  
  
  
def detect_category(text: str, source: str) -> str:  
    """Определить рубрику чанка по содержимому и источнику.  
  
    Всегда возвращает непустую строку (при отсутствии совпадений — DEFAULT_RUBRIC).  
    """  
    norm_source = _normalize_str(source)  
    norm_text = _normalize_str(text)  
  
    scores: dict[str, int] = {rubric: 0 for rubric in RUBRICS}  
  
    # Подсчёт совпадений по ключевым основам в тексте чанка  
    for rubric, stems in RUBRICS.items():  
        for stem in stems:  
            norm_stem = _normalize_str(stem)  
            if norm_stem in norm_text:  
                scores[rubric] += 1  
  
    # Более точные правила для пересдач/задолженностей и переводов/отчислений внутри общих положений  
    if any(k in norm_text for k in ("пересдач", "задолженност", "ликвидац", "комисси")):  
        scores["retake"] += 2  
  
    if any(k in norm_text for k in ("перевод", "восстановлен", "отчислен")):  
        scores["transfer"] += 2  
  
    # Мягкий вес имени файла, чтобы задать приоритет соответствующего документа  
    for hints, rubric in SOURCE_HINTS:  
        if any(_normalize_str(h) in norm_source for h in hints):  
            scores[rubric] += 1  
  
    best_rubric = max(scores, key=lambda r: scores[r])  
    if scores[best_rubric] > 0:  
        return best_rubric  
  
    return DEFAULT_RUBRIC  
  
  
# --- Сборка индекса: готова ---  
  
BATCH_SIZE = 256  
  
  
def _flush(collection, batch: dict) -> None:  
    """Записать накопленные чанки одной операцией и очистить пачку."""  
    if batch:  
        ids = list(batch)  
        collection.upsert(  
            ids=ids,  
            documents=[batch[i][0] for i in ids],  
            metadatas=[batch[i][1] for i in ids],  
        )  
        batch.clear()  
  
  
def ingest() -> int:  
    """Обходит исходные данные, нарезает и складывает в ChromaDB. Возвращает число чанков.  
  
    Индекс собирается заново с нуля: коллекция очищается перед наполнением.  
    Иначе после смены CHUNK_SIZE в коллекции остались бы чанки предыдущей  
    нарезки, и hit-rate замерялся бы по смеси двух — молча, без ошибки.  
    """  
    collection = config.reset_collection()  
    total = 0  
  
    # README.md отсеивается намеренно: это шаблонная заглушка каталога,  
    # а не документ предметной области. Без отсева она попадает в индекс  
    # и выдаётся поиском как знание.  
    files = [  
        p for p in config.RAW_DIR.rglob("*")  
        if p.suffix.lower() in READERS and p.name != "README.md"  
    ]  
    if not files:  
        raise SystemExit(  
            f"В {config.RAW_DIR} нет файлов .txt/.md/.pdf — положите исходные данные (П2)."  
        )  
  
    # Чанки копятся и пишутся в коллекцию пачками: модель эмбеддингов на CPU  
    # обрабатывает пачку в разы быстрее, чем те же чанки по одному.  
    batch: dict[str, tuple[str, dict]] = {}  
  
    for path in files:  
        for page, raw in READERS[path.suffix.lower()](path):  
            cleaned = clean_text(raw)  
            if not cleaned:  
                continue  
  
            for chunk in chunk_text(cleaned, config.CHUNK_SIZE, config.CHUNK_OVERLAP):  
                # id детерминированный: одинаковый текст даёт один и тот же чанк  
                chunk_id = hashlib.sha1(  
                    f"{path.name}:{page}:{chunk}".encode("utf-8")  
                ).hexdigest()  
                batch[chunk_id] = (  
                    chunk,  
                    {  
                        "source": path.name,  
                        "page": page,  
                        "category": detect_category(chunk, path.name),  
                    },  
                )  
                total += 1  
  
                if len(batch) >= BATCH_SIZE:  
                    _flush(collection, batch)  
  
    _flush(collection, batch)  
    return total  
  
  
# --- Замер качества поиска: готов ---  
  
def evaluate_retrieval(k: int = config.TOP_K) -> float:  
    """hit-rate@k по черновику Gold Dataset. Готово, менять не нужно."""  
    import json  
  
    dataset_path = Path(__file__).parent / "gold_dataset.json"  
    cases = [  
        c for c in json.loads(dataset_path.read_text(encoding="utf-8"))  
        if c.get("expected_sources")  
    ]  
  
    if not cases:  
        print("В gold_dataset.json нет вопросов с expected_sources — замерять нечего (П2).")  
        return 0.0  
  
    collection = config.get_collection()  
    hits = 0  
  
    print(f"\nhit-rate@{k}  |  чанк {config.CHUNK_SIZE}, overlap {config.CHUNK_OVERLAP}")  
    for case in cases:  
        result = collection.query(query_texts=[case["question"]], n_results=k)  
        found = {(m or {}).get("source", "") for m in (result.get("metadatas") or [[]])[0]}  
        hit = bool(found & set(case["expected_sources"]))  
        hits += hit  
        print(f"  [{'+' if hit else '-'}] {case['question'][:70]}")  
        if not hit:  
            print(f"      ожидалось: {case['expected_sources']}, найдено: {sorted(found)}")  
  
    rate = hits / len(cases)  
    print(f"\nhit-rate@{k} = {rate:.0%}  ({hits} из {len(cases)})")  
    skipped = len(json.loads(dataset_path.read_text(encoding="utf-8"))) - len(cases)  
    if skipped:  
        print(f"Пропущено вопросов без expected_sources: {skipped} — они меряются на П6.")  
    return rate  
  
  
if __name__ == "__main__":  
    if "--eval" not in sys.argv:  
        count = ingest()  
        print(f"Индекс пересобран заново. Проиндексировано чанков: {count}")  
        print(f"Всего в коллекции: {config.get_collection().count()}")  
  
    evaluate_retrieval()  
