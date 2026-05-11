# text_processor.py
# pav13

import logging
from urllib.parse import unquote
from functools import lru_cache
from num2words import num2words

logger = logging.getLogger(__name__)


class TextProcessor:
    """обработка текста (разбиение на предложения, SSML, числа, транслитерация)"""
    MAX_TEXT_LENGTH = 900
    PITCH_ORDER = ["x-low", "low", "medium", "high", "x-high"]
    pause0, pause1, pause2, pause3, pause4, pause5 = 0, 120, 180, 215, 320, 480
    # pause0, pause1, pause2, pause3, pause4, pause5 = 0, 80, 100, 130, 200, 320
    PUNCT = {'.': pause4, '?': pause4, '!': pause4}
    PUNCT_REPL = {',': pause2, '(': pause2, ')': pause2, '[': pause2, ']': pause2, 
                    ':': pause1, ';': pause3, '—': pause3, '…': pause5}
    ALLOWED = frozenset("_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–*")
    LATIN = frozenset("abcdefghijklmnopqrstuvwxyz&")
    TRANSLIT_MAP = {'ough':'о','augh':'о','eigh':'эй','igh':'ай','tion':'шн','shch':'щ','ture': 'чер','sion': 'жн',
        'tch':'ч','sch':'ск','scr':'скр','thr':'тр','squ':'скв','ear':'ир','air':'эр','are':'эр','the':'зэ','and':'энд',
        'ea':'и','ee':'и','oo':'у','ai':'эй','ay':'эй','ei':'эй','ey':'эй','oi':'ой','oy':'ой','ou':'ау','ow':'ау','au':'о','aw':'о','ie':'и','ui':'у','ue':'ю','uo':'уо','eu':'ю','ew':'ю','oa':'о','oe':'о','sh':'ш','ch':'ч','zh':'ж','th':'з','kh':'х','ts':'ц','ph':'ф','wh':'в','gh':'г','qu':'кв','gu':'г','dg':'дж','ce':'це','ci':'си','cy':'си','ck':'к','ge':'дж','gi':'джи','gy':'джи','er':'эр',
        'a':'а','b':'б','c':'к','d':'д','e':'е','f':'ф','g':'г','h':'х','i':'и','j':'дж','k':'к','l':'л','m':'м','n':'н','o':'о','p':'п','q':'к','r':'р','s':'с','t':'т','u':'у','v':'в','w':'в','x':'кс','y':'и','z':'з','&':'и'}

    def __init__(self):
        self.speed_percent = 100
        self.pitch_level = "medium"
        self.transl_trie = {}
        for k, v in self.TRANSLIT_MAP.items():
            node = self.transl_trie
            for ch in k:
                node = node.setdefault(ch, {})
            node['_'] = v
    
    def set_ssml_params(self, speed_percent: int, pitch: str):
        self.speed_percent = max(40, min(300, speed_percent))
        self.pitch_level = pitch if pitch in self.PITCH_ORDER else "medium"

    def split_sentences(self, text: str) -> list:
        sentences = []
        current_sentence = []
        end_of_sentence_chars = {'.', '!', '?', '…'}
        len_text = len(text)
        
        for i, ch in enumerate(text):
            current_sentence.append(ch)
            if ch in end_of_sentence_chars:
                if i + 1 == len_text or text[i + 1] == ' ' or text[i + 1] == '\n':
                    sentences.append(''.join(current_sentence).strip())
                    current_sentence = []
        
        if current_sentence:
            sentences.append(''.join(current_sentence).strip())
        
        sentences = [s for s in sentences if s]
        result = []
        for sentence in sentences:
            if len(sentence) <= 200:
                result.append(sentence)
                continue

            parts = []
            current_part = []
            minor_delim_chars = {',', ';', ':', '-', '—'}
            
            for ch in sentence:
                current_part.append(ch)
                if ch in minor_delim_chars:
                    parts.append(''.join(current_part).strip())
                    current_part = []
            if current_part:
                parts.append(''.join(current_part).strip())
            parts = [p for p in parts if p]
            current = []
            cur_len = 0
            cur_join = " ".join

            for part in parts:
                if cur_len and cur_len < 60:
                    current.append(part)
                    cur_len += len(part) + 1
                elif current:
                    result.append(cur_join(current))
                    current = [part]
                    cur_len = len(part)
                else:
                    current = [part]
                    cur_len = len(part)

            if current:
                if result and cur_len < 60:
                    result[-1] += " " + cur_join(current)
                else:
                    result.append(cur_join(current))
        
        return result

    def process_sentence(self, text: str, vol_boost: float) -> tuple:
        len_text = len(text)
        if len_text > self.MAX_TEXT_LENGTH:
            len_text = self.MAX_TEXT_LENGTH
            text = text[:len_text]
            logger.warning(f"Text length truncated to {len_text} chars.")
        
        text = unquote(text).lower()
        has_latin = any(ch in self.LATIN for ch in text)
        vol_boost_mod = vol_boost + 3.5 if text.rstrip().endswith('!') else vol_boost
        processed_text = self._proc(text, len_text, has_latin)
        ssml = f'<speak><prosody rate="{self.speed_percent}%" pitch="{self.pitch_level}">{processed_text}</prosody></speak>'
        
        return ssml, vol_boost_mod

    def _proc(self, text: str, len_text: int, has_latin: bool) -> str:
        res, buf, i = [], [], 0
        PUNCT_REPL, PUNCT = self.PUNCT_REPL, self.PUNCT
        ALLOWED, LATIN = self.ALLOWED, self.LATIN
        
        def flush_buf(with_question_mark=False):
            if not buf:
                return None
            s = ''.join(buf).strip()
            if not s:
                return None
            if with_question_mark:
                last_space = s.rfind(' ')
                if last_space == -1:
                    return f"*{s}*"
                return f"{s[:last_space]} *{s[last_space+1:]}*"
            return s
        
        def add_break(ch, is_punct_repl=False):
            time = PUNCT_REPL[ch] if is_punct_repl else PUNCT.get(ch, 0)
            res.append(f'<break time="{time}ms"/>')
        
        while i < len_text:
            ch = text[i]
            
            # Цифры
            if ch.isdigit():
                i, p = self._num(text, i)
                buf.append(p)
                continue
            
            # Транслитерация
            if has_latin and ch in LATIN:
                ni, tr = self._trans(text, i)
                if tr and tr != ch:
                    buf.append(tr)
                i = ni
                continue
            
            # Многоточие
            if ch == '.' and i + 2 < len_text and text[i+1] == '.' and text[i+2] == '.':
                if (s := flush_buf()):
                    res.append(s)
                add_break('…', True)
                i += 3
                continue
            
            # Точка
            if ch == '.':
                if (s := flush_buf()):
                    res.append(s)
                add_break(ch)
                buf.clear()
                i += 1
                continue
            
            # Знаки с заменой
            if ch in PUNCT_REPL:
                if (s := flush_buf()):
                    res.append(s)
                add_break(ch, True)
                buf.clear()
                i += 1
                continue
            
            # Вопросительный знак
            if ch == '?':
                if (s := flush_buf(True)):
                    res.append(s)
                    res.append(ch)
                add_break(ch)
                buf.clear()
                i += 1
                continue
            
            # Восклицательный знак
            if ch == '!':
                if (s := flush_buf()):
                    res.append(s)
                    res.append(ch)
                add_break(ch)
                buf.clear()
                i += 1
                continue
            
            # Дефис/тире
            if ch == '-':
                if i > 0 and text[i-1] == ' ' and i + 1 < len_text and text[i+1] == ' ':
                    if (s := flush_buf()):
                        res.append(s)
                    add_break('—', True)
                    buf.clear()
                    i += 2
                    continue
                
                buf.append(' ')
                i += 1
                continue
            
            # Пробелы
            if ch.isspace():
                if not buf or buf[-1] != ' ':
                    buf.append(' ')
                i += 1
                continue
            
            # Разрешённые символы
            if ch in ALLOWED:
                buf.append(ch)
                i += 1
                continue
            
            # Прочие символы заменяем пробелом
            if not (buf and buf[-1] == ' '):
                buf.append(' ')
            i += 1
        
        # Финальная очистка
        if buf and (s := ''.join(buf).strip()):
            res.append(s)
        
        return ''.join(res).strip()
    
    def _trans(self, text: str, pos: int) -> tuple:
        node, best, best_pos = self.transl_trie, None, pos
        j = pos
        n = len(text)
        while j < n and text[j] in node:
            node = node[text[j]]; j += 1
            if '_' in node: best, best_pos = node['_'], j
        return (best_pos, best) if best else (pos + 1, text[pos] if text[pos] in self.ALLOWED else " ")
    
    def _wrap(self, text: str) -> str:
        last_space = text.rfind(' ')
        if last_space == -1:
            return f"*{text}*"
        
        return f"{text[:last_space]} *{text[last_space+1:]}*"

    def _num(self, text: str, start: int) -> tuple:
        i, n = start, len(text)
        while i < n and text[i].isdigit(): i += 1
        if i == start: return start + 1, text[start]

        num_str = text[start:i]
        num_val = int(num_str)

        # Время
        if i < n and text[i] == ':' and i + 2 < n and text[i+1:i+3].isdigit():
            mm = int(text[i+1:i+3])
            end = i + 3
            res = f"{self._num_to_words(num_val)} часов {self._num_to_words(mm)} минут"
            if end < n and text[end] == ':' and end + 2 < n and text[end+1:end+3].isdigit():
                res += f" {self._num_to_words(int(text[end+1:end+3]))} секунд"
                end += 3
            return end, res

        # Дата
        if i < n and text[i] in '.-/' and i + 1 < n and text[i+1].isdigit():
            j = i + 1
            while j < n and text[j].isdigit(): j += 1
            mm = int(text[i+1:j])

            if j < n and text[j] in '.-/' and j + 1 < n and text[j+1].isdigit():
                l = j + 1
                while l < n and text[l].isdigit(): l += 1
                part3 = int(text[j+1:l])
                months = ["", "января", "февраля", "марта", "апреля", "мая", "июня",
                          "июля", "августа", "сентября", "октября", "ноября", "декабря"]
                month_w = months[mm] if 1 <= mm <= 12 else str(mm)

                if (i - start) == 4:
                    year_w = self._num_to_words(num_val, to='ordinal', case='g')
                    day_w = self._num_to_words(part3, to='ordinal', gender='n')
                else:
                    day_w = self._num_to_words(num_val, to='ordinal', gender='n')
                    year_w = self._num_to_words(part3, to='ordinal', case='g')

                return l, f"{day_w} {month_w} {year_w} года"

        # Десятичные дроби
        if i < n and text[i] in '.,' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            if num_val == 1 and text[i+1:k] == '5':
                return k, "полтора"
            return k, self._num_to_words(text[start:k].replace(',', '.'))

        # Проценты
        if i < n and text[i] == '%':
            return i + 1, f"{self._num_to_words(num_val)} {self._plur(num_val, ('процент', 'процента', 'процентов'))}"

        # Обычные дроби
        if i < n and text[i] == '/' and i + 1 < n and text[i+1].isdigit():
            k = i + 1
            while k < n and text[k].isdigit(): k += 1
            return k, f"{self._num_to_words(num_val)} дробь {self._num_to_words(int(text[i+1:k]))}"

        # Обычное число
        return i, self._num_to_words(num_val)

    @staticmethod
    def _plur(n: int, forms: tuple) -> str:
        if n % 100 in (11, 12, 13, 14): return forms[2]
        if n % 10 == 1: return forms[0]
        if n % 10 in (2, 3, 4): return forms[1]
        return forms[2]
    
    def _num_to_words(self, num, to='cardinal', case='n', gender='m', plural=False, animate=False) -> str:
        """num: int/str | to: cardinal/ordinal | case: n/g/d/a/i/p (им/род/дат/вин/твор/предл)
           gender: m/f/n | plural: bool | animate: bool (одуш., только для вин. п.)"""
        return num2words(num, lang='ru', to=to, case=case, gender=gender, plural=plural, animate=animate)
