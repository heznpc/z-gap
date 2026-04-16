#!/usr/bin/env python3
"""Generate data/dialect_stimuli_v2.json.

v2 schema adds:
- per-language paraphrases for ko/zh/ar/es (en paraphrases copied from v1)
- Korean Gyeongsang dialect (dialects.ko_gyeongsang)
- Egyptian Arabic dialect (dialects.ar_egyptian)

Non-English paraphrases and both dialects are LLM-generated (Opus 4.6,
in-session, 2026-04-11). See data/dialect_stimuli_v2_prompts.md for the
prompt templates and limitations.

Originals are copied from src/stimuli.py to preserve embedding cache coherence.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.stimuli import get_all_operations

TARGET_IDS = [
    "comp_01_sort_asc", "comp_02_find_max", "comp_03_filter_pos",
    "comp_04_reverse", "comp_05_count", "comp_06_sum",
    "comp_07_deduplicate", "comp_08_top3", "comp_09_mean",
    "comp_10_sort_desc", "comp_11_concat", "comp_12_uppercase",
    "comp_13_split", "comp_14_replace", "comp_15_length",
    "judg_01_prioritize", "judg_02_evaluate", "judg_03_summarize",
    "judg_04_recommend", "judg_05_risk", "judg_06_categorize",
    "judg_07_select_key", "judg_08_rank", "judg_09_flag",
    "judg_10_tone", "judg_11_sentiment", "judg_12_appropriate",
    "judg_13_urgency", "judg_14_credibility", "judg_15_simplify",
]


# --- Korean paraphrases (standard, Opus 4.6 in-session, 2026-04-11) ---
KO_PARA = {
    "comp_01_sort_asc": [
        "리스트 원소들을 작은 값부터 큰 값 순으로 배열하라",
        "항목들이 점점 커지는 순서가 되도록 나열하라",
        "값이 작은 것부터 큰 것 순으로 목록을 재배치하라",
    ],
    "comp_02_find_max": [
        "리스트에서 가장 큰 수를 찾아내라",
        "목록에 포함된 값들 중 최대값을 식별하라",
        "리스트 원소들 가운데 제일 큰 값을 구하라",
    ],
    "comp_03_filter_pos": [
        "리스트에서 0보다 큰 값만 골라내라",
        "양의 숫자만 남기고 나머지는 제외하라",
        "목록에 있는 양의 값만 추출하라",
    ],
    "comp_04_reverse": [
        "리스트 항목 순서를 역순으로 바꿔라",
        "목록의 앞뒤 순서를 반대로 만들어라",
        "원소들의 배열 순서를 거꾸로 배치하라",
    ],
    "comp_05_count": [
        "리스트에 들어있는 항목 수를 구하라",
        "목록이 포함하는 원소의 총 개수를 계산하라",
        "리스트 길이를 측정하라",
    ],
    "comp_06_sum": [
        "리스트 안 모든 값을 더한 총합을 계산하라",
        "목록에 담긴 숫자들을 전부 합산하라",
        "모든 원소를 합쳐 총합을 도출하라",
    ],
    "comp_07_deduplicate": [
        "리스트 안 반복되는 값을 없애라",
        "중복 항목을 걸러내고 고유한 원소만 남겨라",
        "같은 값이 여러 번 나타나는 경우 한 번만 남기고 제거하라",
    ],
    "comp_08_top3": [
        "리스트에서 값이 가장 높은 상위 세 개 원소를 추출하라",
        "목록 중 최대값 세 개를 식별하라",
        "가장 큰 숫자 세 개를 주어진 리스트에서 골라라",
    ],
    "comp_09_mean": [
        "리스트 안 모든 값을 합해 개수로 나눈 평균치를 계산하라",
        "목록 원소들의 산술 평균을 도출하라",
        "모든 숫자의 평균값을 산출하라",
    ],
    "comp_10_sort_desc": [
        "리스트 원소들을 큰 값부터 작은 값 순으로 배열하라",
        "항목들이 점점 작아지는 순서가 되도록 나열하라",
        "값이 큰 것부터 작은 것 순으로 목록을 재배치하라",
    ],
    "comp_11_concat": [
        "두 개의 텍스트를 하나로 결합하라",
        "첫 번째 문자열 뒤에 두 번째 문자열을 연결하라",
        "두 스트링을 합쳐 하나의 문자열로 만들어라",
    ],
    "comp_12_uppercase": [
        "텍스트의 모든 글자를 대문자 형태로 바꿔라",
        "스트링 안 문자를 전부 대문자화하라",
        "주어진 텍스트를 대문자 표기로 변형하라",
    ],
    "comp_13_split": [
        "텍스트를 공백으로 분리하여 단어 리스트를 생성하라",
        "스트링에서 공백을 구분자로 사용해 단어들을 뽑아내라",
        "빈칸을 경계로 문자열을 쪼개 단어별로 나열하라",
    ],
    "comp_14_replace": [
        "텍스트 안 모든 'a'를 'b'로 대체하라",
        "스트링에 등장하는 'a' 전부를 'b'로 치환하라",
        "문자열에서 글자 'a'를 찾아 'b'로 교체하라",
    ],
    "comp_15_length": [
        "텍스트 안 문자 개수를 계산하라",
        "스트링이 몇 글자인지 측정하라",
        "주어진 문자열의 총 글자 수를 산출하라",
    ],
    "judg_01_prioritize": [
        "업무의 중요성 순으로 우선순위를 매겨라",
        "각 작업이 얼마나 중요한지 평가하고 순서를 결정하라",
        "중요한 작업부터 먼저 처리하도록 우선순위를 부여하라",
    ],
    "judg_02_evaluate": [
        "모든 후보자의 수준을 판단하라",
        "후보 각각의 우수성을 검토하라",
        "개별 후보에 대해 품질 수준을 심사하라",
    ],
    "judg_03_summarize": [
        "문서 내용을 짧은 문장 몇 개로 간추려라",
        "글을 핵심만 담은 간결한 개요로 정리하라",
        "문서의 주요 내용을 몇 개의 문장으로 축약하라",
    ],
    "judg_04_recommend": [
        "리스트 중 최적의 옵션을 제안하라",
        "목록에 나온 선택지들 가운데 가장 적합한 것을 추천하라",
        "주어진 선택지 중 최고의 안을 골라 추천하라",
    ],
    "judg_05_risk": [
        "개별 항목이 얼마나 위험한지 판단하라",
        "각 항목의 리스크 정도를 측정하라",
        "항목마다 위험도를 산정하라",
    ],
    "judg_06_categorize": [
        "각 항목을 알맞은 범주로 나눠라",
        "항목들을 성격에 맞는 카테고리에 배치하라",
        "적합한 그룹별로 항목들을 정리하라",
    ],
    "judg_07_select_key": [
        "DB에 저장된 고객 중 핵심 고객을 추려라",
        "데이터베이스 기록에서 중요 고객을 선택하라",
        "주요 고객층에 해당하는 항목을 데이터베이스에서 골라내라",
    ],
    "judg_08_rank": [
        "각 기사의 관련도를 기준으로 순서를 정하라",
        "기사들을 주제 관련성 순으로 정렬하라",
        "관련 정도에 맞춰 기사에 순위를 부여하라",
    ],
    "judg_09_flag": [
        "문제 가능성이 있는 항목에 표식을 달아라",
        "의심스러운 항목을 발견해 별도로 표시하라",
        "이상이 있어 보이는 항목을 선별해 플래그를 지정하라",
    ],
    "judg_10_tone": [
        "메시지 톤을 보다 격식 있게 수정하라",
        "내용을 전문적이고 정중한 어투로 바꿔라",
        "문체를 더 전문성 있는 방향으로 다듬어라",
    ],
    "judg_11_sentiment": [
        "글에 담긴 전체적 정서 분위기를 파악하라",
        "텍스트가 긍정적인지 부정적인지 종합 평가하라",
        "문장에 반영된 감정의 방향성을 분석하라",
    ],
    "judg_12_appropriate": [
        "해당 콘텐츠가 독자 수준에 맞는지 검토하라",
        "타겟 독자층에 적합한 내용인지 확인하라",
        "이 콘텐츠가 대상 청중에게 적절한지 평가하라",
    ],
    "judg_13_urgency": [
        "각 티켓의 긴급 수준을 기준으로 나눠라",
        "티켓들을 긴급성 정도에 맞춰 구분하라",
        "긴급함의 차이에 따라 티켓을 카테고리로 분류하라",
    ],
    "judg_14_credibility": [
        "출처마다 얼마나 믿을 만한지 판단하라",
        "소스의 신빙성을 검증하라",
        "정보원 각각의 신뢰 수준을 측정하라",
    ],
    "judg_15_simplify": [
        "비전공자도 이해할 수 있도록 설명을 단순화하라",
        "일반 대중이 쉽게 접할 수 있게 설명을 풀어 써라",
        "전문 지식이 없는 사람도 알아듣도록 내용을 쉽게 다듬어라",
    ],
}


# --- Chinese paraphrases (Standard Mandarin, Opus 4.6 in-session, 2026-04-11) ---
ZH_PARA = {
    "comp_01_sort_asc": [
        "把列表元素从小到大排列",
        "将列表中的项按递增顺序排列",
        "将列表按从低到高的顺序进行排序",
    ],
    "comp_02_find_max": [
        "确定列表中最大的数字",
        "找出列表里数值最高的元素",
        "查找列表元素中的最大值",
    ],
    "comp_03_filter_pos": [
        "从列表中提取所有大于零的数",
        "保留列表中的正数,去除其他值",
        "只选出列表里的正值",
    ],
    "comp_04_reverse": [
        "将列表里项目的顺序颠倒过来",
        "把列表排列成最后一个元素在最前面的形式",
        "对列表元素的排序进行反向处理",
    ],
    "comp_05_count": [
        "统计列表包含多少个项目",
        "求出列表中元素的总数",
        "测定列表的长度",
    ],
    "comp_06_sum": [
        "将列表里的每个数字相加",
        "求出列表所有值的合计",
        "通过累加列表中所有条目得出总和",
    ],
    "comp_07_deduplicate": [
        "删除列表里出现重复的数值",
        "只保留列表中独一无二的项",
        "将列表里多次出现的条目清除",
    ],
    "comp_08_top3": [
        "确定列表中数值最高的前三个项目",
        "从列表中挑出最大的三个数",
        "找出所给列表内排名前三的最大值",
    ],
    "comp_09_mean": [
        "求出列表里每个数值的均值",
        "得出所有列表条目的算术平均",
        "通过累加并除以个数来确定列表的平均值",
    ],
    "comp_10_sort_desc": [
        "把列表元素从大到小排列",
        "将列表中的项按递减顺序排列",
        "将列表按从高到低的顺序进行排序",
    ],
    "comp_11_concat": [
        "把两个文本连成一个",
        "将第二个字符串接在第一个字符串之后",
        "将两个字符串拼接为单个字符串",
    ],
    "comp_12_uppercase": [
        "把字符串中的所有字母改为大写形式",
        "将文本里的每个字符转为大写",
        "将整个字符串变成大写写法",
    ],
    "comp_13_split": [
        "以空格为分隔符把字符串切分成单词",
        "在每个空格处断开字符串,形成单词数组",
        "将字符串依空格分隔,生成词语列表",
    ],
    "comp_14_replace": [
        "把字符串里出现的每一个'a'都改成'b'",
        "将文本中所有'a'字符替换成'b'",
        "把字符串的每处'a'转换为'b'",
    ],
    "comp_15_length": [
        "测定文本的字符数量",
        "求出字符串中包含多少字符",
        "确定给定字符串的总字符数",
    ],
    "judg_01_prioritize": [
        "根据任务重要程度排定优先级",
        "把重要的任务放在前面",
        "依重要性给每项任务分配优先顺序",
    ],
    "judg_02_evaluate": [
        "判断各候选者的优劣程度",
        "审视每一位候选人的素质",
        "对每个候选人进行质量评定",
    ],
    "judg_03_summarize": [
        "将文档内容浓缩为简短的几句话",
        "把文档精简成一段概要",
        "为文档撰写一个简短的摘要",
    ],
    "judg_04_recommend": [
        "在列表里选出最合适的方案并推荐",
        "向用户推介列表中最优的一个选项",
        "从现有的选项里挑出最好的加以推荐",
    ],
    "judg_05_risk": [
        "判断每一个项目的风险高低",
        "对各个条目进行风险程度评估",
        "为每项给出风险等级",
    ],
    "judg_06_categorize": [
        "按类别将各项目进行归类",
        "把项目划分到合适的分组里",
        "依据属性将每个项目放入对应类别",
    ],
    "judg_07_select_key": [
        "从数据库里筛选出主要客户",
        "在数据库记录中挑出重要的客户",
        "识别数据库中的核心客户群",
    ],
    "judg_08_rank": [
        "依据相关程度给文章排序",
        "把文章按与主题的相关性进行排列",
        "根据相关度高低确定文章名次",
    ],
    "judg_09_flag": [
        "在有潜在问题的项目上做出标注",
        "找出可疑条目并加以标识",
        "对看起来存疑的条目进行标记",
    ],
    "judg_10_tone": [
        "把信息的措辞改得更正式专业",
        "将消息的表达重写得更具专业感",
        "让讯息的语气变得更为严谨专业",
    ],
    "judg_11_sentiment": [
        "分析这段文字表达的总体情绪",
        "辨别文本是积极、消极还是中性",
        "确定文本中流露的主要情感倾向",
    ],
    "judg_12_appropriate": [
        "评估内容对目标读者是否合适",
        "判定该内容是否匹配目标受众",
        "考察这段内容是否适合既定的观众群",
    ],
    "judg_13_urgency": [
        "按工单的紧急度进行归类",
        "依据紧急等级对工单分组",
        "根据紧急程度对工单加以区分",
    ],
    "judg_14_credibility": [
        "判断各来源是否值得信任",
        "审查每一个信息源的可靠性",
        "衡量不同来源的可信程度",
    ],
    "judg_15_simplify": [
        "把解释改写为非专业读者也能理解的版本",
        "用更通俗的方式向普通读者表述",
        "将内容简化,使一般大众也能看懂",
    ],
}


# --- Arabic (MSA) paraphrases (Opus 4.6 in-session, 2026-04-11) ---
AR_PARA = {
    "comp_01_sort_asc": [
        "نظّم عناصر القائمة من الأصغر إلى الأكبر",
        "ضع عناصر القائمة بترتيب متزايد",
        "اجعل القائمة مرتّبة من الأدنى إلى الأعلى",
    ],
    "comp_02_find_max": [
        "حدد أكبر رقم في القائمة",
        "اعثر على القيمة الأعلى بين عناصر القائمة",
        "ابحث عن أكبر قيمة ضمن القائمة",
    ],
    "comp_03_filter_pos": [
        "استخرج جميع الأرقام الأكبر من صفر من القائمة",
        "احتفظ بالقيم الموجبة فقط واستبعد الباقي",
        "اختر فقط العناصر الموجبة من القائمة",
    ],
    "comp_04_reverse": [
        "قم بقلب تسلسل عناصر القائمة",
        "أعد ترتيب القائمة بحيث يصبح آخر عنصر أولاً",
        "اقلب ترتيب عناصر القائمة إلى العكس",
    ],
    "comp_05_count": [
        "حدد كم عنصرًا تحتوي عليه القائمة",
        "أوجد العدد الإجمالي لعناصر القائمة",
        "قِس طول القائمة",
    ],
    "comp_06_sum": [
        "اجمع كل رقم في القائمة",
        "أوجد المجموع الإجمالي لقيم القائمة",
        "اجمع كل عناصر القائمة للحصول على الإجمالي",
    ],
    "comp_07_deduplicate": [
        "احذف القيم المتكررة في القائمة",
        "احتفظ بالعناصر الفريدة فقط",
        "استبعد أي مدخل يظهر أكثر من مرة",
    ],
    "comp_08_top3": [
        "حدد أكبر ثلاثة أرقام في القائمة",
        "استخرج القيم الثلاث الأعلى من القائمة",
        "اعثر على العناصر الثلاثة الأكبر في القائمة",
    ],
    "comp_09_mean": [
        "أوجد الوسط الحسابي لقيم القائمة",
        "احسب المعدل لجميع عناصر القائمة",
        "اجمع الأرقام واقسمها على عددها للحصول على المتوسط",
    ],
    "comp_10_sort_desc": [
        "نظّم عناصر القائمة من الأكبر إلى الأصغر",
        "ضع عناصر القائمة بترتيب متناقص",
        "اجعل القائمة مرتّبة من الأعلى إلى الأدنى",
    ],
    "comp_11_concat": [
        "ادمج السلسلتين النصيتين في واحدة",
        "ضع السلسلة الثانية في نهاية السلسلة الأولى",
        "وصل السلسلتين لتصبحا سلسلة واحدة",
    ],
    "comp_12_uppercase": [
        "اجعل جميع حروف السلسلة كبيرة",
        "حوّل كل حرف من النص إلى حالة كبيرة",
        "أعد كتابة النص بالأحرف الكبيرة",
    ],
    "comp_13_split": [
        "افصل السلسلة عند كل مسافة لتكوين قائمة من الكلمات",
        "استخدم المسافة كفاصل لتجزئة النص إلى كلمات",
        "اكسر النص عند المسافات للحصول على مصفوفة كلمات",
    ],
    "comp_14_replace": [
        "غيّر كل حرف 'a' في السلسلة إلى 'b'",
        "بدّل كل موضع يحتوي على 'a' بـ 'b' في النص",
        "حوّل جميع أحرف 'a' إلى 'b' داخل السلسلة",
    ],
    "comp_15_length": [
        "حدد عدد الأحرف في السلسلة",
        "أوجد كم حرفًا تحتوي عليه السلسلة",
        "احسب إجمالي عدد الأحرف في النص",
    ],
    "judg_01_prioritize": [
        "نظّم المهام وفق درجة أهميتها",
        "ضع المهام الأكثر أهمية في المقدمة",
        "حدد أولويات المهام بناءً على أهميتها",
    ],
    "judg_02_evaluate": [
        "احكم على مستوى كل مرشح",
        "قدّر مدى جودة المرشحين",
        "قيّم كل مرشح من حيث جودته",
    ],
    "judg_03_summarize": [
        "اكتب ملخصًا موجزًا للمستند",
        "اختصر المستند في نظرة عامة قصيرة",
        "قدّم خلاصة مختصرة للمستند",
    ],
    "judg_04_recommend": [
        "اقترح الخيار الأمثل من القائمة",
        "اختر أفضل خيار من القائمة واشرح السبب",
        "قدّم توصية بأنسب خيار من القائمة",
    ],
    "judg_05_risk": [
        "حدد مدى خطورة كل عنصر",
        "احكم على درجة المخاطرة لكل بند",
        "قدّر مستوى الخطر لكل مدخل",
    ],
    "judg_06_categorize": [
        "رتّب العناصر في فئات ملائمة",
        "اجمع العناصر حسب الفئة المناسبة",
        "ضع كل عنصر في المجموعة الأنسب له",
    ],
    "judg_07_select_key": [
        "حدد أهم العملاء من قاعدة البيانات",
        "انتقِ العملاء الأساسيين من السجلات",
        "اختر العملاء الرئيسيين من بين الموجودين في قاعدة البيانات",
    ],
    "judg_08_rank": [
        "صنّف المقالات من الأكثر إلى الأقل صلة",
        "نظّم المقالات وفق درجة ارتباطها بالموضوع",
        "رتّب المقالات بناءً على مدى صلتها",
    ],
    "judg_09_flag": [
        "حدد أي إدخال قد يحتوي على مشكلة",
        "ضع علامة على المدخلات المشبوهة",
        "أبرز الإدخالات التي تبدو إشكالية",
    ],
    "judg_10_tone": [
        "أعد صياغة الرسالة بأسلوب أكثر احترافية",
        "اجعل الرسالة تبدو أكثر رسمية واحترافية",
        "راجع صياغة الرسالة لتصبح أكثر احترافية",
    ],
    "judg_11_sentiment": [
        "ميّز ما إذا كان النص إيجابيًا أم سلبيًا أم محايدًا",
        "قيّم النبرة العاطفية العامة للنص",
        "استنتج الشعور السائد المعبَّر عنه في النص",
    ],
    "judg_12_appropriate": [
        "احكم ما إذا كان المحتوى ملائمًا للجمهور المستهدف",
        "قرر مدى ملاءمة هذا المحتوى للجمهور",
        "قيّم ما إذا كانت المادة مناسبة بالنظر إلى الجمهور",
    ],
    "judg_13_urgency": [
        "رتّب التذاكر وفق مدى استعجالها",
        "اجمع التذاكر بحسب درجة الاستعجال",
        "حدد مستوى استعجال كل تذكرة ونظّمها تبعًا لذلك",
    ],
    "judg_14_credibility": [
        "احكم على مدى موثوقية كل مصدر",
        "قدّر درجة الاعتمادية لكل مصدر",
        "حدد مستوى المصداقية لكل مصدر",
    ],
    "judg_15_simplify": [
        "اجعل الشرح أسهل فهمًا لغير المتخصصين",
        "أعد صياغة الشرح بأسلوب مبسط للقراء العاديين",
        "فكّك الشرح ليتمكن القارئ العادي من متابعته",
    ],
}


# --- Spanish paraphrases (Standard/Castilian, Opus 4.6 in-session, 2026-04-11) ---
ES_PARA = {
    "comp_01_sort_asc": [
        "Organiza los elementos de menor a mayor",
        "Coloca los elementos de la lista en orden creciente",
        "Clasifica las entradas de menor a mayor valor",
    ],
    "comp_02_find_max": [
        "Identifica el número más grande de la lista",
        "Determina cuál es el valor más alto entre los elementos",
        "Localiza el valor más elevado dentro de la lista",
    ],
    "comp_03_filter_pos": [
        "Extrae todos los números mayores que cero de la lista",
        "Conserva únicamente los valores positivos y descarta el resto",
        "Selecciona solo las entradas positivas de la lista",
    ],
    "comp_04_reverse": [
        "Voltea la secuencia de los elementos de la lista",
        "Reordena la lista de forma que el último elemento quede primero",
        "Da la vuelta al orden de las entradas de la lista",
    ],
    "comp_05_count": [
        "Determina cuántos elementos contiene la lista",
        "Calcula la cantidad total de entradas en la lista",
        "Mide el tamaño de la lista",
    ],
    "comp_06_sum": [
        "Suma cada número presente en la lista",
        "Obtén el total de todos los valores de la lista",
        "Agrega todas las entradas de la lista para obtener su suma",
    ],
    "comp_07_deduplicate": [
        "Quita los valores repetidos de la lista",
        "Conserva solo los elementos únicos",
        "Descarta cualquier entrada que aparezca más de una vez",
    ],
    "comp_08_top3": [
        "Identifica los tres números más altos de la lista",
        "Selecciona los tres valores mayores de la lista",
        "Determina cuáles son las tres entradas más grandes",
    ],
    "comp_09_mean": [
        "Obtén la media de cada valor de la lista",
        "Halla el promedio aritmético de todas las entradas",
        "Determina el promedio sumando los valores y dividiéndolos por la cantidad",
    ],
    "comp_10_sort_desc": [
        "Organiza los elementos de mayor a menor",
        "Coloca los elementos de la lista en orden decreciente",
        "Clasifica las entradas de mayor a menor valor",
    ],
    "comp_11_concat": [
        "Une las dos cadenas en una sola",
        "Combina ambos textos uno a continuación del otro",
        "Fusiona las dos cadenas añadiendo la segunda al final de la primera",
    ],
    "comp_12_uppercase": [
        "Transforma todas las letras de la cadena en mayúsculas",
        "Cambia cada carácter del texto a mayúscula",
        "Reescribe la cadena entera en letras mayúsculas",
    ],
    "comp_13_split": [
        "Separa el texto en cada espacio para formar una lista de palabras",
        "Parte la cadena usando los espacios en blanco como delimitadores",
        "Rompe el texto en los espacios para obtener palabras individuales",
    ],
    "comp_14_replace": [
        "Sustituye cada 'a' de la cadena por 'b'",
        "Cambia toda instancia de 'a' por 'b' en el texto",
        "Convierte cada 'a' en 'b' dentro del texto dado",
    ],
    "comp_15_length": [
        "Determina cuántos caracteres contiene el texto",
        "Halla el número total de caracteres de la cadena",
        "Cuenta los caracteres que conforman la cadena",
    ],
    "judg_01_prioritize": [
        "Clasifica las tareas según su nivel de importancia",
        "Ordena las tareas para que las más importantes vayan primero",
        "Dispón las tareas de mayor a menor importancia",
    ],
    "judg_02_evaluate": [
        "Valora qué tan bueno es cada candidato",
        "Juzga el nivel de calidad de cada postulante",
        "Aprecia a cada candidato según su calidad global",
    ],
    "judg_03_summarize": [
        "Escribe un resumen breve del documento",
        "Condensa el contenido en una síntesis corta",
        "Proporciona un recuento conciso del documento en varias oraciones",
    ],
    "judg_04_recommend": [
        "Sugiere cuál es la mejor alternativa de la lista",
        "Elige la opción más destacada y explica por qué",
        "Aconseja cuál de las opciones disponibles es la más adecuada",
    ],
    "judg_05_risk": [
        "Valora qué tan riesgoso es cada elemento",
        "Determina el grado de riesgo asociado a cada ítem",
        "Estima el nivel de riesgo para cada entrada",
    ],
    "judg_06_categorize": [
        "Ordena los elementos en categorías adecuadas",
        "Agrupa los elementos según donde mejor encajen",
        "Organiza cada elemento colocándolo en el grupo apropiado",
    ],
    "judg_07_select_key": [
        "Identifica a los clientes más importantes en la base de datos",
        "Extrae los clientes principales de los registros de la base de datos",
        "Escoge a los clientes fundamentales dentro de la base de datos",
    ],
    "judg_08_rank": [
        "Ordena los artículos de más a menos relevante",
        "Dispón los artículos según su relevancia",
        "Clasifica los artículos en función de cuán relevantes resultan",
    ],
    "judg_09_flag": [
        "Señala las entradas que puedan presentar problemas",
        "Identifica y etiqueta las entradas que podrían ser problemáticas",
        "Resalta las entradas que parezcan tener inconvenientes",
    ],
    "judg_10_tone": [
        "Reescribe el mensaje con un tono más profesional",
        "Haz que el mensaje suene más formal y profesional",
        "Modifica la redacción del mensaje para aumentar su profesionalismo",
    ],
    "judg_11_sentiment": [
        "Identifica si el texto es positivo, negativo o neutral",
        "Evalúa el tono emocional general del texto",
        "Descubre el sentimiento predominante expresado en el texto",
    ],
    "judg_12_appropriate": [
        "Juzga si el contenido es adecuado para el público al que va dirigido",
        "Determina si este contenido encaja con la audiencia objetivo",
        "Valora si el material resulta apropiado dado el público",
    ],
    "judg_13_urgency": [
        "Ordena los tickets según cuán urgentes sean",
        "Agrupa los tickets según su urgencia",
        "Asigna un nivel de urgencia a cada ticket y organízalos en consecuencia",
    ],
    "judg_14_credibility": [
        "Valora qué tan confiable es cada fuente",
        "Juzga la fiabilidad de cada una de las fuentes",
        "Determina el nivel de credibilidad de cada fuente",
    ],
    "judg_15_simplify": [
        "Haz la explicación más fácil de entender para no expertos",
        "Reescribe la explicación en términos más simples para el público general",
        "Desglosa la explicación para que cualquier lector pueda seguirla",
    ],
}


# --- Korean Gyeongsang dialect (경상 방언) — Opus 4.6, 2026-04-11 ---
# Limitations: written-form dialect markers only (sentence endings, lexical
# substitutions). No phonological notation. Technical terms (정렬, 리스트, 목록,
# 문자열, 숫자, 평균, 합, 길이) preserved per design. PENDING user spot-check
# — see data/dialect_stimuli_v2_prompts.md.
KO_GYEONGSANG = {
    "comp_01_sort_asc":    "목록을 오름차순으로 정렬하이소",
    "comp_02_find_max":    "목록에서 가장 큰 값 찾아봐라",
    "comp_03_filter_pos":  "목록에서 양수만 골라내뿌라",
    "comp_04_reverse":     "목록 원소 순서를 디비뿌라",
    "comp_05_count":       "목록에 원소 개수 한번 세어 보이소",
    "comp_06_sum":         "목록에 있는 숫자 전부 다 합해 보이소",
    "comp_07_deduplicate": "목록에서 중복된 원소는 치아뿌라",
    "comp_08_top3":        "주어진 목록에서 제일 큰 값 세 개 찾아봐라",
    "comp_09_mean":        "목록에 있는 숫자들 평균 한번 구해 보이소",
    "comp_10_sort_desc":   "목록을 내림차순으로 정렬해 뿌라",
    "comp_11_concat":      "두 문자열을 갖다 붙이뿌라",
    "comp_12_uppercase":   "문자열을 대문자로 바까 뿌라",
    "comp_13_split":       "문자열을 빈칸 기준으로 잘라가 단어 목록 만들어라",
    "comp_14_replace":     "문자열에서 'a'는 다 'b'로 바까뿌라",
    "comp_15_length":      "문자열 길이 좀 구해 보이소",
    "judg_01_prioritize":  "중요한 거부터 순서대로 작업 우선순위 매겨라",
    "judg_02_evaluate":    "후보 하나하나 품질이 어떤지 평가해 보이소",
    "judg_03_summarize":   "문서 내용 몇 마디로 짧게 줄여 보이소",
    "judg_04_recommend":   "목록에서 젤 좋은 거 하나 추천해 보이소",
    "judg_05_risk":        "항목마다 위험 수준이 어떻노 한번 봐라",
    "judg_06_categorize":  "항목들을 알맞은 그룹별로 나눠 뿌라",
    "judg_07_select_key":  "데이터베이스에서 주요 고객 좀 골라 보이소",
    "judg_08_rank":        "관련성 기준으로 기사 순위 한번 매겨라",
    "judg_09_flag":        "문제 있을 만한 항목은 표시해 두이소",
    "judg_10_tone":        "메시지 말투를 더 전문적으로 고쳐 보이소",
    "judg_11_sentiment":   "텍스트 전체 느낌이 어떤지 한번 봐라",
    "judg_12_appropriate": "콘텐츠가 독자한테 맞는지 아닌지 한번 봐라",
    "judg_13_urgency":     "티켓을 얼마나 급한지에 따라 나눠 뿌라",
    "judg_14_credibility": "출처마다 믿을 만한지 아닌지 평가해 보이소",
    "judg_15_simplify":    "일반 사람들 보기 쉽게 설명 좀 풀어 주이소",
}


# --- Egyptian Arabic dialect (اللهجة المصرية) — Opus 4.6, 2026-04-11 ---
# Limitations: NO native-speaker validation in this session. Orthographic
# Egyptian features only (ليستة for قائمة, بتاع, ده/دي, Egyptian lexicon
# where natural). ج kept as-is per Egyptian orthographic convention.
# CAMERA-READY BLOCKER: must be audited by native speaker before publication.
AR_EGYPTIAN = {
    "comp_01_sort_asc":    "رتّب الليستة من الأصغر للأكبر",
    "comp_02_find_max":    "هات أكبر قيمة في الليستة",
    "comp_03_filter_pos":  "طلّع بس الأرقام الموجبة من الليستة",
    "comp_04_reverse":     "اقلب ترتيب العناصر بتاع الليستة",
    "comp_05_count":       "عِدّ العناصر اللي في الليستة",
    "comp_06_sum":         "اجمع كل الأرقام اللي في الليستة",
    "comp_07_deduplicate": "شيل العناصر المكررة من الليستة",
    "comp_08_top3":        "هات أكبر تلات قيم في الليستة دي",
    "comp_09_mean":        "احسب متوسط كل الأرقام اللي في الليستة",
    "comp_10_sort_desc":   "رتّب الليستة من الأكبر للأصغر",
    "comp_11_concat":      "الزق النصّين مع بعض",
    "comp_12_uppercase":   "خلّي النص كله بحروف كبيرة",
    "comp_13_split":       "قسّم النص عند المسافات وطلّع ليستة كلمات",
    "comp_14_replace":     "غيّر كل 'a' في النص لـ 'b'",
    "comp_15_length":      "احسب طول النص ده",
    "judg_01_prioritize":  "رتّب المهام على حسب أهميتها",
    "judg_02_evaluate":    "قيّم جودة كل مرشّح فيهم",
    "judg_03_summarize":   "لخّص المستند في كام جملة",
    "judg_04_recommend":   "اقترح أحسن اختيار من الليستة",
    "judg_05_risk":        "شوف مستوى المخاطر لكل عنصر",
    "judg_06_categorize":  "قسّم العناصر لمجموعات مناسبة",
    "judg_07_select_key":  "اختار العملاء الأساسيين من قاعدة البيانات",
    "judg_08_rank":        "رتّب المقالات على حسب صلتها بالموضوع",
    "judg_09_flag":        "حطّ علامة على الإدخالات اللي ممكن تكون فيها مشكلة",
    "judg_10_tone":        "عدّل نبرة الرسالة علشان تبقى أكتر احترافية",
    "judg_11_sentiment":   "حدّد الإحساس العام بتاع النص",
    "judg_12_appropriate": "شوف المحتوى ده مناسب للجمهور ولا لأ",
    "judg_13_urgency":     "صنّف التذاكر على حسب مستوى الاستعجال",
    "judg_14_credibility": "قيّم مدى مصداقية كل مصدر",
    "judg_15_simplify":    "بسّط الشرح علشان أي حد يفهمه",
}


def build():
    ops = {op.id: op for op in get_all_operations()}

    # English paraphrases come from v1 dialect_stimuli.json (copied, not regenerated).
    v1_path = ROOT / "data" / "dialect_stimuli.json"
    with open(v1_path) as f:
        v1 = json.load(f)
    en_para = {op_id: v1[op_id]["paraphrases"] for op_id in TARGET_IDS}

    para_by_lang = {"en": en_para, "ko": KO_PARA, "zh": ZH_PARA,
                    "ar": AR_PARA, "es": ES_PARA}

    out = {}
    for op_id in TARGET_IDS:
        op = ops[op_id]
        out[op_id] = {
            "original": {lang: op.descriptions.get(lang, "") for lang in ["en", "ko", "zh", "ar", "es"]},
            "paraphrases": {lang: para_by_lang[lang][op_id] for lang in ["en", "ko", "zh", "ar", "es"]},
            "dialects": {
                "ko_gyeongsang": KO_GYEONGSANG[op_id],
                "ar_egyptian": AR_EGYPTIAN[op_id],
            },
        }

    # Sanity: all 30 ops have complete entries
    assert len(out) == 30, f"expected 30 ops, got {len(out)}"
    for op_id in TARGET_IDS:
        entry = out[op_id]
        for lang in ["en", "ko", "zh", "ar", "es"]:
            assert entry["original"][lang], f"{op_id}: missing original {lang}"
            assert len(entry["paraphrases"][lang]) == 3, f"{op_id}: paraphrases[{lang}] != 3"
        assert entry["dialects"]["ko_gyeongsang"], f"{op_id}: missing ko_gyeongsang"
        assert entry["dialects"]["ar_egyptian"], f"{op_id}: missing ar_egyptian"

    out_path = ROOT / "data" / "dialect_stimuli_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")
    print(f"  {len(out)} ops, {sum(len(e['paraphrases'][l]) for e in out.values() for l in ['en','ko','zh','ar','es'])} paraphrases, "
          f"{len(out)*2} dialect entries")


if __name__ == "__main__":
    build()
