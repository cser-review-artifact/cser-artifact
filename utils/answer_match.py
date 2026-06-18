import re
import math
import contextlib

import sympy
from pylatexenc import latex2text
from sympy.parsing import sympy_parser

import utils.math_normalize as math_normalize
import utils.math_verify_box as math_verify_box
from utils.math_lighteval import compute_score as init_compute_score

BAD_SUBSTRINGS = ["^{", "^("]
BAD_REGEXES = ["\^[0-9]+\^", "\^[0-9][0-9]+"]
TUPLE_CHARS = "()[]"

def _sympy_parse(expr: str):
    py_expr = expr.replace("^", "**")
    return sympy_parser.parse_expr(
        py_expr,
        transformations=(sympy_parser.standard_transformations + (sympy_parser.implicit_multiplication_application,)),
    )

def _parse_latex(expr: str) -> str:
    expr = expr.replace("\\tfrac", "\\frac")
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\frac", " \\frac")
    expr = latex2text.LatexNodes2Text().latex_to_text(expr)

    expr = expr.replace("√", "sqrt")
    expr = expr.replace("π", "pi")
    expr = expr.replace("∞", "inf")
    expr = expr.replace("∪", "U")
    expr = expr.replace("·", "*")
    expr = expr.replace("×", "*")

    return expr.strip()

def _strip_properly_formatted_commas(expr: str):
    p1 = re.compile("(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr

def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except ValueError:
        return False

def _is_int(x: float) -> bool:
    try:
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False

def _is_frac(expr: str) -> bool:
    return bool(re.search(r"^-?[0-9]+.?/0*[1-9][0-9]*.?$", expr))

def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except Exception:
        return False

def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    x = float(x)
    return int(x)

def _inject_implicit_mixed_number(step: str):
    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step

def _normalize(expr: str) -> str:
    if expr is None:
        return None

    m = re.search("^\\\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "day",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "liter",
    ]:
        expr = re.sub(f"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)
    expr = re.sub("\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = re.sub(",\\\\! *", "", expr)
    if _is_float(expr) and _is_int(float(expr)):
        expr = str(int(round(float(expr))))
    if "\\" in expr:
        with contextlib.suppress(Exception):
            expr = _parse_latex(expr)

    expr = re.sub("- *", "-", expr)

    expr = _inject_implicit_mixed_number(expr)

    expr = expr.lower()

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr

def _last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    left_brace_idx = None
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
            if left_brace_idx is None:
                left_brace_idx = i
        elif string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break

        i += 1

    if left_brace_idx is None or right_brace_idx is None:
        return None

    return string[left_brace_idx + 1 : right_brace_idx].strip()

def match_answer(response):
    is_matched = False
    for ans_marker in ["answer:", "answer is", "answers are"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    for ans_marker in ["is answer", "is the answer", "are answers", "are the answers"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[:ans_idx].strip()
            if response.endswith("\n"):
                response = response[:-2]

    ans_boxed = _last_boxed_only_string(response)
    if ans_boxed:
        is_matched = True
        response = ans_boxed

    if ". " in response:
        dot_idx = response.lower().rfind(". ")
        if dot_idx != -1:
            response = response[:dot_idx].strip()

    for ans_marker in ["be ", "is ", "are ", "=", ": ", "get ", "be\n", "is\n", "are\n", ":\n", "get\n"]:
        ans_idx = response.lower().rfind(ans_marker)
        if ans_idx != -1:
            is_matched = True
            response = response[ans_idx + len(ans_marker) :].strip()
            if response.endswith("\n"):
                response = response[:-2]

    is_matched = is_matched if any([c.isdigit() for c in response]) else False
    return is_matched, response

def count_unknown_letters_in_expr(expr: str):
    expr = expr.replace("sqrt", "")
    expr = expr.replace("frac", "")
    letters_in_expr = set([x for x in expr if x.isalpha()])
    return len(letters_in_expr)

def should_allow_eval(expr: str):
    if count_unknown_letters_in_expr(expr) > 2:
        return False

    for bad_string in BAD_SUBSTRINGS:
        if bad_string in expr:
            return False

    return all(re.search(bad_regex, expr) is None for bad_regex in BAD_REGEXES)

def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str):
    are_equal = False
    try:
        expr = f"({ground_truth_normalized})-({given_normalized})"
        if should_allow_eval(expr):
            sympy_diff = _sympy_parse(expr)
            simplified = sympy.simplify(sympy_diff)
            if simplified == 0:
                are_equal = True
    except Exception:
        pass
    return are_equal

def split_tuple(expr: str):
    expr = _strip_properly_formatted_commas(expr)
    if len(expr) == 0:
        return []
    if len(expr) > 2 and expr[0] in TUPLE_CHARS and expr[-1] in TUPLE_CHARS and all([ch not in expr[1:-1] for ch in TUPLE_CHARS]):
        elems = [elem.strip() for elem in expr[1:-1].split(",")]
    else:
        elems = [expr]
    return elems

def grade_answer(given_answer: str, ground_truth: str) -> bool:
    if given_answer is None:
        return False

    ground_truth_normalized_mathd = math_normalize.normalize_answer(ground_truth)
    given_answer_normalized_mathd = math_normalize.normalize_answer(given_answer)

    if ground_truth_normalized_mathd == given_answer_normalized_mathd:
        return True

    ground_truth_normalized = _normalize(ground_truth)
    given_normalized = _normalize(given_answer)

    if ground_truth_normalized is None:
        return False

    if ground_truth_normalized == given_normalized:
        return True

    if len(given_normalized) == 0:
        return False

    ground_truth_elems = split_tuple(ground_truth_normalized)
    given_elems = split_tuple(given_normalized)

    if len(ground_truth_elems) > 1 and (ground_truth_normalized[0] != given_normalized[0] or ground_truth_normalized[-1] != given_normalized[-1]) or len(ground_truth_elems) != len(given_elems):
        is_correct = False
    else:
        for ground_truth_elem, given_elem in zip(ground_truth_elems, given_elems):
            if _is_frac(ground_truth_elem) and _is_frac(given_elem):
                is_correct = ground_truth_elem == given_elem
            elif _str_is_int(ground_truth_elem) != _str_is_int(given_elem):
                is_correct = False
            else:
                try:
                    is_correct = are_equal_under_sympy(ground_truth_elem, given_elem)
                except Exception as e:
                    is_correct = False
                    print(f"Error: {e} from are_equal_under_sympy, {ground_truth_elem}, {given_elem}")
            if not is_correct:
                break

    return is_correct

def compute_score(model_output: str, ground_truth: str, answer: str = "") -> bool:
    original_model_output = model_output[-300:]
    original_ground_truth = ground_truth
    model_output = str(model_output)
    ground_truth = str(ground_truth)

    is_matched, extracted_model_output = match_answer(model_output)
    if "<answer>" and  "</answer>" in model_output:
        try:
            extracted_model_output = model_output.split("<answer>")[1].split("</answer>")[0]
        except:
            extracted_model_output = extracted_model_output
    format_correctness = "\\boxed{" in model_output
    
    if grade_answer(extracted_model_output, ground_truth):
        return True
    
    init_check = init_compute_score(model_output, ground_truth)
    if init_check:
        return True

    is_correct_math_verify = math_verify_box.compute_score(original_model_output, original_ground_truth)
    if is_correct_math_verify:
        return True


if __name__ == "__main__":
    responses = ['### Step 3: Calculate the Total Cost for 20 Packs Betty wants to buy **20 packs** of nuts. Therefore, the total cost is:\[\text{Total cost} = 20 \times \text{Cost of one pack} = 20 \times 100 = \\boxed{\$2000}\]### Final Answer\nBetty will pay \(\\boxed{\$2000}\) for the 20 packs of nuts.', 
                'The problem states that **\"Twice Betty\'s age is the cost of a pack of nuts."** \n So, if Betty is \( 50 \) years old, then:\[\text{Cost of one pack} = 2 \times B = 2 \times 50 = \\boxed{\$100}']

    ground_truths = ['\$2000', '\$100']

    rewards = []
    for i in range(len(responses)):
        rewards.append(compute_score(responses[i], ground_truths[i])[0])
    print(rewards)

