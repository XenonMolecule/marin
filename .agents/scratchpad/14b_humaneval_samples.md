# HumanEval Sample Comparison: Qwen3-14B-Base

Comparing outputs from three models on the same HumanEval problems:

- **Baseline**: Qwen3-14B-Base (no SFT) — 89/164 = 54.3%
- **V3 SFT**: Best V3 extraction config (lr=2e-6, bs=32, wd=0.05, wu=0.0) — 127/164 = 77.4%
- **Resiliparse SFT**: lr=1e-5, bs=64 — 97/164 = 59.1%

## Problem Breakdown

| Category | Count |
|---|---|
| All 3 pass | 66 |
| Both SFT pass, baseline fails | 28 |
| Only V3 passes | 17 |
| Only resiliparse passes | 3 |
| Baseline passes, V3 fails | 7 |
| All 3 fail | 27 |

## Failure Mode Analysis: How Much Is Format vs Genuine Skill?

V3 SFT fixes 45 of the baseline's 75 failures (and introduces 7 regressions = +38 net).

**Of the 45 fixes:**

| Category | Count | % of fixes |
|---|---|---|
| **Format/presentation fixes** | **35** | **78%** |
| Genuine algorithm improvements | 10 | 22% |

**Format fix breakdown:** All 35 are markdown fence contamination — the baseline generates the correct code but appends ` ``` ` or other markdown artifacts that break the test harness. The SFT teaches the model to output clean code and stop.

**Of the 30 unfixed failures:**

| Category | Count |
|---|---|
| Format issues + wrong algorithm | 10 |
| Pure algorithmic failures (clean output, wrong code) | 20 |

**Bottom line:** ~78% of the HumanEval improvement is the SFT teaching the model to stop generating after the code (format/stopping behavior). ~22% (10 problems) represents genuine new coding skill — problems where the baseline's algorithm is wrong and V3 SFT produces a correct one. The base model already knows how to code most of these; it just doesn't know when to stop talking.

If you strip out the format effect, the "real" coding improvement is roughly **54.3% → 60.4%** (10 additional algorithmic solves), not 54.3% → 77.4%. The remaining +17pp is output formatting.

---

## 1. SFT Teaches New Skills (Baseline FAIL → V3 PASS)

These problems show what the SFT training adds.

### HumanEval/2 — Both SFT pass, baseline fails
**Results:** Baseline=FAIL, V3=PASS, Resiliparse=PASS

**Prompt:**
```python


def truncate_number(number: float) -> float:
    """ Given a positive floating point number, it can be decomposed into
    and integer part (largest integer smaller than given number) and decimals
    (leftover part always smaller than 1).

    Return the decimal part of the number.
    >>> truncate_number(3.5)
    0.5
    """

```

**Canonical Solution:**
```python
    return number % 1.0

```

**Baseline Output (FAIL):**
```python
    return number - int(number)
```
```

**V3 SFT Output (PASS):**
```python
    return number - int(number)

```

**Resiliparse Output (PASS):**
```python
    return number - int(number)

```

---

### HumanEval/18 — Both SFT pass, baseline fails
**Results:** Baseline=FAIL, V3=PASS, Resiliparse=PASS

**Prompt:**
```python


def how_many_times(string: str, substring: str) -> int:
    """ Find how many times a given substring can be found in the original string. Count overlaping cases.
    >>> how_many_times('', 'a')
    0
    >>> how_many_times('aaa', 'a')
    3
    >>> how_many_times('aaaa', 'aa')
    3
    """

```

**Canonical Solution:**
```python
    times = 0

    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            times += 1

    return times

```

**Baseline Output (FAIL):**
```python
    count = 0
    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            count += 1
    return count
```
```

**V3 SFT Output (PASS):**
```python
    count = 0
    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            count += 1
    return count

```

**Resiliparse Output (PASS):**
```python
    count = 0
    for i in range(len(string) - len(substring) + 1):
        if string[i:i+len(substring)] == substring:
            count += 1
    return count

```

---

### HumanEval/1 — Only V3 passes
**Results:** Baseline=FAIL, V3=PASS, Resiliparse=FAIL

**Prompt:**
```python
from typing import List


def separate_paren_groups(paren_string: str) -> List[str]:
    """ Input to this function is a string containing multiple groups of nested parentheses. Your goal is to
    separate those group into separate strings and return the list of those.
    Separate groups are balanced (each open brace is properly closed) and not nested within each other
    Ignore any spaces in the input string.
    >>> separate_paren_groups('( ) (( )) (( )( ))')
    ['()', '(())', '(()())']
    """

```

**Canonical Solution:**
```python
    result = []
    current_string = []
    current_depth = 0

    for c in paren_string:
        if c == '(':
            current_depth += 1
            current_string.append(c)
        elif c == ')':
            current_depth -= 1
            current_string.append(c)

            if current_depth == 0:
                result.append(''.join(current_string))
                current_string.clear()
... (3 more lines)
```

**Baseline Output (FAIL):**
```python
    paren_string = paren_string.replace(' ', '')
    stack = []
    result = []
    for char in paren_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if len(stack) == 0:
                result.append('')
            else:
                stack.pop()
                if len(stack) == 0:
                    result.append('')
    return result
```

**V3 SFT Output (PASS):**
```python
    stack = []
    result = []
    current_group = ""
    for char in paren_string:
        if char == "(":
            stack.append(char)
            current_group += char
        elif char == ")":
            if stack:
                stack.pop()
                current_group += char
                if not stack:
                    result.append(current_group)
                    current_group = ""
            else:
                raise ValueError("Unbalanced parentheses")
        elif char == " ":
            continue
        else:
            raise ValueError("Invalid character")
... (4 more lines)
```

**Resiliparse Output (FAIL):**
```python
    stack = []
    result = []
    for char in paren_string:
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
            else:
                result.append(''.join(stack))
                stack = []
    if stack:
        result.append(''.join(stack))
    return result 
```

---

### HumanEval/17 — Only V3 passes
**Results:** Baseline=FAIL, V3=PASS, Resiliparse=FAIL

**Prompt:**
```python
from typing import List


def parse_music(music_string: str) -> List[int]:
    """ Input to this function is a string representing musical notes in a special ASCII format.
    Your task is to parse this string and return list of integers corresponding to how many beats does each
    not last.

    Here is a legend:
    'o' - whole note, lasts four beats
    'o|' - half note, lasts two beats
    '.|' - quater note, lasts one beat

    >>> parse_music('o o| .| o| o| .| .| .| .| o o')
    [4, 2, 1, 2, 2, 1, 1, 1, 1, 4, 4]
... (2 more lines)
```

**Canonical Solution:**
```python
    note_map = {'o': 4, 'o|': 2, '.|': 1}
    return [note_map[x] for x in music_string.split(' ') if x]

```

**Baseline Output (FAIL):**
```python
To solve this problem, we need to parse a string of musical notes represented in a special ASCII format and convert it into a list of integers indicating the duration of each note in beats. Here's a step-by-step approach to achieve this:

1. **Identify the Notes**: The input string contains three types of notes:
   - `'o'` - whole note, lasts four beats
   - `'o|'` - half note, lasts two beats
   - `'.|'` - quarter note, lasts one beat

2. **Split the String**: We need to split the input string into individual notes. Since the notes are separated by spaces, we can use the `split()` method to achieve this.

3. **Map Notes to Beats**: We will create a dictionary to map each note to its corresponding beat duration.

4. **Convert Notes to Beats**: Iterate over the list of notes, look up their duration in the dictionary, and append the result to a new list.

5. **Return the Result**: Finally, return the list of beat durations.

Here's the implementation of the `parse_music` function:

```python
```

**V3 SFT Output (PASS):**
```python
    notes = music_string.split()
    beats = []
    for note in notes:
        if note == 'o':
            beats.append(4)
        elif note == 'o|':
            beats.append(2)
        elif note == '.|':
            beats.append(1)
    return beats 
```

**Resiliparse Output (FAIL):**
```python
To solve this problem, we can use a dictionary to map each note to its corresponding number of beats. Then, we can split the input string into individual notes and use the dictionary to convert each note to its corresponding number of beats. Finally, we can return the list of beats.

Here is the implementation of the parse_music function:

```

---

## 2. Base Model Already Capable (All 3 PASS)

The base model solves these — SFT preserves the ability.

### HumanEval/0 — All 3 pass
**Results:** Baseline=PASS, V3=PASS, Resiliparse=PASS

**Prompt:**
```python
from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """

```

**Canonical Solution:**
```python
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False

```

**Baseline Output (PASS):**
```python
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

```

**V3 SFT Output (PASS):**
```python
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False 
```

**Resiliparse Output (PASS):**
```python
    for idx, elem in enumerate(numbers):
        for idx2, elem2 in enumerate(numbers):
            if idx != idx2:
                distance = abs(elem - elem2)
                if distance < threshold:
                    return True

    return False 
```

---

### HumanEval/3 — All 3 pass
**Results:** Baseline=PASS, V3=PASS, Resiliparse=PASS

**Prompt:**
```python
from typing import List


def below_zero(operations: List[int]) -> bool:
    """ You're given a list of deposit and withdrawal operations on a bank account that starts with
    zero balance. Your task is to detect if at any point the balance of account fallls below zero, and
    at that point function should return True. Otherwise it should return False.
    >>> below_zero([1, 2, 3])
    False
    >>> below_zero([1, 2, -4, 5])
    True
    """

```

**Canonical Solution:**
```python
    balance = 0

    for op in operations:
        balance += op
        if balance < 0:
            return True

    return False

```

**Baseline Output (PASS):**
```python
    balance = 0
    for operation in operations:
        balance += operation
        if balance < 0:
            return True
    return False

```

**V3 SFT Output (PASS):**
```python
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False 
```

**Resiliparse Output (PASS):**
```python
    balance = 0
    for op in operations:
        balance += op
        if balance < 0:
            return True
    return False 
```

---

## 3. SFT Regressions (Baseline PASS → V3 FAIL)

These problems the base model gets right but V3 SFT breaks.

### HumanEval/9 — Baseline passes, V3 fails
**Results:** Baseline=PASS, V3=FAIL, Resiliparse=FAIL

**Prompt:**
```python
from typing import List, Tuple


def rolling_max(numbers: List[int]) -> List[int]:
    """ From a given list of integers, generate a list of rolling maximum element found until given moment
    in the sequence.
    >>> rolling_max([1, 2, 3, 2, 3, 4, 2])
    [1, 2, 3, 3, 3, 4, 4]
    """

```

**Canonical Solution:**
```python
    running_max = None
    result = []

    for n in numbers:
        if running_max is None:
            running_max = n
        else:
            running_max = max(running_max, n)

        result.append(running_max)

    return result

```

**Baseline Output (PASS):**
```python
    max_list = []
    for i in range(len(numbers)):
        max_list.append(max(numbers[:i + 1]))
    return max_list


```

**V3 SFT Output (FAIL):**
```python
    max_so_far = numbers[0]
    max_so_far_list = [max_so_far]
    for number in numbers[1:]:
        if number > max_so_far:
            max_so_far = number
        max_so_far_list.append(max_so_far)
    return max_so_far_list


```

**Resiliparse Output (FAIL):**
```python
    max_so_far = numbers[0]
    max_list = [max_so_far]
    for number in numbers[1:]:
        max_so_far = max(max_so_far, number)
        max_list.append(max_so_far)
    return max_list


```

---

### HumanEval/50 — Baseline passes, V3 fails
**Results:** Baseline=PASS, V3=FAIL, Resiliparse=FAIL

**Prompt:**
```python


def encode_shift(s: str):
    """
    returns encoded string by shifting every character by 5 in the alphabet.
    """
    return "".join([chr(((ord(ch) + 5 - ord("a")) % 26) + ord("a")) for ch in s])


def decode_shift(s: str):
    """
    takes as input string encoded with encode_shift function. Returns decoded string.
    """

```

**Canonical Solution:**
```python
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])

```

**Baseline Output (PASS):**
```python
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])


```

**V3 SFT Output (FAIL):**
```python
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])


assert decode_shift(encode_shift("hi")) == "hi"
assert decode_shift(encode_shift("asdfghjkl")) == "asdfghjkl"
assert decode_shift(encode_shift("gf")) == "gf"
assert decode_shift(encode_shift("et")) == "et"
assert decode_shift(encode_shift("zx")) == "zx"
assert decode_shift(encode_shift("abcdefghijklmnopqrstuvwxyz")) == "abcdefghijklmnopqrstuvwxyz"
assert decode_shift(encode_shift("ABCDEFGHIJKLMNOPQRSTUVWXYZ")) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
assert decode_shift(encode_shift("1234567890")) == "1234567890"
assert decode_shift(encode_shift("!@#$%^&*()")) == "!@#$%^&*()"
assert decode_shift(encode_shift("Hello, World!")) == "Hello, World!"
assert decode_shift(encode_shift("The quick brown fox jumps over the lazy dog.")) == "The quick brown fox jumps over the lazy dog."
assert decode_shift(encode_shift("Lorem ipsum dolor sit amet, consectetur adipiscing elit.")) == "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
assert decode_shift(encode_shift("Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo.")) == "Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo."
assert decode_shift(encode_shift("Nunc eget sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam id sapien vel sapien tincidunt tincidunt. Nulla facilisi. Nullam
```

**Resiliparse Output (FAIL):**
```python
    return "".join([chr(((ord(ch) - 5 - ord("a")) % 26) + ord("a")) for ch in s])



[docs]def encode(s: str):
    """
    returns encoded string by shifting every character by 5 in the alphabet.
    """
    return encode_shift(s)



[docs]def decode(s: str):
    """
    takes as input string encoded with encode_shift function. Returns decoded string.
    """
    return decode_shift(s) 
```

---

## 4. Too Hard for Everyone (All 3 FAIL)

### HumanEval/8 — All 3 fail
**Results:** Baseline=FAIL, V3=FAIL, Resiliparse=FAIL

**Prompt:**
```python
from typing import List, Tuple


def sum_product(numbers: List[int]) -> Tuple[int, int]:
    """ For a given list of integers, return a tuple consisting of a sum and a product of all the integers in a list.
    Empty sum should be equal to 0 and empty product should be equal to 1.
    >>> sum_product([])
    (0, 1)
    >>> sum_product([1, 2, 3, 4])
    (10, 24)
    """

```

**Canonical Solution:**
```python
    sum_value = 0
    prod_value = 1

    for n in numbers:
        sum_value += n
        prod_value *= n
    return sum_value, prod_value

```

**Baseline Output (FAIL):**
```python
    if not numbers:
        return (0, 1)
    else:
        return (sum(numbers), reduce(lambda x, y: x * y, numbers))
```

**V3 SFT Output (FAIL):**
```python
    if not numbers:
        return 0, 1
    else:
        return sum(numbers), reduce(lambda x, y: x * y, numbers) 
```

**Resiliparse Output (FAIL):**
```python
    if not numbers:
        return 0, 1
    else:
        return sum(numbers), reduce(lambda x, y: x * y, numbers) 
```

---

### HumanEval/39 — All 3 fail
**Results:** Baseline=FAIL, V3=FAIL, Resiliparse=FAIL

**Prompt:**
```python


def prime_fib(n: int):
    """
    prime_fib returns n-th number that is a Fibonacci number and it's also prime.
    >>> prime_fib(1)
    2
    >>> prime_fib(2)
    3
    >>> prime_fib(3)
    5
    >>> prime_fib(4)
    13
    >>> prime_fib(5)
    89
... (2 more lines)
```

**Canonical Solution:**
```python
    import math

    def is_prime(p):
        if p < 2:
            return False
        for k in range(2, min(int(math.sqrt(p)) + 1, p - 1)):
            if p % k == 0:
                return False
        return True
    f = [0, 1]
    while True:
        f.append(f[-1] + f[-2])
        if is_prime(f[-1]):
            n -= 1
        if n == 0:
... (2 more lines)
```

**Baseline Output (FAIL):**
```python
    count = 0
    fib_num = 0
    while count < n:
        fib_num += 1
        if is_prime(fib_num) and is_fibonacci(fib_num):
            count += 1
    return fib_num
```

```

**V3 SFT Output (FAIL):**
```python
    count = 0
    fib_num = 0
    i = 2
    while True:
        fib_num = fibonacci(i)
        if is_prime(fib_num):
            count += 1
            if count == n:
                return fib_num
        i += 1

```

**Resiliparse Output (FAIL):**
```python
    count = 0
    curr = 1
    prev = 0
    while True:
        temp = curr
        curr += prev
        prev = temp
        if is_prime(curr):
            count += 1
            if count == n:
                return curr

The above code is correct, but it is not very efficient. It takes a long time to compute the 10th prime Fibonacci number. We can improve the efficiency of the code by using a more efficient algorithm to check if a number is prime.

Here is an improved version of the code:

from math import sqrt

```

---

## 5. Resiliparse Unique Wins

### HumanEval/32 — Only resiliparse passes
**Results:** Baseline=FAIL, V3=FAIL, Resiliparse=PASS

**Prompt:**
```python
import math


def poly(xs: list, x: float):
    """
    Evaluates polynomial with coefficients xs at point x.
    return xs[0] + xs[1] * x + xs[1] * x^2 + .... xs[n] * x^n
    """
    return sum([coeff * math.pow(x, i) for i, coeff in enumerate(xs)])


def find_zero(xs: list):
    """ xs are coefficients of a polynomial.
    find_zero find x such that poly(x) = 0.
    find_zero returns only only zero point, even if there are many.
... (9 more lines)
```

**Canonical Solution:**
```python
    begin, end = -1., 1.
    while poly(xs, begin) * poly(xs, end) > 0:
        begin *= 2.0
        end *= 2.0
    while end - begin > 1e-10:
        center = (begin + end) / 2.0
        if poly(xs, center) * poly(xs, begin) > 0:
            begin = center
        else:
            end = center
    return begin

```

**Baseline Output (FAIL):**
```python
    # Your code here
    if len(xs) % 2 != 0:
        raise ValueError("xs must have an even number of coefficients")
    if xs[-1] == 0:
        raise ValueError("xs must have a non-zero coefficient")
    a = xs[-1]
    b = xs[-2]
    c = xs[-3]
    d = xs[-4]
    e = xs[-5]
    f = xs[-6]
    g = xs[-7]
    h = xs[-8]
    i = xs[-9]
    j = xs[-10]
    k = xs[-11]
    l = xs[-12]
    m = xs[-13]
    n = xs[-14]
    o = xs[-15]
... (105 more lines)
```

**V3 SFT Output (FAIL):**
```python
    if len(xs) % 2 != 0:
        raise ValueError("xs must have even number of coefficients")
    if xs[-1] == 0:
        raise ValueError("xs must have largest non zero coefficient")
    x = 0
    while poly(xs, x) != 0:
        x += 0.0001
    return round(x, 2) 
```

**Resiliparse Output (PASS):**
```python
    if len(xs) % 2 != 0:
        raise ValueError("xs must have an even number of coefficients")
    if xs[-1] == 0:
        raise ValueError("xs must have a non-zero largest coefficient")

    # Use the bisection method to find the zero of the polynomial
    a, b = -100, 100
    while abs(b - a) > 1e-10:
        mid = (a + b) / 2
        if poly(xs, mid) * poly(xs, a) <= 0:
            b = mid
        else:
            a = mid
    return a 
```

---
