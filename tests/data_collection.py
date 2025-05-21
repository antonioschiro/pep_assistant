# Test input codes
inputs = [

    # 1. Simple function: spacing, naming
    "def   sumTwo(x,y):return x+y",

    # 2. Function with loop: indentation, naming, inline logic
    "def printer(items):\n for i in items:print(i)",

    # 3. Mixed tab/space, bad var names
    "def check(val):\n\tif val==True:\n\t print('yes')\n\telse:print('no')",

    # 4. Class: poor naming, no init spacing, no methods
    "class temp:\n def __init__(self,a):self.a=a",

    # 5. Function with default values and type hints: bad formatting
    "def greet(name:str='User',age:int =18)->str:\n return 'Hi '+name",

    # 6. Long line, hardcoded constants, unclear logic
    "def f():print('Total is: '+str(3*100+4*200))",

    # 7. Unclear naming, complex inline logic
    "def do(x):return x*x if x>0 else -x*x",

    # 8. Class with multiple methods: bad naming, missing docstring, compact layout
    "class calc:\n def add(self,x,y):return x+y\n def sub(self,x,y):return x-y",

    # 9. Nested function, lack of spacing, unclear logic
    "def outer(a):\n def inner(b):return a+b\n return inner(5)",

    # 10. Class with inheritance and multiple issues
    "class emp(person):\n def __init__(self,n):\n  super().__init__(n)\n  self.id=0\n def show(self):print(self.id)",

    # 11. Deeply nested loops, non-descriptive function name, no type hints, and modifies input in-place.
    "def Do(data):\n for i in range(len(data)):\n  for j in range(len(data[i])):\n   if data[i][j] == 0:\n    data[i][j] = None",

    # 12. Uses chained `if` instead of cleaner alternatives; no type hints; no zero division handling.
    "def calc(a, b, op='+'):\n if op=='+': return a+b\n elif op=='-': return a-b\n elif op=='*': return a*b\n elif op=='/': return a/b",

    # 13. Uses a mutable default argument which can lead to bugs.
    "def foo(bar=[]):\n bar.append(1)\n return bar",

    # 14. Overuses `lambda` and `map/filter`; less readable and idiomatic.
    "nums = [1,2,3,4,5,6,7,8,9]\nsquares = map(lambda x: x*x, filter(lambda x: x%2==0, nums))\nprint(list(squares))",

    # 15. File not safely handled (no context manager), no encoding specified, no type hint.
    "def f():\n x = open('data.txt').read()\n return x.strip()",

    # 16. Duplicate functionality (`log_info` duplicates `log_event`); uses old-style string formatting.
    "def log_event(event_type, message):\n print('[{}] {}'.format(event_type, message))\n\ndef log_info(msg):\n print('[INFO] {}'.format(msg))",

    # 17. Class name not capitalized; cryptic method and variable names; no type hints.
    "import math\n\nclass circle:\n def __init__(self,r): self.r=r\n def a(self): return math.pi*self.r**2",

    # 18. No input type validation, function name too generic, lacks type hints.
    "def transform(items):\n return [i.strip().lower().replace('-', '_') for i in items if i]",

    # 19. Verbose nested loop that can be simplified with a generator expression.
    "def sum_nested(lst):\n total = 0\n for sub in lst:\n  for val in sub:\n   total += val\n return total",

    # 20. Uses `type()` instead of `isinstance()`, doesn't handle unknown types.
    "def check(x):\n if type(x)==int:\n  print('int')\n elif type(x)==str:\n  print('str')",
    
]

# Ground truth output codes
reference_responses = [
    # 1. Corrected function with proper naming and spacing
    "def sum_two(x, y):\n    return x + y",

    # 2. Corrected loop and indentation
    "def printer(items):\n    for item in items:\n        print(item)",

    # 3. Fixed spacing, naming, and boolean check
    "def check(value):\n    if value is True:\n        print('yes')\n    else:\n        print('no')",

    # 4. Class renamed and formatted
    "class Temperature:\n    def __init__(self, value):\n        self.value = value",

    # 5. Fixed default value spacing and string formatting
    "def greet(name: str = 'User', age: int = 18) -> str:\n    return f'Hi {name}'",

    # 6. Refactored long line with constants and print formatting
    "def calculate_total():\n    item1_price = 3 * 100\n    item2_price = 4 * 200\n    total = item1_price + item2_price\n    print(f'Total is: {total}')",

    # 7. Clearer naming, expanded logic
    "def compute_square_or_negative(x):\n    if x > 0:\n        return x * x\n    else:\n        return -x * x",

    # 8. Class with proper naming, docstring, spacing
    "class Calculator:\n    \"\"\"Simple calculator class with basic operations.\"\"\"\n\n    def add(self, x, y):\n        return x + y\n\n    def subtract(self, x, y):\n        return x - y",

    # 9. Improved readability, spacing
    "def outer(a):\n    def inner(b):\n        return a + b\n\n    return inner(5)",

    # 10. Full class rewrite with proper naming, spacing, docstring
    "class Employee(Person):\n    def __init__(self, name):\n        super().__init__(name)\n        self.employee_id = 0\n\n    def show(self):\n        print(self.employee_id)",

    # 11. Added type hints, renamed function, improved readability with clearer loop structure.
    "def replace_zeros_with_none(matrix: list[list[int]]) -> None:\n    for row in matrix:\n        for i, value in enumerate(row):\n            if value == 0:\n                row[i] = None",

    # 12. Replaced chained ifs with `match-case`, added type hints, handled division by zero.
    "from typing import Union\n\ndef calc(a: float, b: float, op: str = '+') -> Union[float, None]:\n    match op:\n        case '+': return a + b\n        case '-': return a - b\n        case '*': return a * b\n        case '/': return a / b if b != 0 else None\n        case _: return None",

    # 13. Replaced mutable default with None, added type hints, used `Optional`.
    "from typing import Optional\n\ndef foo(bar: Optional[list[int]] = None) -> list[int]:\n    if bar is None:\n        bar = []\n    bar.append(1)\n    return bar",

    # 14. Replaced lambda/map/filter with list comprehension for readability.
    "nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]\neven_squares = [x * x for x in nums if x % 2 == 0]\nprint(even_squares)",

    # 15. Used context manager to safely handle file, added encoding, type hint.
    "def f() -> str:\n    with open('data.txt', 'r', encoding='utf-8') as file:\n        return file.read().strip()",

    # 16. Removed duplicated formatting logic by reusing `log_event`; used f-strings.
    "def log_event(event_type: str, message: str) -> None:\n    print(f'[{event_type}] {message}')\n\ndef log_info(message: str) -> None:\n    log_event('INFO', message)",

    # 17. Capitalized class name, added descriptive method/variable names, type hints.
    "import math\n\nclass Circle:\n    def __init__(self, radius: float):\n        self.radius = radius\n\n    def area(self) -> float:\n        return math.pi * self.radius ** 2",

    # 18. Added type hints, clarified transformation logic and intent.
    "def transform(items: list[str]) -> list[str]:\n    return [item.strip().lower().replace('-', '_') for item in items if item]",

    # 19. Replaced nested loops with a concise generator expression.
    "def sum_nested(lst: list[list[int]]) -> int:\n    return sum(sum(sublist) for sublist in lst)",

    # 20. Used `isinstance()` instead of `type()`, added fallback case.
    "def check(x: object) -> None:\n    if isinstance(x, int):\n        print('int')\n    elif isinstance(x, str):\n        print('str')",

]