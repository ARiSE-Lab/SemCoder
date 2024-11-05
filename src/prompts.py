# NL2Code
NL2CODE_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable <Code> according to <NL_Description>

<NL_Description>
{instruction}

<Code>
{response}"""

# Used to format the `{instruction}` part above
SRC_INSTRUCT_INSTRUCTION_PROMPT = """Write a solution to the following coding problem:
{problem}"""

# Used to format src-instruct data points
SRC_INSTRUCT_ILLUSTRATION_PROMPT = """[Problem]
{problem}

[Solution]
{solution}"""


# Execution Deduction
EXEC_I_PROMPT = """Deduce the Semantic Constraints: You are given a Python program and its expected output. Find one input such that executing the program with the input leads to the given output. Complete the assertion with one such input in between [ANSWER] and [/ANSWER].
{instruction}
{response}"""

# Execution Simulation
EXEC_O_PROMPT = """Simulate the Execution: You are given a Python function and an assertion containing a function input. Complete the assertion containing the execution output corresponding to the given input in [ANSWER] and [/ANSWER] tags.
{instruction}
{response}"""


# Refinement

REFINE_PROMPT = """Debug and Refine the Code: You are given a a problem to be implemented in Python, and a buggy code that tries to solve the problem but fails the test case.
You should firstly simulate the execution with the buggy code and the failed test to identify the root cause of the failure. 
Then, you should fix the bug and wrap the refined code in between [Refined] and [/Refined].
{instruction}
{response}"""


